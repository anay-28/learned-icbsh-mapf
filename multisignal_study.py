"""MS8: official data, resumable isolated jobs, pilot and frozen evaluation.

Use `python -m multisignal_study --help`. All writes are local. No remote publish.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path('results/multisignal_v1')
DATA = Path('data/multisignal')
SEED = 20261104


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def source_manifest():
    files = sorted(Path('mapf').glob('*.py')) + sorted(Path('.').glob('multisignal*.py'))
    return {p.as_posix(): sha(p) for p in files}


def prepare():
    from mapf.grid import GridMap
    from mapf.movingai import parse_scen, first_k
    from mapf.low_level import compute_bfs_h_table
    map_path = DATA / 'empty-8-8.map'
    lines = map_path.read_text().splitlines()
    if lines[:4] != ['type octile', 'height 8', 'width 8', 'map'] or lines[4:] != ['.' * 8] * 8:
        raise ValueError('Unexpected official map contents')
    grid = GridMap.from_file(map_path)
    groups, instances, seen = [], [], {}
    for kind in ('even', 'random'):
        files = sorted((DATA / ('scen-' + kind)).glob('*.scen'))
        if len(files) != 25:
            raise ValueError('Expected 25 official files per category')
        random.Random(SEED + (kind == 'random')).shuffle(files)
        splits = ['pilot'] * 2 + ['train'] * 13 + ['validation'] * 4 + ['test'] * 6
        for file, split in zip(files, splits):
            family = file.stem
            rows = parse_scen(file)
            groups.append(dict(family_id=family, kind=kind, split=split,
                               scenario_path=file.as_posix(), scenario_sha256=sha(file)))
            for k in (4, 8, 12, 16, 24):
                starts, goals = first_k(rows, k, grid, map_path.name)
                if any(s not in compute_bfs_h_table(grid, g) for s, g in zip(starts, goals)):
                    raise ValueError('Unreachable official assignment')
                fingerprint = digest(sorted(zip(starts, goals)))
                if fingerprint in seen:
                    raise ValueError(f'Duplicate instance {family} and {seen[fingerprint]}')
                seen[fingerprint] = family
                instances.append(dict(id=f'{family}-k{k}', family_id=family, kind=kind,
                                      split=split, agents=k, starts=starts, goals=goals,
                                      fingerprint=fingerprint))
    archives = {p.name: dict(sha256=sha(p), url='https://www.movingai.com/benchmarks/mapf/' + p.name)
                for p in sorted((DATA / 'archives').glob('*.zip'))}
    manifest = dict(schema='ms8-input-v1', created_utc=utc(), seed=SEED,
                    map_path=map_path.as_posix(), map_sha256=sha(map_path), archives=archives,
                    families=groups, instances=instances,
                    split_counts=dict(Counter(g['split'] for g in groups)))
    write_new(ROOT / 'manifest.json', manifest)
    print(json.dumps(manifest['split_counts']), flush=True)


def policy_selector(policy, model_path=None, tau=None):
    from mapf.baselines import f_aware_residual_selector
    from mapf.ml_revision_policy import RevisionSelector, SavedLinear
    if policy == 'plain':
        return None
    if policy == 'residual':
        return f_aware_residual_selector
    if policy == 'pool_random':
        return RevisionSelector('pool_random', seed=71)
    if policy == 'model_b':
        return RevisionSelector('model_b', SavedLinear('results/recovery/ml_revision/model_b.json'))
    if policy in ('confirmed', 'confirmed_aux'):
        from mapf.multisignal_model import load_model, MultisignalSelector
        return MultisignalSelector(load_model(model_path), tau=tau)
    raise ValueError(policy)


def worker(job_path, output):
    job = read(job_path)
    if source_manifest() != job['source']:
        raise ValueError('Source changed after job creation')
    if job.get('model_path') and sha(job['model_path']) != job['model_sha256']:
        raise ValueError('Frozen model changed')
    manifest = read(ROOT / 'manifest.json')
    if sha(manifest['map_path']) != manifest['map_sha256']:
        raise ValueError('Map hash changed')
    from mapf.grid import GridMap
    from mapf.icbs_h import ICBShSolver
    from mapf.validation import validate_solution
    from mapf.low_level import get_astar_call_count
    instance = job['instance']
    grid = GridMap.from_file(manifest['map_path'])
    starts = list(map(tuple, instance['starts']))
    goals = list(map(tuple, instance['goals']))
    began = time.perf_counter()
    cpu = time.process_time()
    result = dict(job_id=job['job_id'], started_utc=utc(), job_sha256=digest(job))
    try:
        if job['task'] == 'collect':
            from mapf.multisignal_teacher import collect_instance
            result['data'] = collect_instance(grid, starts, goals,
                source_policy=job['policy'], timeout=job['timeout'],
                max_expansions=job['max_expansions'], decision_indices=(0, 3, 15))
        elif job['task'] == 'replay':
            from mapf.multisignal_teacher import replay_candidate
            result['data'] = replay_candidate(grid, starts, goals, job['query'],
                job['candidate_index'], timeout=job['timeout'], max_expansions=job['max_expansions'])
        elif job['task'] == 'solve':
            selector = policy_selector(job['policy'], job.get('model_path'), job.get('tau'))
            progress = {}
            solve_start = time.perf_counter()
            data = ICBShSolver(grid, conflict_selector=selector).solve(starts, goals,
                timeout=job['timeout'], max_expansions=job.get('max_expansions'), progress=progress)
            data['runtime'] = time.perf_counter() - solve_start
            data['observed_hl_expansions'] = progress.get('nodes_expanded', 0)
            data['astar_calls'] = get_astar_call_count()
            try:
                from mapf.low_level import get_astar_expansion_count
                data['ll_expansions'] = get_astar_expansion_count()
            except ImportError:
                pass
            data['solution_validated'] = False
            if data['status'] == 'SOLVED':
                validate_solution(grid, starts, goals, data['paths'], data['cost'])
                data['solution_validated'] = True
            data['selector_stats'] = getattr(selector, 'stats', {})
            result['data'] = data
        else:
            raise ValueError('Unknown job task')
        result['status'] = 'OK'
    except Exception:
        result.update(status='ERROR', traceback=traceback.format_exc())
    result.update(finished_utc=utc(), wall_seconds=time.perf_counter()-began,
                  cpu_seconds=time.process_time()-cpu)
    write_new(output, result)


def execute_job(job, directory):
    """Monitor a fresh process. Complete and failed attempts are immutable."""
    import psutil
    directory = Path(directory)
    path = directory / (job['job_id'] + '.job.json')
    output = directory / (job['job_id'] + '.json')
    monitor = directory / (job['job_id'] + '.monitor.json')
    if monitor.exists():
        previous = read(monitor)
        if previous['job_sha256'] != digest(job):
            raise ValueError('Resume job identity mismatch')
        return previous
    if path.exists():
        if read(path) != job:
            raise ValueError('Existing pending job differs')
    else:
        write_new(path, job)
    if output.exists():
        # Never overwrite evidence after an interrupted parent monitor.
        recovered = dict(job_id=job['job_id'], job_sha256=digest(job),
                         exit_code=None, monitor_status='RECOVERED_OUTPUT',
                         process_wall_seconds=None, peak_rss_bytes=None,
                         output_sha256=sha(output))
        write_new(monitor, recovered)
        return recovered
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONHASHSEED='71', PYTHONIOENCODING='utf-8')
    began = time.perf_counter()
    peak = 0
    status = 'EXITED'
    with (directory / (job['job_id'] + '.log')).open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-m', 'multisignal_study', 'worker',
            '--job', str(path), '--output', str(output)], stdout=log, stderr=log, env=env)
        while process.poll() is None:
            try:
                peak = max(peak, psutil.Process(process.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak > job.get('memory_bytes', 3 * 2**30):
                status = 'MEMORY_LIMIT'
                process.kill()
            elif time.perf_counter() - began > job['timeout'] + 15:
                status = 'EXTERNAL_TIMEOUT'
                process.kill()
            time.sleep(.05)
        code = process.wait()
    result = dict(job_id=job['job_id'], job_sha256=digest(job), exit_code=code,
                  monitor_status=status, process_wall_seconds=time.perf_counter()-began,
                  peak_rss_bytes=peak, finished_utc=utc(),
                  output_sha256=sha(output) if output.exists() else None)
    write_new(monitor, result)
    return result


def make_job(task, instance, policy, timeout, source, **kwargs):
    job = dict(task=task, instance=instance, policy=policy, timeout=timeout,
               source=source, seed=71, manifest_sha256=sha(ROOT / 'manifest.json'), **kwargs)
    job['job_id'] = digest(job)[:24]
    return job


def run_jobs(jobs, directory, workers=1, ceiling=14400):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lock = directory / '.running.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        # Bounded submission keeps the stop deadline meaningful.
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            iterator = iter(jobs)
            while True:
                while len(pending) < workers and time.perf_counter()-started < ceiling:
                    job = next(iterator, None)
                    if job is None:
                        break
                    future = pool.submit(execute_job, job, directory)
                    pending[future] = job
                if not pending:
                    break
                future = next(as_completed(pending))
                future.result()
                del pending[future]
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(json.dumps(dict(directory=str(directory), completed=done,
                                          scheduled=len(jobs), elapsed_seconds=round(time.perf_counter()-started,1))), flush=True)
        return done
    finally:
        lock.unlink()


def jobs_outputs(directory):
    for job_path in sorted(Path(directory).glob('*.job.json')):
        output = job_path.with_name(job_path.name.replace('.job.json', '.json'))
        yield read(job_path), read(output) if output.exists() else None


def pilot(workers):
    manifest = read(ROOT / 'manifest.json')
    source = source_manifest()
    jobs = [make_job('solve', i, p, 8., source) for i in manifest['instances']
            if i['split'] == 'pilot' for p in ('plain', 'residual')]
    random.Random(SEED).shuffle(jobs)
    write_new(ROOT / 'pilot_schedule.json', jobs)
    run_jobs(jobs, ROOT / 'pilot', workers, 600)
    import statistics
    rows = defaultdict(list)
    for job, result in jobs_outputs(ROOT / 'pilot'):
        data = (result or {}).get('data', {})
        rows[job['instance']['agents']].append((data.get('status') == 'SOLVED',
                      min(job['timeout'], data.get('runtime', job['timeout']))))
    summary = {k: dict(runs=len(v), solved=sum(s for s,t in v),
                       median_capped_seconds=statistics.median(t for s,t in v)) for k,v in rows.items()}
    useful = [k for k,v in summary.items() if v['solved'] >= v['runs']/2 and
              v['median_capped_seconds'] >= .05]
    ranked = sorted(useful, key=lambda k: summary[k]['median_capped_seconds'], reverse=True)
    if len(ranked) < 2:
        ranked += [k for k in sorted(summary, reverse=True) if k not in ranked]
    selected = sorted(ranked[:2])
    value = dict(created_utc=utc(), selected_agent_counts=selected, pilot=summary,
                 rule='two hardest counts with >=50% solves and median>=0.05s; highest-count fallback',
                 collection=dict(timeout=15., max_expansions=4096, decision_indices=[0,3,15],
                                 policies=['residual','plain']), source=source)
    write_new(ROOT / 'design.json', value)
    print(json.dumps(value), flush=True)


def collect(workers):
    manifest, design = read(ROOT/'manifest.json'), read(ROOT/'design.json')
    source = source_manifest()
    frozen = ROOT/'collection_source.zip'
    if not frozen.exists():
        with zipfile.ZipFile(frozen, 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in source:
                archive.write(path)
    instances = [i for i in manifest['instances'] if i['split'] in ('train','validation')
                 and i['agents'] in design['selected_agent_counts']]
    jobs = [make_job('collect', i, p, 15., source, max_expansions=4096)
            for i in instances for p in ('residual','plain')]
    random.Random(SEED).shuffle(jobs)
    schedule = ROOT/'collection_schedule.json'
    if not schedule.exists():
        write_new(schedule, jobs)
    elif read(schedule) != jobs:
        raise ValueError('Collection schedule/source changed')
    run_jobs(jobs, ROOT/'collection', workers, 14400)
    replays = []
    for job, output in jobs_outputs(ROOT/'collection'):
        if not output or output.get('status') != 'OK':
            continue
        for query in output['data']['queries']:
            query_id = digest(dict(instance=job['instance']['id'], policy=job['policy'],
                                   decision_index=query['decision_index'], fingerprint=query['fingerprint']))[:24]
            for c in range(len(query['candidates'])):
                replays.append(make_job('replay', job['instance'], job['policy'], 15., source,
                    max_expansions=4096, query=query, query_id=query_id, candidate_index=c))
    random.Random(SEED + 1).shuffle(replays)
    replay_schedule = ROOT/'replay_schedule.json'
    if not replay_schedule.exists():
        write_new(replay_schedule, replays)
    elif read(replay_schedule) != replays:
        raise ValueError('Replay schedule/source changed')
    run_jobs(replays, ROOT/'replays', workers, 14400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','pilot','collect','worker'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--job')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'worker':
        worker(args.job, args.output)
    elif args.command == 'pilot':
        pilot(args.workers)
    elif args.command == 'collect':
        collect(args.workers)


if __name__ == '__main__':
    main()

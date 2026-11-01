"""Serial cold-process final benchmark, with family-first capped-time analysis."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
import psutil
import numpy as np

from cro_protocol import (ROOT, append, artifact, campaign_lock, digest, provenance,
                          read_protocol, readlines, sha, utc, write_new)


def run_isolated(spec, memory_mb=4096):
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / 'input.json', Path(tmp) / 'output.json'
        inp.write_text(json.dumps(spec))
        env = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        began, started = time.perf_counter(), utc()
        proc = subprocess.Popen([sys.executable, '-m', 'mapf.override_benchmark_worker', str(inp), str(out)],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        monitor = psutil.Process(proc.pid)
        peak, cpu, killed = 0, 0., None
        while proc.poll() is None:
            try:
                info = monitor.memory_info()
                peak = max(peak, info.rss)
                times = monitor.cpu_times()
                cpu = max(cpu, times.user + times.system)
                if time.perf_counter() - began >= spec['timeout_seconds']:
                    killed = 'HARD_TIMEOUT'
                elif info.rss > memory_mb * 1024 ** 2:
                    killed = 'MEMORY_LIMIT'
                if killed:
                    proc.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.005)
        _, stderr = proc.communicate()
        elapsed = time.perf_counter() - began
        if killed:
            row = dict(status=killed, cost=None)
        elif proc.returncode or not out.exists():
            row = dict(status='PROCESS_ERROR', cost=None, detail=stderr.decode(errors='replace')[-2000:])
        else:
            row = json.loads(out.read_text())
        row.update(end_to_end_wall_seconds=elapsed, utc_started=started, utc_finished=utc(),
                   process_cpu_seconds=max(cpu, row.get('process_cpu_seconds', 0)),
                   peak_rss_bytes=max(peak, row.pop('process_peak_rss_bytes', 0)),
                   timeout_seconds=spec['timeout_seconds'], model_loading_included=True,
                   cache_mode='cold isolated process', timing_fault=elapsed > spec['timeout_seconds'] + 1,
                   telemetry_note='Hard-killed CPU/RSS are sampled lower bounds; missing selector telemetry is not zero')
        return row


def validate_result(row, instance):
    if row['status'] != 'SOLVED':
        return
    from mapf.grid import load_map, GridMap
    from mapf.validation import validate_solution
    grid = load_map(instance['map_path']) if 'map_path' in instance else GridMap(
        instance['width'], instance['height'], {tuple(p) for p in instance.get('obstacles', [])})
    validate_solution(grid, list(map(tuple, instance['starts'])), list(map(tuple, instance['goals'])),
                      [list(map(tuple, path)) for path in row['paths']], row['cost'])
    if not row.get('solution_validated'):
        raise ValueError('Worker solution was not validated')


def holdout(root, p):
    manifest = json.loads((Path(root) / 'movingai_holdout_manifest.json').read_text())
    if digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError('Holdout manifest modified')
    if manifest['protocol_sha256'] != p['protocol_sha256'] or len(manifest['instances']) != 24:
        raise ValueError('Holdout protocol/dimensions mismatch')
    if sha(Path(root) / 'validation.json') != manifest['validation_sha256']:
        raise ValueError('Validation artifact changed')
    for ins in manifest['instances']:
        if (sha(ins['map_path']) != ins['map_sha256'] or sha(ins['scen_path']) != ins['scen_sha256']
                or digest({k: v for k, v in ins.items() if k != 'instance_sha256'}) != ins['instance_sha256']):
            raise ValueError('Holdout map, scenario or instance changed')
    return manifest


def freeze_evaluation(root=ROOT):
    root = Path(root)
    p = read_protocol(root)
    validation = json.loads((root / 'validation.json').read_text())
    if not validation['gate_passed'] or p['smoke']:
        raise ValueError('No confirmatory deployment after failed validation gate')
    manifest = holdout(root, p)
    paths = dict(model_b='results/recovery/ml_revision/model_b.json', override=str(root / 'model_logistic.json'))
    if sha(paths['override']) != manifest['override_model_sha256']:
        raise ValueError('Model changed after holdout freeze')
    config = p['evaluation']
    jobs = [dict(instance_id=i['instance_id'], family_id=i['family_id'], map_name=i['map_name'],
                 k=i['k'], repeat=r, policy=policy) for i in manifest['instances']
            for r in range(config['repetitions']) for policy in config['policies']]
    random.Random(config['seed']).shuffle(jobs)
    for index, job in enumerate(jobs):
        job.update(job_id=f'job-{index:04d}', schedule_index=index)
    schedule = dict(**provenance(p), frozen_utc=utc(), config=config, jobs=jobs,
                    holdout_sha256=sha(root / 'movingai_holdout_manifest.json'), model_paths=paths,
                    model_sha256={name: sha(path) for name, path in paths.items()},
                    model_metadata_sha256={name: sha(path + '.metadata.json') for name, path in paths.items()},
                    timing_rule='Preserve every run; any timing fault invalidates the speedup claim; no outcome-driven reruns')
    schedule['schedule_sha256'] = digest(schedule)
    write_new(root / 'evaluation_schedule.json', schedule)
    return schedule


def verify_evaluation(root, require_complete=False):
    root = Path(root)
    p = read_protocol(root)
    manifest = holdout(root, p)
    schedule = json.loads((root / 'evaluation_schedule.json').read_text())
    if digest({k: v for k, v in schedule.items() if k != 'schedule_sha256'}) != schedule['schedule_sha256']:
        raise ValueError('Evaluation schedule modified')
    if schedule['holdout_sha256'] != sha(root / 'movingai_holdout_manifest.json'):
        raise ValueError('Evaluation holdout changed')
    for name, path in schedule['model_paths'].items():
        if sha(path) != schedule['model_sha256'][name] or sha(path + '.metadata.json') != schedule['model_metadata_sha256'][name]:
            raise ValueError('Evaluation model changed')
    instances = {i['instance_id']: i for i in manifest['instances']}
    jobs = {j['job_id']: j for j in schedule['jobs']}
    rows = readlines(root / 'evaluation_runs.jsonl')
    if len(jobs) != 192 or len({r['job_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid/duplicate final schedule or runs')
    if require_complete and {r['job_id'] for r in rows} != set(jobs):
        raise ValueError('Final matrix incomplete')
    for row in rows:
        if row['job_id'] not in jobs or any(row[k] != v for k, v in jobs[row['job_id']].items()):
            raise ValueError('Run not in frozen schedule')
        if digest({k: v for k, v in row.items() if k != 'record_sha256'}) != row['record_sha256']:
            raise ValueError('Final result fingerprint mismatch')
        if row['schedule_sha256'] != schedule['schedule_sha256'] or row['source_sha256'] != p['source_sha256']:
            raise ValueError('Mixed final evidence')
        validate_result(row, instances[row['instance_id']])
    by_instance = defaultdict(set)
    for row in rows:
        if row['status'] == 'SOLVED':
            by_instance[row['instance_id']].add(row['cost'])
    if any(len(costs) != 1 for costs in by_instance.values()):
        raise ValueError('Final policies disagree on solution cost')
    return p, schedule, instances, rows


def run(root=ROOT):
    root = Path(root)
    p, schedule, instances, rows = verify_evaluation(root)
    done = {r['job_id'] for r in rows}
    for job in schedule['jobs']:
        if job['job_id'] in done:
            continue
        spec = dict(instance=instances[job['instance_id']], policy=job['policy'], seed=p['evaluation']['seed'],
                    timeout_seconds=p['evaluation']['timeout_seconds'], model_paths=schedule['model_paths'])
        row = run_isolated(spec, p['evaluation']['memory_mb'])
        row.update(**job, **provenance(p), schedule_sha256=schedule['schedule_sha256'])
        row['record_sha256'] = digest(row)
        append(root / 'evaluation_runs.jsonl', row)
        validate_result(row, instances[job['instance_id']])
        done.add(job['job_id'])
        print(json.dumps(dict(completed=len(done), total=192, job_id=job['job_id'], policy=job['policy'],
                              status=row['status'], wall=row['end_to_end_wall_seconds'])), flush=True)
        if row['status'] not in ('SOLVED', 'TIMEOUT', 'HARD_TIMEOUT', 'MEMORY_LIMIT', 'HORIZON_LIMIT'):
            raise ValueError('Unexpected worker error retained; investigate without test-driven tuning')
    verify_evaluation(root, require_complete=True)


def capped(row, cap=10):
    return min(row['end_to_end_wall_seconds'], cap) if row['status'] == 'SOLVED' else cap


def family_ratios(rows, policy='override', baseline='residual', cap=10):
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        families[(row['map_name'], row['family_id'])][row['policy']].append(capped(row, cap))
    return [dict(map_name=m, family_id=f, policy_seconds=statistics.mean(v[policy]),
                 residual_seconds=statistics.mean(v[baseline]),
                 ratio=statistics.mean(v[policy]) / statistics.mean(v[baseline]))
            for (m, f), v in sorted(families.items())]


def stratified_interval(families, repetitions=10000, seed=20261101):
    strata = defaultdict(list)
    for f in families:
        strata[f['map_name']].append(math.log(f['ratio']))
    if not strata:
        raise ValueError('No final family ratios')
    rng = np.random.default_rng(seed)
    sums = np.zeros(repetitions)
    for _, values in sorted(strata.items()):
        a = np.asarray(values)
        sums += a[rng.integers(len(a), size=(repetitions, len(a)))].sum(axis=1)
    return np.quantile(np.exp(sums / len(families)), [.025, .975]).tolist()


def solved_summary(rows, policy):
    group = [r for r in rows if r['policy'] == policy]
    by_instance = defaultdict(list)
    for r in group:
        by_instance[r['instance_id']].append(r)
    return dict(solved_runs=sum(r['status'] == 'SOLVED' for r in group), total_runs=len(group),
                instances_solved_both=sum(len(v) == 2 and all(r['status'] == 'SOLVED' for r in v) for v in by_instance.values()))


def analyze(root=ROOT):
    root = Path(root)
    p, _, _, rows = verify_evaluation(root, require_complete=True)
    coverage = {name: solved_summary(rows, name) for name in p['evaluation']['policies']}
    families = family_ratios(rows)
    estimate = math.exp(statistics.mean(math.log(f['ratio']) for f in families))
    interval = stratified_interval(families)
    coverage_ok = all(coverage['override'][k] >= coverage['residual'][k] for k in ('solved_runs', 'instances_solved_both'))
    timing_ok = not any(r['timing_fault'] for r in rows)
    permitted = interval[1] < 1 and coverage_ok and timing_ok
    matched = []
    by_key = {(r['instance_id'], r['repeat'], r['policy']): r for r in rows}
    for instance in sorted({r['instance_id'] for r in rows}):
        group = [by_key[(instance, r, policy)] for r in (0, 1) for policy in ('residual', 'override')]
        if not all(r['status'] == 'SOLVED' for r in group):
            continue
        matched.append(dict(instance_id=instance, values={policy: {
            name: statistics.mean(by_key[(instance, r, policy)][name] for r in (0, 1))
            for name in ('nodes_expanded', 'astar_calls', 'peak_rss_bytes')} for policy in ('residual', 'override')}))
    summary = dict(**provenance(p), runs_sha256=sha(root / 'evaluation_runs.jsonl'),
                   coverage=coverage, family_ratios=families, geometric_mean_ratio=estimate,
                   bootstrap_95=interval, bootstrap_replicates=10000, bootstrap_seed=20261101,
                   timing_valid=timing_ok, coverage_gate=coverage_ok, established_speedup=permitted,
                   conclusion=('Established speedup on this frozen test' if permitted else
                               'No statistically established speedup relative to residual.'),
                   matched_solved_instances=matched, conditional_efficiency_warning='Matched-solved expansion/A*/RSS comparisons have selection bias',
                   selector_diagnostics=[dict(job_id=r['job_id'], policy=r['policy'], stats=r.get('selector_stats'))
                                         for r in rows if r['policy'] in ('override', 'model_b')])
    artifact(root, 'evaluation_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('coverage', 'geometric_mean_ratio', 'bootstrap_95', 'conclusion')}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'run', 'analyze'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--timeout', type=int, choices=[10], default=10)
    parser.add_argument('--repetitions', type=int, choices=[2], default=2)
    parser.add_argument('--seed', type=int, choices=[71], default=71)
    parser.add_argument('--bootstrap-replicates', type=int, choices=[10000], default=10000)
    parser.add_argument('--bootstrap-seed', type=int, choices=[20261101], default=20261101)
    args = parser.parse_args()
    with campaign_lock(args.root):
        {'freeze': freeze_evaluation, 'run': run, 'analyze': analyze}[args.stage](args.root)


if __name__ == '__main__':
    main()

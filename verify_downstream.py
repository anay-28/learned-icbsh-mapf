"""Independent audit of hashes, replay, solved paths and strict pair evidence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import zipfile
import shutil
import subprocess
import sys
import tempfile
import numpy as np

from cro_protocol import ROOT, artifact, digest, provenance, read_protocol, readlines, sha, write_new
from collect_downstream import (candidate_id, evidence_index, histories_for, selected_records, snapshots)
from analyze_downstream import latest_stage, load_queries, headroom_summary
from mapf.day3_calibration import restore
from mapf.ml_revision import replay
from mapf.override_features import ORDERED_PAIRS, symmetrize
from mapf.features import FEATURE_NAMES
from mapf.validation import validate_solution, validate_constrained_paths
from mapf.cardinal import build_constraint_for_agent


def verify_legacy(root=ROOT):
    """Run the unmodified old suite on byte-identical historical files only.

    Its historical source hash glob includes every mapf/*.py, so additive new
    modules intentionally change directory membership. An isolated copy retains
    the old checksum's scope without editing that test or the frozen hash.
    """
    root = Path(root)
    p = read_protocol(root)
    with tempfile.TemporaryDirectory() as tmp:
        for name, expected in p['historical_files'].items():
            if sha(name) != expected:
                raise ValueError('Historical source changed')
            target = Path(tmp) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        result = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'mapf.test_recovery',
                                 'mapf.test_day3', 'mapf.test_repair', 'mapf.test_ml_revision', 'test_final_study'],
                                cwd=tmp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        submission = subprocess.run([sys.executable, '-m', 'verify_submission'], cwd=tmp, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path = root / 'legacy_regressions.txt'
    if path.exists():
        raise FileExistsError('Legacy regression evidence already retained')
    with path.open('x') as stream:
        stream.write(result.stdout)
        stream.write('\nHistorical submission verification:\n' + submission.stdout)
    artifact(root, 'legacy_regressions.json', dict(**provenance(p), returncode=result.returncode,
             submission_returncode=submission.returncode,
             log_sha256=sha(path), scope='Unchanged historical files only; old checksum excludes the additive BDRO modules'))
    print(result.stdout)
    print(submission.stdout)
    if result.returncode or submission.returncode:
        raise ValueError('Historical regression suite failed')


def independent_relation(a, b):
    """No generation helper: compare exact completion or observed bounds directly."""
    done_a, done_b = a['status'] in ('SOLVED', 'INFEASIBLE'), b['status'] in ('SOLVED', 'INFEASIBLE')
    if done_a and done_b:
        return ('RESIDUAL_BETTER' if a['expanded'] < b['expanded'] else
                'CHALLENGER_BETTER' if b['expanded'] < a['expanded'] else 'TIE')
    if done_a and a['expanded'] < b['expanded']:
        return 'RESIDUAL_BETTER'
    if done_b and b['expanded'] < a['expanded']:
        return 'CHALLENGER_BETTER'
    return 'UNKNOWN'


def validate_rollout(row, record, env):
    grid, starts, goals, node, _, _, chosen = env
    i = row['candidate_index']
    candidate = record['candidates'][i]
    if (row['candidate_id'] != candidate_id(candidate) or row['conflict'] != candidate['conflict']
            or row['state_sha256'] != record['state_sha256']
            or row['analytical_key'] != candidate['analytical_key']
            or row['features_raw'] != candidate['features']
            or row['family_id'] != record['instance']['family_id']
            or row['split'] != record['instance']['split']):
        raise ValueError('Rollout candidate/state metadata mismatch')
    expected_hash = row['record_sha256']
    if digest({k: v for k, v in row.items() if k != 'record_sha256'}) != expected_hash:
        raise ValueError('Rollout record fingerprint mismatch')
    e = row['evidence']
    if not e['forced_split_applied'] or row['budget'] != e['budget']:
        raise ValueError('First conflict not forced or budget mismatch')
    budget = {'q32': 32, 'q128': 128, 'q512': 512}[row['budget']]
    if not 0 <= e['expanded'] <= budget or e['wall_seconds'] < 0 or e['cpu_seconds'] < 0:
        raise ValueError('Invalid measured work')
    if e['status'] == 'EXPANSION_CAP' and e['expanded'] != budget:
        raise ValueError('Expansion-cap evidence does not reach the cap')
    np.testing.assert_array_equal(symmetrize(candidate['features']), row['features_symmetric'])
    if e['status'] != 'SOLVED':
        if e.get('solution_paths') is not None or e.get('solution_cost') is not None:
            raise ValueError('Censored/infeasible run contains invented solution')
        return 0
    if not e['solution_validated']:
        raise ValueError('Missing solution-validation marker')
    paths = [list(map(tuple, path)) for path in e['solution_paths']]
    validate_solution(grid, starts, goals, paths, e['solution_cost'])
    validate_constrained_paths(grid, starts, goals, paths, node.constraints)
    # Returned solution must belong to at least one ordinary child of the requested split.
    split_satisfied = False
    for agent in (chosen[i].conflict.agent_i, chosen[i].conflict.agent_j):
        cs = list(node.constraints)
        cs[agent] = build_constraint_for_agent(chosen[i].conflict, agent, cs[agent])
        try:
            validate_constrained_paths(grid, starts, goals, paths, cs)
            split_satisfied = True
        except ValueError:
            pass
    if not split_satisfied:
        raise ValueError('Solution satisfies neither branch of the forced conflict')
    return 1


def verify(root=ROOT, final=False):
    root = Path(root)
    p = read_protocol(root)
    source = json.loads((root / 'source.json').read_text())
    if source['source_sha256'] != p['source_sha256'] or sha(root / 'source.zip') != source['archive_sha256']:
        raise ValueError('Frozen source archive mismatch')
    with zipfile.ZipFile(root / 'source.zip') as archive:
        import hashlib
        if set(archive.namelist()) != set(p['source_files']):
            raise ValueError('Source archive membership changed')
        for name, expected in p['source_files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Archived source hash mismatch')
    records = selected_records(p)
    selection = json.loads((root / 'selection.json').read_text())
    ids = set(selection['query_ids'])
    records = [r for r in records if r['query_id'] in ids]
    if len(records) != len(ids) or len(ids) != len(selection['query_ids']):
        raise ValueError('Invalid query selection')
    if not p['smoke'] and ids != set(p['input_query_ids']):
        raise ValueError('Full study dropped input states')
    frozen_input = json.loads((root / 'inputs_manifest.json').read_text())
    manifest = {r['query_id']: r for r in frozen_input['records']}
    old_test = json.loads(Path('results/recovery/day3/sealed_test_manifest.json').read_text())
    test_ids = {i['family_id'] for i in old_test['instances']}
    if test_ids & {r['instance']['family_id'] for r in records}:
        raise ValueError('Old test family in training/development states')
    rows = readlines(root / 'rollouts.jsonl')
    index = evidence_index(rows, p)
    if any(r['query_id'] not in ids for r in rows):
        raise ValueError('Unexpected rollout query')
    jobs = readlines(root / 'rollout_jobs.jsonl')
    job_keys = {(r['query_id'], r['candidate_id'], r['budget']) for r in jobs}
    if len(job_keys) != len(jobs) or job_keys != set(index):
        raise ValueError('Rollout job/result matrix is incomplete')
    validated, replayed = 0, 0
    for r in records:
        if digest(r) != manifest[r['query_id']]['record_sha256']:
            raise ValueError('Frozen input record changed')
        replay(r)
        env = restore(r)
        replayed += 1
        costs, statuses = set(), set()
        for c in r['candidates']:
            raw = np.asarray(c['features'], dtype=float)
            swapped = raw.copy()
            for a, b in ORDERED_PAIRS:
                i, j = FEATURE_NAMES.index(a), FEATURE_NAMES.index(b)
                swapped[i], swapped[j] = raw[j], raw[i]
            np.testing.assert_array_equal(symmetrize(raw), symmetrize(swapped))
        for row in rows:
            if row['query_id'] != r['query_id']:
                continue
            validated += validate_rollout(row, r, env)
            e = row['evidence']
            if e['status'] in ('SOLVED', 'INFEASIBLE'):
                statuses.add(e['status'])
            if e['status'] == 'SOLVED':
                costs.add(e['solution_cost'])
        if len(statuses) > 1 or len(costs) > 1:
            raise ValueError('Complete continuations disagree on solvability/cost')
    stage = latest_stage(root)
    expected_queries, expected_pairs = snapshots(records, index, p, stage)
    expected_queries.sort(key=lambda r: r['query_id'])
    expected_pairs.sort(key=lambda r: (r['query_id'], r['candidate_index']))
    queries = load_queries(root, stage)
    if queries != expected_queries or json.loads((root / f'pairs_{stage}.json').read_text()) != expected_pairs:
        raise ValueError('Snapshots cannot be regenerated from raw rollout evidence')
    for q in queries:
        for pair in q['pairs']:
            relation = independent_relation(pair['residual_evidence'], pair['challenger_evidence'])
            label = {'CHALLENGER_BETTER': 1, 'RESIDUAL_BETTER': 0}.get(relation)
            if pair['relation'] != relation or pair['label'] != label:
                raise ValueError('Pair label unsupported by raw work evidence')
    if any(p['label'] is None for p in expected_pairs):
        raise ValueError('Tie/unknown included in strict training pairs')
    summary = headroom_summary(queries, p['gates'])
    quality = dict(**provenance(p), stage=stage, integrity_passed=True, replay_mismatches=0,
                   invalid_solutions=0, queries_replayed=replayed, rollouts=len(rows),
                   solved_rollouts_validated=validated, forced_splits_applied=len(rows),
                   strict_pairs=len(expected_pairs), raw_rollouts_sha256=sha(root / 'rollouts.jsonl'),
                   queries_sha256=sha(root / f'queries_{stage}.json'),
                   headroom_gate_passed=summary['gate_passed'], headroom_checks=summary['checks'])
    artifact(root, f'quality_{stage}.json', quality)
    if final or stage == 'q512' or (stage == 'q128' and not summary['gate_passed']) or p['smoke']:
        artifact(root, 'quality.json', quality)
        artifact(root, 'headroom.json', dict(**provenance(p), stage=stage, **summary))
        pair_path = root / 'pairs.jsonl'
        if pair_path.exists():
            if readlines(pair_path) != expected_pairs:
                raise ValueError('Canonical strict pairs changed')
        else:
            with pair_path.open('x') as stream:
                for pair in expected_pairs:
                    stream.write(json.dumps(pair, sort_keys=True, allow_nan=False) + '\n')
    if final:
        from train_override import verify_models
        if (root / 'model_logistic.json').exists():
            verify_models(root, p)
        if (root / 'evaluation_schedule.json').exists():
            from override_study import verify_evaluation
            verify_evaluation(root, require_complete=True)
    print(json.dumps(quality, indent=2))
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--legacy-regressions', action='store_true')
    args = parser.parse_args()
    if args.legacy_regressions:
        verify_legacy(args.root)
    else:
        verify(args.root, args.final)


if __name__ == '__main__':
    main()

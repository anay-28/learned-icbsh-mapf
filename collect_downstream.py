"""Resumable adaptive labeling. Only the parent process writes canonical evidence."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
from cro_protocol import ROOT, append, artifact, campaign_lock, digest, freeze, provenance, read_protocol, readlines
from mapf.day3_calibration import restore
from mapf.ml_revision import replay
from mapf.ml_revision_policy import candidate_pool
from mapf.features import FEATURE_NAMES
from mapf.override_features import symmetrize, SYMMETRIC_FEATURE_NAMES
from mapf.downstream_teacher import (BUDGETS, IntegrityError, RolloutEvidence, best_evidence,
                                     check_consistency, compare, needed_candidates, rollout_candidate)


def candidate_id(candidate):
    return 'c-' + digest(candidate['conflict'])[:24]


def evidence_index(rows, p):
    result = {}
    for r in rows:
        if any(r[k] != v for k, v in provenance(p).items()):
            raise IntegrityError('Mixed rollout provenance')
        key = (r['query_id'], r['candidate_id'], r['budget'])
        if key in result:
            raise IntegrityError('Duplicate rollout key')
        result[key] = r
    return result


def histories_for(record, index):
    return [{b.name: index[(record['query_id'], candidate_id(c), b.name)]['evidence']
             for b in BUDGETS if (record['query_id'], candidate_id(c), b.name) in index}
            for c in record['candidates']]


def replay_job(record, indices, budget, p):
    """One independent saved state per worker; candidate rollouts each rebuild search."""
    if record['instance']['split'] not in ('train', 'validation'):
        raise IntegrityError('Test state in collection')
    ctx, chosen = replay(record)
    _, _, _, node, ht, _, _ = restore(record)
    if candidate_pool(chosen, ctx)[0].conflict != chosen[0].conflict:
        raise IntegrityError('Residual is not candidate zero')
    result = []
    for i in indices:
        c = record['candidates'][i]
        evidence = asdict(rollout_candidate(node, chosen[i], ctx, ht, budget))
        result.append(dict(**provenance(p), query_id=record['query_id'],
                           split=record['instance']['split'], family_id=record['instance']['family_id'],
                           map_name=record['instance']['map_name'], state_sha256=record['state_sha256'],
                           input_protocol_sha256=record['protocol_sha256'], input_source_sha256=record['source_sha256'],
                           candidate_id=candidate_id(c), candidate_index=i, conflict=c['conflict'],
                           analytical_key=c['analytical_key'], features_raw=c['features'],
                           features_symmetric=symmetrize(c['features']).tolist(),
                           feature_names_raw=FEATURE_NAMES, feature_names_symmetric=SYMMETRIC_FEATURE_NAMES,
                           budget=budget.name, evidence=evidence))
    return result


def snapshots(records, index, p, stage):
    queries, strict = [], []
    for r in records:
        hs = histories_for(r, index)
        if any(not h for h in hs):
            raise IntegrityError('Incomplete Q32 matrix')
        best = [best_evidence(h) for h in hs]
        check_consistency(best)
        pairs = []
        for i in range(1, len(best)):
            pair = dict(**provenance(p), query_id=r['query_id'], split=r['instance']['split'],
                        family_id=r['instance']['family_id'], map_name=r['instance']['map_name'],
                        state_sha256=r['state_sha256'], residual_candidate_id=candidate_id(r['candidates'][0]),
                        challenger_candidate_id=candidate_id(r['candidates'][i]), candidate_index=i,
                        **asdict(compare(best[0], best[i])),
                        residual_evidence=asdict(best[0]), challenger_evidence=asdict(best[i]),
                        feature_difference=(symmetrize(r['candidates'][i]['features']) -
                                            symmetrize(r['candidates'][0]['features'])).tolist())
            pair['relative_saving'] = ((best[0].expanded - best[i].expanded) / best[0].expanded
                                      if pair['label'] == 1 and best[0].is_complete and best[i].is_complete else None)
            # If residual is censored, the known bound supports only a saving lower bound.
            pair['relative_saving_lower_bound'] = ((best[0].expanded - best[i].expanded) / best[0].expanded
                                                   if pair['label'] == 1 and best[0].expanded else None)
            pair['pair_sha256'] = digest(pair)
            pairs.append(pair)
            if pair['label'] is not None:
                strict.append(pair)
        queries.append(dict(**provenance(p), query_id=r['query_id'], split=r['instance']['split'],
                            family_id=r['instance']['family_id'], map_name=r['instance']['map_name'],
                            state_sha256=r['state_sha256'], stage=stage,
                            residual_candidate_id=candidate_id(r['candidates'][0]),
                            candidates=[dict(candidate_id=candidate_id(c), conflict=c['conflict'],
                                             analytical_key=c['analytical_key'], features_raw=c['features'],
                                             features_symmetric=symmetrize(c['features']).tolist(), rollouts=h)
                                        for c, h in zip(r['candidates'], hs)], pairs=pairs))
    return queries, strict


def selected_records(p, limit=None):
    records = readlines(Path(p['input_path']))
    if [r['query_id'] for r in records] != p['input_query_ids']:
        raise IntegrityError('Input order changed')
    return records[:limit] if limit is not None else records


def collect(root, stage, workers=6, resume=False, limit=None):
    root = Path(root)
    if limit is not None and limit < 1:
        raise ValueError('Smoke query limit must be positive')
    if not (root / 'protocol.json').exists():
        if limit is None:
            raise ValueError('Freeze the protocol before collection')
        freeze(root, smoke=True, workers=workers)
    p = read_protocol(root)
    if limit is not None and not p['smoke']:
        raise ValueError('Query limits require a separate smoke campaign')
    records = selected_records(p, limit)
    artifact(root, 'selection.json', dict(**provenance(p), query_ids=[r['query_id'] for r in records]))
    if stage == 'q512':
        gate = json.loads((root / 'headroom_q128.json').read_text())
        if not gate['gate_passed'] and not p['smoke']:
            raise ValueError('Frozen Q128 headroom gate failed; Q512 not authorized by protocol')
    budget = next(b for b in BUDGETS if b.name == stage)
    rows = readlines(root / 'rollouts.jsonl')
    budget_order = {b.name: i for i, b in enumerate(BUDGETS)}
    if any(budget_order[row['budget']] > budget_order[stage] for row in rows):
        raise ValueError('Cannot recollect an earlier stage after escalation; its immutable snapshot is already retained')
    index = evidence_index(rows, p)
    jobs_path = root / 'rollout_jobs.jsonl'
    job_keys = set()
    for job in readlines(jobs_path):
        key = (job['query_id'], job['candidate_id'], job['budget'])
        if key in job_keys or any(job[k] != v for k, v in provenance(p).items()):
            raise IntegrityError('Duplicate/mixed rollout job')
        job_keys.add(key)
    work = []
    for record in records:
        indices = needed_candidates(histories_for(record, index), budget)
        if any((record['query_id'], candidate_id(record['candidates'][i]), stage) in job_keys for i in indices) and not resume:
            raise ValueError('Interrupted work exists; pass --resume')
        for i in indices:
            key = (record['query_id'], candidate_id(record['candidates'][i]), stage)
            if key not in job_keys:
                append(jobs_path, dict(**provenance(p), query_id=key[0], candidate_id=key[1], budget=stage,
                                       state_sha256=record['state_sha256']))
                job_keys.add(key)
        if indices:
            work.append((record, indices, budget, p))
    def retain(batch):
        for row in batch:
            key = (row['query_id'], row['candidate_id'], row['budget'])
            if key in index:
                raise IntegrityError('Attempt to recalculate completed rollout')
            row['record_sha256'] = digest(row)
            append(root / 'rollouts.jsonl', row)
            index[key] = row
    if workers == 1:
        for n, job in enumerate(work, 1):
            retain(replay_job(*job))
            print(f'{stage}: {n}/{len(work)} queries', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(replay_job, *job) for job in work]
            for n, future in enumerate(as_completed(futures), 1):
                retain(future.result())
                print(f'{stage}: {n}/{len(work)} queries', flush=True)
    queries, strict = snapshots(records, index, p, stage)
    # Snapshots are sorted and immutable; each stage records all unresolved relations too.
    artifact(root, f'queries_{stage}.json', sorted(queries, key=lambda r: r['query_id']))
    artifact(root, f'pairs_{stage}.json', sorted(strict, key=lambda r: (r['query_id'], r['candidate_index'])))
    artifact(root, f'collection_{stage}.json', dict(**provenance(p), stage=stage,
             queries=len(records), stage_rollouts=sum(key[2] == stage for key in index),
             rollouts_sha256=__import__('cro_protocol').sha(root / 'rollouts.jsonl'),
             snapshot_sha256=__import__('cro_protocol').sha(root / f'queries_{stage}.json')))


def main():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=[b.name for b in BUDGETS], required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=6)
    parser.add_argument('--limit-queries', type=int)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with campaign_lock(args.output_root):
        collect(args.output_root, args.stage, args.workers, args.resume, args.limit_queries)


if __name__ == '__main__':
    main()

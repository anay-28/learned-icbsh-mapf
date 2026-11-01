"""Pre-model headroom gates and descriptive family-clustered offline evidence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import numpy as np

from cro_protocol import ROOT, artifact, provenance, read_protocol, sha


def latest_stage(root):
    for stage in ('q512', 'q128', 'q32'):
        if (Path(root) / f'queries_{stage}.json').exists():
            return stage
    raise ValueError('No completed collection stage')


def load_queries(root, stage=None):
    stage = stage or latest_stage(root)
    return json.loads((Path(root) / f'queries_{stage}.json').read_text())


def query_headroom(q):
    strict = [pair for pair in q['pairs'] if pair['label'] is not None]
    positives = [pair for pair in strict if pair['label'] == 1]
    if not strict:
        return None
    winner = min(positives, key=lambda pair: pair['challenger_lb']) if positives else None
    return dict(query_id=q['query_id'], family_id=q['family_id'], map_name=q['map_name'],
                positive=bool(positives),
                saving_lower_bound=winner['relative_saving_lower_bound'] if winner else None,
                saving_exact=winner['relative_saving'] if winner else None,
                absolute_saving_lower_bound=winner['residual_lb'] - winner['challenger_lb'] if winner else 0)


def family_headroom_interval(rows, seed=20261101, repetitions=10000):
    """Resample whole families within map strata; retain all queries in each draw."""
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[row['map_name']][row['family_id']].append(row['positive'])
    if not strata:
        return None
    rng = np.random.default_rng(seed)
    draws = np.zeros((repetitions, 2))
    for families in strata.values():
        values = np.array([[sum(v), len(v)] for _, v in sorted(families.items())])
        picks = rng.integers(0, len(values), size=(repetitions, len(values)))
        draws += values[picks].sum(axis=1)
    rates = draws[:, 0] / draws[:, 1]
    return np.quantile(rates, [.025, .975]).tolist()


def headroom_summary(queries, gates):
    train = [q for q in queries if q['split'] == 'train']
    rows = [r for q in train if (r := query_headroom(q)) is not None]
    positives = [r for r in rows if r['positive']]
    savings = [r['saving_lower_bound'] for r in positives]
    stats = dict(informative_train_queries=len(rows), positive_train_queries=len(positives),
                 positive_train_families=len({r['family_id'] for r in positives}),
                 positive_train_maps=len({r['map_name'] for r in positives}),
                 headroom_rate=len(positives) / len(rows) if rows else 0.,
                 median_relative_saving=statistics.median(savings) if savings else 0.)
    checks = {name: stats[name] >= gates[name] for name in stats}
    rates = {}
    for name in sorted({q['map_name'] for q in train}):
        subset = [r for r in rows if r['map_name'] == name]
        rates[name] = dict(informative=len(subset), positives=sum(r['positive'] for r in subset),
                           rate=sum(r['positive'] for r in subset) / len(subset) if subset else None)
    return dict(statistics=stats, checks=checks, gate_passed=all(checks.values()),
                saving_rule='Use conservative observed relative-saving lower bound when residual is censored; no fabricated completion count',
                family_bootstrap_95=family_headroom_interval(rows), by_map=rates, informative_queries=rows,
                relations=dict(Counter(pair['relation'] for q in queries for pair in q['pairs'])),
                conclusion=('Sufficient bounded downstream headroom for frozen model fitting' if all(checks.values()) else
                            'STOP: insufficient supported headroom/training support; distinguish censoring from residual saturation'))


def old_policy_comparison(queries, records):
    from mapf.ml_revision_policy import SavedLinear, teacher_key
    from mapf.downstream_teacher import best_evidence, compare
    model = SavedLinear('results/recovery/ml_revision/model_b.json')
    by_id = {r['query_id']: r for r in records}
    out = []
    for q in queries:
        r = by_id[q['query_id']]
        keys = [teacher_key(c['teacher']) for c in r['candidates']]
        teacher = max(range(len(keys)), key=lambda i: keys[i]) if all(k is not None for k in keys) else None
        model_index = int(np.argmax(model.predict(np.array([c['features'] for c in r['candidates']]))))
        best = [best_evidence(c['rollouts']) for c in q['candidates']]
        row = dict(query_id=q['query_id'], split=q['split'], family_id=q['family_id'],
                   residual_work=best[0].expanded if best[0].is_complete else None,
                   residual_lower_bound=best[0].expanded)
        for name, i in (('teacher', teacher), ('model_b', model_index)):
            row[name] = None if i is None else dict(candidate_index=i, disagrees=i != 0,
                    work=best[i].expanded if best[i].is_complete else None,
                    lower_bound=best[i].expanded, relation=compare(best[0], best[i]).relation)
        out.append(row)
    return out


def analyze(root=ROOT, stage=None):
    from cro_protocol import readlines
    root = Path(root)
    p = read_protocol(root)
    stage = stage or latest_stage(root)
    queries = load_queries(root, stage)
    summary = dict(**provenance(p), stage=stage, queries_sha256=sha(root / f'queries_{stage}.json'),
                   **headroom_summary(queries, p['gates']))
    summary['old_teacher_comparison'] = old_policy_comparison(queries, readlines(Path(p['input_path'])))
    artifact(root, f'headroom_{stage}.json', summary)
    print(json.dumps({k: summary[k] for k in ('stage', 'statistics', 'checks', 'gate_passed', 'conclusion')}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=['q32', 'q128', 'q512'])
    parser.add_argument('--headroom', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    if args.validation:
        print((args.root / 'validation.json').read_text())
    else:
        analyze(args.root, args.stage)


if __name__ == '__main__':
    main()

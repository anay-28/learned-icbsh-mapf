"""Export only measured BDRO figures. Missing/gated experiments stay unplotted."""
import argparse
import io
import json
from pathlib import Path
import numpy as np

from cro_protocol import ROOT, artifact, provenance, read_protocol, sha
from analyze_downstream import latest_stage, load_queries, query_headroom


def figures(root=ROOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    p = read_protocol(root)
    stage = latest_stage(root)
    queries = load_queries(root, stage)
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    manifest = dict(**provenance(p), queries_sha256=sha(root / f'queries_{stage}.json'), stage=stage, files={})

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'svg'):
            buffer = io.BytesIO()
            fig.savefig(buffer, format=suffix, dpi=160, metadata={'Date': None} if suffix == 'svg' else None)
            path = out / f'{name}.{suffix}'
            # Figures are immutable too; reruns verify the evidence manifest instead of replacing files.
            with path.open('xb') as stream:
                stream.write(buffer.getvalue())
            manifest['files'][str(path.relative_to(root))] = sha(path)
        plt.close(fig)

    if (out / 'manifest.json').exists():
        previous = json.loads((out / 'manifest.json').read_text())
        if previous['queries_sha256'] != manifest['queries_sha256']:
            raise ValueError('Figures belong to a different stage; use a separate output root')
        for path, expected in previous['files'].items():
            if sha(root / path) != expected:
                raise ValueError('Figure checksum mismatch')
        return previous
    headroom = [r for q in queries if q['split'] == 'train' and (r := query_headroom(q)) is not None]
    positive = [r for r in headroom if r['positive']]
    if positive:
        values = np.sort([r['absolute_saving_lower_bound'] for r in positive])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post')
        ax.set(xlabel='Observed lower bound on CT expansions saved', ylabel='ECDF among positive train queries', ylim=(0, 1.02))
        save(fig, 'headroom_ecdf')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([r['saving_lower_bound'] for r in positive], bins=10, range=(0, 1))
        ax.set(xlabel='Relative saving (conservative lower bound)', ylabel='Positive training queries')
        save(fig, 'relative_savings')
    fig, ax = plt.subplots(figsize=(6, 4))
    names, rates = [], []
    for name in ('q32', 'q128', 'q512'):
        path = root / f'queries_{name}.json'
        if path.exists():
            pairs = [pair for q in json.loads(path.read_text()) for pair in q['pairs']]
            names.append(name.upper())
            rates.append(sum(pair['relation'] != 'UNKNOWN' for pair in pairs) / len(pairs))
    ax.plot(names, rates, marker='o')
    ax.set(xlabel='Completed adaptive budget stage', ylabel='Fraction of all pairs resolved (including ties)', ylim=(0, 1))
    save(fig, 'budget_resolution')
    fig, ax = plt.subplots(figsize=(8, 4))
    names = sorted({r['map_name'] for r in headroom})
    ax.bar(names, [sum(r['positive'] for r in headroom if r['map_name'] == name) /
                       sum(r['map_name'] == name for r in headroom) for name in names])
    ax.axhline(.15, linestyle='--', color='black', label='Frozen overall headroom threshold')
    ax.set(ylabel='Strict challenger-win fraction of informative train queries', ylim=(0, 1))
    ax.tick_params(axis='x', labelrotation=12)
    ax.legend(fontsize=8)
    save(fig, 'headroom_by_map')
    old_path = root / f'headroom_{stage}.json'
    if old_path.exists():
        comparison = json.loads(old_path.read_text())['old_teacher_comparison']
        fig, ax = plt.subplots(figsize=(6, 4))
        labels, values = [], []
        for name in ('teacher', 'model_b'):
            known = [r[name] for r in comparison if r[name] and r[name]['disagrees'] and
                     r[name]['relation'] in ('CHALLENGER_BETTER', 'RESIDUAL_BETTER', 'TIE')]
            labels.append(name)
            values.append(sum(r['relation'] == 'CHALLENGER_BETTER' for r in known) / len(known) if known else 0)
        ax.bar(labels, values)
        ax.set(ylabel='Strict downstream win rate on resolved disagreements', ylim=(0, 1))
        save(fig, 'old_teacher_comparison')
    if (root / 'validation.json').exists():
        validation = json.loads((root / 'validation.json').read_text())
        known = [r for r in validation['predictions'] if r['label'] is not None]
        bins = np.linspace(0, 1, 6)
        fig, ax = plt.subplots(figsize=(5, 4))
        points = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            group = [r for r in known if lo <= r['probability'] < hi or hi == 1 and r['probability'] == 1]
            if group:
                points.append((np.mean([r['probability'] for r in group]), np.mean([r['label'] for r in group])))
        if points:
            ax.plot(*zip(*points), marker='o')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set(xlabel='Mean predicted challenger-win probability', ylabel='Observed strict-pair win frequency', xlim=(0, 1), ylim=(0, 1))
        save(fig, 'override_calibration')
        best = {}
        for row in validation['predictions']:
            if row['query_id'] not in best or row['probability'] > best[row['query_id']]['probability']:
                best[row['query_id']] = row
        fig, ax = plt.subplots(figsize=(6, 4))
        xs, ys = [], []
        for tau in np.linspace(.5, .95, 10):
            accepted = [r for r in best.values() if r['probability'] >= tau]
            if accepted:
                xs.append(len(accepted) / len(best))
                ys.append(sum(r['label'] == 1 for r in accepted) / len(accepted))
        ax.plot(xs, ys, marker='o')
        ax.set(xlabel='Validation query override coverage', ylabel='Conservative override precision', xlim=(0, 1), ylim=(0, 1))
        save(fig, 'precision_coverage')
    if (root / 'evaluation_summary.json').exists():
        result = json.loads((root / 'evaluation_summary.json').read_text())
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter([r['residual_seconds'] for r in result['family_ratios']], [r['policy_seconds'] for r in result['family_ratios']])
        ax.plot([0, 10], [0, 10], linestyle='--', color='gray')
        ax.set(xlabel='Residual family mean capped seconds', ylabel='BDRO family mean capped seconds')
        save(fig, 'final_capped_runtime')
    artifact(out, 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(figures(args.root), indent=2))


if __name__ == '__main__':
    main()

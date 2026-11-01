"""Freeze untouched official scenario families by seeded SHA-256 ordering."""
import argparse
import json
from pathlib import Path
import re

from cro_protocol import ROOT, digest, provenance, read_protocol, sha, utc, write_new
from mapf.grid import load_map
from mapf.movingai import parse_scen, first_k


def used_scenarios(roots):
    names, scanned = set(), {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        paths = set((root / 'results').rglob('*'))
        paths.update(root.glob('*.json'))
        paths.update(root.glob('*.py'))
        paths.update(root.glob('*.md'))
        paths.update(root.rglob('config*'))
        for path in sorted(paths):
            if (not path.is_file() or path.suffix.lower() not in ('.json', '.jsonl', '.yaml', '.yml', '.csv', '.txt', '.md', '.py', '.toml')
                    or 'counterfactual_override' in path.parts or '.venv' in path.parts or 'venv' in path.parts):
                continue
            content = path.read_text(encoding='utf-8', errors='replace')
            names.update(re.findall(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.scen\b', content))
            scanned[str(path)] = sha(path)
    return names, scanned


def prepare(root=ROOT, benchmark_root=Path('data/movingai'), exclude_roots=None):
    root, benchmark_root = Path(root), Path(benchmark_root)
    p = read_protocol(root)
    validation = json.loads((root / 'validation.json').read_text())
    if not validation['gate_passed'] or p['smoke']:
        raise ValueError('Frozen validation gate must pass before holdout preparation')
    if (root / 'movingai_holdout_manifest.json').exists():
        raise FileExistsError('Holdout is already frozen')
    roots = exclude_roots or [Path('.'), Path('../learned-icbsh-mapf')]
    excluded, scanned = used_scenarios(roots)
    seed = p['holdout']['selection_seed']
    families, instances = [], []
    for map_name, counts in p['holdout']['map_agents'].items():
        map_paths = list(benchmark_root.rglob(map_name + '.map'))
        if len(map_paths) != 1:
            raise ValueError(f'Need exactly one official {map_name}.map under {benchmark_root}')
        map_path = map_paths[0]
        grid = load_map(str(map_path))
        for kind in ('random', 'even'):
            paths = sorted(benchmark_root.rglob(f'{map_name}-{kind}-*.scen'))
            if len({path.name for path in paths}) != len(paths):
                raise ValueError('Duplicate scenario basenames')
            available = [path for path in paths if path.name not in excluded]
            available.sort(key=lambda path: (digest([seed, map_name, kind, path.name]), path.name))
            if len(available) < 2:
                raise ValueError(f'Need at least two untouched official {kind} files for {map_name}')
            for path in available[:2]:
                rows = parse_scen(path)
                family = f'{map_name}/{path.stem}'
                families.append(dict(family_id=family, map_name=map_name, kind=kind, map_path=map_path.as_posix(),
                                     scen_path=path.as_posix(), map_sha256=sha(map_path), scen_sha256=sha(path)))
                for k in counts:
                    starts, goals = first_k(rows, k, grid, map_name + '.map')
                    ins = dict(instance_id=f'{family}/k{k}', family_id=family, map_name=map_name, k=k, split='test',
                               map_path=map_path.as_posix(), scen_path=path.as_posix(),
                               map_sha256=sha(map_path), scen_sha256=sha(path), starts=starts, goals=goals)
                    ins['instance_sha256'] = digest(ins)
                    instances.append(ins)
    if len(instances) != 24 or len(families) != 12:
        raise ValueError('Unexpected holdout dimensions')
    manifest = dict(**provenance(p), frozen_utc=utc(), selection_seed=seed, instances=instances, families=families,
                    excluded_scenario_basenames=sorted(excluded), exclusion_scan_sha256=scanned,
                    validation_sha256=sha(root / 'validation.json'),
                    override_model_sha256=sha(root / 'model_logistic.json'),
                    selection='SHA256 of compact sorted JSON [seed,map,kind,basename]; first 2 unused of each kind; first k rows')
    manifest['manifest_sha256'] = digest(manifest)
    write_new(root / 'movingai_holdout_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--benchmark-root', type=Path, default=Path('data/movingai'))
    parser.add_argument('--selection-seed', choices=['bdro-holdout-20261101'], default='bdro-holdout-20261101')
    parser.add_argument('--random-files-per-map', type=int, choices=[2], default=2)
    parser.add_argument('--even-files-per-map', type=int, choices=[2], default=2)
    parser.add_argument('--exclude-root', action='append', type=Path)
    args = parser.parse_args()
    print(json.dumps(dict(instances=len(prepare(args.root, args.benchmark_root, args.exclude_root)['instances']))))


if __name__ == '__main__':
    main()

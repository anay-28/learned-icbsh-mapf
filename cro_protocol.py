"""Freeze BDRO source, immutable inputs and decision rules before new outcomes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from contextlib import contextmanager

ROOT = Path('results/counterfactual_override')
INPUT = Path('results/recovery/ml_revision/labels.jsonl')
STUDY_FILES = ['cro_protocol.py', 'collect_downstream.py', 'verify_downstream.py',
               'analyze_downstream.py', 'train_override.py', 'prepare_movingai_holdout.py',
               'override_study.py', 'override_figures.py', 'test_counterfactual_override.py']
TAU = .75
GATES = dict(informative_train_queries=40, positive_train_queries=20,
             positive_train_families=8, positive_train_maps=2, headroom_rate=.15,
             median_relative_saving=.10, validation_auc=.70, validation_precision=.70,
             validation_overrides=5, validation_families=3)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def readlines(path):
    """Fail on partial/truncated rows rather than silently treating work as absent."""
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def campaign_lock(root):
    """OS-released lock prevents two parent processes from appending concurrently."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.campaign.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another study process owns this output root') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def artifact(root, name, value):
    """Idempotent exclusive creation; different content needs a new artifact name."""
    path = Path(root) / name
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Immutable artifact differs: {path}')
    else:
        write_new(path, value)
    return path


def source_files():
    return sorted(Path('mapf').glob('*.py')) + [Path(n) for n in STUDY_FILES]


def source_manifest():
    return {p.as_posix(): sha(p) for p in source_files()}


def historical_manifest():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], text=True).split('\0')
    return {p: sha(p) for p in paths if p and not p.startswith('results/counterfactual_override/')
            and p not in STUDY_FILES and not p.startswith('mapf/override_')
            and p not in ('mapf/downstream_teacher.py', 'mapf/movingai.py', 'COUNTERFACTUAL_OVERRIDE_REPORT.md')}


def freeze(root=ROOT, input_path=INPUT, seed=20261101, workers=6, smoke=False):
    from dataclasses import asdict
    from mapf.downstream_teacher import BUDGETS
    from mapf.override_features import SYMMETRIC_FEATURE_NAMES
    root, input_path = Path(root), Path(input_path)
    if (root / 'protocol.json').exists():
        raise FileExistsError('Protocol already frozen; use a new output root')
    records = readlines(input_path)
    if not records or len({r['query_id'] for r in records}) != len(records):
        raise ValueError('Empty or duplicate input states')
    families = {s: sorted({r['instance']['family_id'] for r in records if r['instance']['split'] == s})
                for s in ('train', 'validation')}
    if any(r['instance']['split'] not in families for r in records) or set(families['train']) & set(families['validation']):
        raise ValueError('Test input or family leakage')
    parent = input_path.parent / 'protocol.json'
    parent_p = json.loads(parent.read_text())
    check = dict(parent_p)
    if digest({k: v for k, v in check.items() if k != 'protocol_sha256'}) != check['protocol_sha256']:
        raise ValueError('Input protocol mismatch')
    if any(r['protocol_sha256'] != check['protocol_sha256'] for r in records):
        raise ValueError('Mixed input protocols')
    files = source_manifest()
    protected = historical_manifest()
    p = dict(schema_version='bdro-protocol-v1', frozen_utc=utc(), seed=seed, workers=workers,
             smoke=smoke, base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             input_path=input_path.as_posix(), input_sha256=sha(input_path),
             input_protocol_path=parent.as_posix(), input_protocol_sha256=sha(parent),
             input_query_ids=[r['query_id'] for r in records], families=families,
             source_sha256=digest(files), source_files=files, historical_files=protected,
             budgets=[asdict(b) for b in BUDGETS], tau=TAU, gates=GATES,
             target='CT expansions to complete residual continuation after same-state forced first split',
             censoring='Only observed expanded is a lower bound; strict comparisons; UNKNOWN is not TIE',
             features=SYMMETRIC_FEATURE_NAMES,
             models=dict(primary='logistic L2 C=1 liblinear intercept max_iter=10000 seed=71',
                         diagnostic='Ridge alpha=10 no intercept', scaling='training pair differences only; query weight one'),
             holdout=dict(selection_seed='bdro-holdout-20261101', files_per_kind=2,
                          map_agents={'random-32-32-20': [20, 28], 'room-32-32-4': [16, 24],
                                      'warehouse-10-20-10-2-1': [16, 24]}),
             evaluation=dict(policies=['plain', 'residual', 'model_b', 'override'], repetitions=2,
                             timeout_seconds=10, seed=71, bootstrap_replicates=10000,
                             bootstrap_seed=20261101, memory_mb=4096, serial=True),
             environment=dict(python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
                              dependencies=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()))
    p['protocol_sha256'] = digest(p)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(name, name)
    write_new(root / 'source.json', dict(source_sha256=p['source_sha256'], files=files, archive_sha256=sha(root / 'source.zip')))
    write_new(root / 'inputs_manifest.json', dict(input_sha256=p['input_sha256'],
               records=[dict(query_id=r['query_id'], state_sha256=r['state_sha256'],
                             record_sha256=digest(r), split=r['instance']['split'],
                             family_id=r['instance']['family_id']) for r in records],
               protocol_sha256=p['protocol_sha256'], source_sha256=p['source_sha256']))
    write_new(root / 'protocol.json', p)
    return p


def read_protocol(root=ROOT, check_source=True):
    p = json.loads((Path(root) / 'protocol.json').read_text())
    if digest({k: v for k, v in p.items() if k != 'protocol_sha256'}) != p['protocol_sha256']:
        raise ValueError('BDRO protocol modified')
    if sha(p['input_path']) != p['input_sha256'] or sha(p['input_protocol_path']) != p['input_protocol_sha256']:
        raise ValueError('Frozen input modified')
    if check_source and source_manifest() != p['source_files']:
        raise ValueError('Source changed; start a separate frozen campaign')
    for path, expected in p['historical_files'].items():
        if sha(path) != expected:
            raise ValueError(f'Historical file modified: {path}')
    return p


def provenance(p):
    return dict(schema_version='bdro-v1', protocol_sha256=p['protocol_sha256'],
                source_sha256=p['source_sha256'], input_sha256=p['input_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze'])
    parser.add_argument('--input', type=Path, default=INPUT)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=6)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    p = freeze(args.output_root, args.input, args.seed, args.workers, args.smoke)
    print(json.dumps(dict(protocol_sha256=p['protocol_sha256'], queries=len(p['input_query_ids']))))


if __name__ == '__main__':
    main()

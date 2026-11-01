"""One cold four-policy run. Imports, loading and validation are inside the cap."""
import json
from pathlib import Path
import sys
import time


def run(spec):
    began = time.perf_counter()
    import numpy as np
    from cro_protocol import sha
    from mapf.grid import GridMap, load_map
    from mapf.icbs_h import ICBShSolver
    from mapf.baselines import f_aware_residual_selector
    from mapf.ml_revision_policy import RevisionSelector, SavedLinear
    from mapf.validation import validate_solution

    ins = spec['instance']
    if 'map_path' in ins:
        if sha(ins['map_path']) != ins['map_sha256']:
            raise ValueError('Benchmark map checksum mismatch')
        grid = load_map(ins['map_path'])
    else:
        grid = GridMap(ins['width'], ins['height'], {tuple(p) for p in ins.get('obstacles', [])})
    starts, goals = list(map(tuple, ins['starts'])), list(map(tuple, ins['goals']))
    policy = spec['policy']
    if policy == 'plain':
        selector = None
    elif policy == 'residual':
        selector = f_aware_residual_selector
    elif policy == 'model_b':
        selector = RevisionSelector('model_b', SavedLinear(spec['model_paths']['model_b']), spec['seed'])
    elif policy == 'override':
        from mapf.override_policy import ResidualOverrideSelector, SavedOverride
        selector = ResidualOverrideSelector(SavedOverride(spec['model_paths']['override']))
    else:
        raise ValueError(f'Unknown policy: {policy}')
    remaining = max(0., spec['timeout_seconds'] - (time.perf_counter() - began))
    result = ICBShSolver(grid, conflict_selector=selector).solve(starts, goals, timeout=remaining)
    if hasattr(selector, 'stats'):
        result['selector_stats'] = selector.stats
    if result['status'] == 'SOLVED':
        validate_solution(grid, starts, goals, result['paths'], result['cost'])
        result['solution_validated'] = True
    result.update(worker_wall_seconds=time.perf_counter() - began,
                  runtime_environment=dict(python=sys.version.split()[0], numpy=np.__version__))
    return result


def main():
    import psutil
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(spec)
    except Exception as exc:
        result = dict(status='ERROR', cost=None, detail=f'{type(exc).__name__}: {exc}')
    info = psutil.Process().memory_info()
    if sys.platform != 'win32':
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['process_peak_rss_bytes'] = int(peak * (1 if sys.platform == 'darwin' else 1024))
        result['peak_memory_method'] = 'native ru_maxrss'
    else:
        result['process_peak_rss_bytes'] = info.peak_wset
        result['peak_memory_method'] = 'native peak_wset'
    result['process_cpu_seconds'] = time.process_time()
    with Path(sys.argv[2]).open('x') as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == '__main__':
    main()

"""Full-solver, one-decision counterfactual evidence for the fresh MAPF study.

Every intervention re-executes the original search prefix. It therefore retains
the actual OPEN frontier, cache contents and node ordering, unlike a descendant
restart. A rolling digest verifies every prior decision context at the target.
Timing is diagnostic under parallel collection; work labels are not runtime
ground truth. Replays include all setup and prefix costs in their total fields.
"""
from dataclasses import asdict
import hashlib
import json
import math
import time

import numpy as np

from mapf.baselines import f_aware_residual_selector
from mapf.icbs_h import ICBShSolver
from mapf.low_level import get_astar_call_count, get_astar_expansion_count
from mapf.ml_revision_policy import candidate_pool
from mapf.validation import validate_solution

FEATURE_SCHEMA = 'multisignal-candidate-v1'
_BRANCH_NAMES = ('infeasible', 'cost_increase', 'path_cost', 'remaining_path',
                 'conflict_degree', 'cardinal_degree', 'alt_conflicts',
                 'vertex_constraints', 'edge_constraints')
FEATURE_NAMES = ['time', 'relative_time', 'edge', 'cardinal', 'semicardinal',
                 'pair_history', 'time_after_shorter_path'] + [
    f'branch_{i}_{name}' for i in (0, 1) for name in _BRANCH_NAMES]
DEFAULT_DECISIONS = (0, 1, 3, 7, 15, 31)
COMPLETE = ('SOLVED', 'UNSOLVABLE', 'NO_ROOT_PATH')


def _plain(value):
    """Canonical strict-JSON representation, including infeasible branch costs."""
    if isinstance(value, float) and not math.isfinite(value):
        return '+inf' if value > 0 else '-inf' if value < 0 else 'nan'
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def candidate_features(pool, context):
    """25 finite decision-time features with complete branch-record symmetry.

    Both branch records are sorted lexicographically as whole records, retaining
    relationships between e.g. path length, degree and replanning cost. There is
    no clipping: infeasibility has its own flag and an unused cost value of zero.
    These cached features make no A* calls and use no rollout measurements.
    """
    paths = context['paths']
    constraints = context['per_agent_constraints']
    classified = context['classified']
    degree = [0] * len(paths)
    cardinal_degree = [0] * len(paths)
    for cc in classified:
        for agent in (cc.conflict.agent_i, cc.conflict.agent_j):
            degree[agent] += 1
            cardinal_degree[agent] += int(cc.classification == 'CARDINAL')
    rows = []
    horizon = max(1, max(len(p) - 1 for p in paths))
    for cc in pool:
        c = cc.conflict
        records = []
        for suffix, agent in (('i', c.agent_i), ('j', c.agent_j)):
            alternate = getattr(cc, f'alt_path_{suffix}')
            feasible = alternate is not None
            cost = len(paths[agent]) - 1
            alt_conflicts = getattr(cc, f'alt_total_conflicts_{suffix}') if feasible else 0
            if feasible and (not isinstance(alt_conflicts, (int, float)) or
                             not math.isfinite(alt_conflicts) or alt_conflicts >= 999999):
                raise ValueError('Candidate lacks cached exact alternative conflict count')
            cs = constraints[agent]
            records.append((int(not feasible), len(alternate)-1-cost if feasible else 0,
                cost, max(0, cost-c.time), degree[agent], cardinal_degree[agent],
                alt_conflicts, len(cs.vertex_constraints), len(cs.edge_constraints)))
        records.sort()
        pair = tuple(sorted((c.agent_i, c.agent_j)))
        row = [c.time, c.time/horizon, int(c.conflict_type == 'EDGE'),
               int(cc.classification == 'CARDINAL'), int(cc.classification == 'SEMI_CARDINAL'),
               context.get('pair_counts', {}).get(pair, 0),
               max(0, c.time-min(len(paths[a])-1 for a in pair))]
        rows.append(row + list(records[0]) + list(records[1]))
    result = np.asarray(rows, dtype=float).reshape(len(pool), len(FEATURE_NAMES))
    if not np.isfinite(result).all():
        raise ValueError('Non-finite decision-time features')
    return result


def _context_payload(candidates, context):
    # Includes all available mutable selection state, not just chosen conflict.
    return dict(map=dict(width=context['grid'].width, height=context['grid'].height,
                         obstacles=sorted(context['grid'].obstacles)),
        paths=context['paths'], starts=context['starts'], goals=context['goals'],
        constraints=[dict(vertex=sorted(cs.vertex_constraints), edge=sorted(cs.edge_constraints))
                     for cs in context['per_agent_constraints']],
        effective_h=context['effective_h'], mvc_size=context['mvc_size'],
        pairs=sorted((list(k), v) for k, v in context.get('pair_counts', {}).items()),
        classified=[asdict(cc) for cc in context['classified']],
        eligible=[asdict(cc.conflict) for cc in candidates])


class ReplayIntegrityError(ValueError):
    """The replay does not reconstruct the declared decision."""


class _Selector:
    def __init__(self, source_policy, progress, decision_indices=(), query=None, candidate_index=None):
        if source_policy not in ('residual', 'plain'):
            raise ValueError('Unknown source policy')
        self.source_policy = source_policy
        self.progress = progress
        self.indices = set(decision_indices)
        self.query = query
        self.target_index = candidate_index
        self.queries = []
        self.ordinal = 0
        self.chain = 'multisignal-replay-v1'
        self.applied = False
        self.prefix = None
        self.last_diagnostic = None

    def __call__(self, candidates, context):
        # Full context hashing is collector-only overhead; deployed inference
        # does not use this wrapper. It changes neither paths nor constraints.
        self.chain = _digest([self.chain, _context_payload(candidates, context)])
        current = self.ordinal
        self.ordinal += 1
        self.last_diagnostic = dict(decision_index=current,
            current_node_conflicts=len(context['classified']),
            current_node_effective_f=sum(len(p)-1 for p in context['paths']) + context['effective_h'])
        default = (f_aware_residual_selector(candidates, context)
                   if self.source_policy == 'residual' else candidates[0])
        if current not in self.indices and (self.query is None or current != self.query['decision_index']):
            return default
        pool = candidate_pool(candidates, context)
        identities = [asdict(c.conflict) for c in pool]
        if current in self.indices:
            self.queries.append(_plain(dict(schema='multisignal-query-v1',
                feature_schema=FEATURE_SCHEMA, feature_names=FEATURE_NAMES,
                source_policy=self.source_policy, decision_index=current,
                fingerprint=self.chain, candidates=identities,
                features=candidate_features(pool, context).tolist(),
                baseline_index=next((i for i, cc in enumerate(pool) if cc is default), None),
                prefix_nodes_expanded=self.progress.get('nodes_expanded', 0),
                prefix_ll_expansions=get_astar_expansion_count(),
                prefix_astar_calls=get_astar_call_count())))
        if self.query is not None and current == self.query['decision_index']:
            if self.query['fingerprint'] != self.chain or _plain(identities) != self.query['candidates']:
                raise ReplayIntegrityError('Search prefix or candidate identities changed')
            self.prefix = dict(nodes_expanded=self.progress.get('nodes_expanded', 0),
                ll_expansions=get_astar_expansion_count(), astar_calls=get_astar_call_count(),
                wall=time.perf_counter(), cpu=time.process_time())
            self.applied = True
            return pool[self.target_index]
        return default


def _run(grid, starts, goals, selector, progress, timeout, max_expansions):
    began, cpu = time.perf_counter(), time.process_time()
    result = ICBShSolver(grid, conflict_selector=selector).solve(starts, goals,
        timeout=timeout, max_expansions=max_expansions, progress=progress)
    finished, cpu_finished = time.perf_counter(), time.process_time()
    status = result['status']
    valid = False
    if status == 'SOLVED':
        validate_solution(grid, starts, goals, result['paths'], result['cost'])
        valid = True
    evidence = dict(schema='multisignal-evidence-v1', status=status,
        intervention_applied=selector.applied,
        nodes_expanded=result.get('nodes_expanded') if result.get('nodes_expanded') is not None
                       else progress.get('nodes_expanded', 0),
        ll_expansions=get_astar_expansion_count(), astar_calls=get_astar_call_count(),
        wall_seconds=finished-began, cpu_seconds=cpu_finished-cpu,
        cost=result.get('cost'), paths=result.get('paths'), solution_validated=valid,
        decision_calls=selector.ordinal, last_decision_diagnostic=selector.last_diagnostic,
        timeout=timeout, max_expansions=max_expansions)
    if selector.prefix is not None:
        for key in ('nodes_expanded', 'll_expansions', 'astar_calls'):
            evidence['prefix_'+key] = selector.prefix[key]
            evidence['suffix_'+key] = evidence[key] - selector.prefix[key]
        evidence['suffix_wall_seconds'] = finished-selector.prefix['wall']
        evidence['suffix_cpu_seconds'] = cpu_finished-selector.prefix['cpu']
    return _plain(evidence)


def collect_instance(grid, starts, goals, source_policy='residual', timeout=30.,
                     max_expansions=2048, decision_indices=DEFAULT_DECISIONS):
    """Collect fixed ordinals, keeping unsuccessful trajectories and denominators."""
    if any(type(i) is not int or i < 0 for i in decision_indices):
        raise ValueError('Decision indices must be nonnegative integers')
    progress = {}
    selector = _Selector(source_policy, progress, decision_indices=decision_indices)
    evidence = _run(grid, starts, goals, selector, progress, timeout, max_expansions)
    return dict(queries=selector.queries, evidence=evidence)


def replay_candidate(grid, starts, goals, query, candidate_index, timeout=30., max_expansions=2048):
    """Force one target choice during a full solve; preserve ordinary bypassing."""
    if query.get('schema') != 'multisignal-query-v1' or query.get('feature_schema') != FEATURE_SCHEMA:
        raise ReplayIntegrityError('Unsupported query schema')
    if type(candidate_index) is not int or not 0 <= candidate_index < len(query['candidates']):
        raise ValueError('Candidate index out of range')
    progress = {}
    selector = _Selector(query['source_policy'], progress, query=query, candidate_index=candidate_index)
    evidence = _run(grid, starts, goals, selector, progress, timeout, max_expansions)
    evidence.update(decision_index=query['decision_index'], candidate_index=candidate_index,
                    query_fingerprint=query['fingerprint'])
    if not selector.applied and evidence['status'] in COMPLETE:
        raise ReplayIntegrityError('Complete replay never reached target decision')
    return evidence


def compare_evidence(residual, challenger, metric='ll_expansions'):
    """Strict complete/censored work comparison; equal lower bounds are unknown.

    Uses post-intervention work, verifying that deterministic prefix counters
    agree. No runtime label is inferred from an expansion count. A candidate
    stopped before the intervention cannot provide a preference.
    """
    if metric not in ('ll_expansions', 'nodes_expanded', 'astar_calls'):
        raise ValueError('Unsupported deterministic work metric')
    if not residual.get('intervention_applied') or not challenger.get('intervention_applied'):
        return dict(label=None, relation='NOT_REACHED', metric=metric)
    for key in ('query_fingerprint', 'prefix_'+metric):
        if residual.get(key) != challenger.get(key):
            raise ReplayIntegrityError('Comparisons must share the same reconstructed prefix')
    complete_r = residual['status'] in COMPLETE
    complete_c = challenger['status'] in COMPLETE
    if complete_r and complete_c:
        if residual['status'] != challenger['status'] or residual.get('cost') != challenger.get('cost'):
            raise ReplayIntegrityError('Complete full-solver interventions disagree on optimal outcome')
    for evidence in (residual, challenger):
        if evidence['status'] == 'SOLVED' and not evidence.get('solution_validated'):
            raise ReplayIntegrityError('Unvalidated solved evidence')
    r, c = residual['suffix_'+metric], challenger['suffix_'+metric]
    if any(type(v) is not int or v < 0 for v in (r, c)):
        raise ReplayIntegrityError('Invalid observed work')
    label, relation = None, 'UNKNOWN'
    if complete_r and complete_c:
        label = 1 if c < r else 0 if r < c else None
        relation = 'CHALLENGER_BETTER' if c < r else 'RESIDUAL_BETTER' if r < c else 'TIE'
    elif complete_c and c < r:
        label, relation = 1, 'CHALLENGER_BETTER'
    elif complete_r and r < c:
        label, relation = 0, 'RESIDUAL_BETTER'
    return dict(label=label, relation=relation, metric=metric,
                residual_observed=r, challenger_observed=c,
                residual_complete=complete_r, challenger_complete=complete_c)

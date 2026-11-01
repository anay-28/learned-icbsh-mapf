"""BDRO safety, censoring, invariance, provenance and cold-worker regression checks."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from cro_protocol import GATES, artifact, digest, sha, write_new
from collect_downstream import candidate_id, replay_job
from mapf.downstream_teacher import (BUDGETS, IntegrityError, RolloutBudget, RolloutEvidence,
                                     best_evidence, check_consistency, compare, needed_candidates, rollout_candidate)
from mapf.override_features import ORDERED_PAIRS, symmetrize, SYMMETRIC_FEATURE_NAMES
from mapf.override_policy import SavedOverride, ResidualOverrideSelector
from mapf.features import FEATURE_NAMES
from mapf.ml_revision_policy import candidate_pool
from mapf.baselines import f_aware_residual_selector
from mapf.grid import GridMap
from mapf.low_level import ConstraintSet, compute_bfs_h_table
from mapf.cardinal import classify_conflicts
from mapf.conflict_graph import compute_h4_and_graph
from mapf.icbs_h import ICBSNode
from mapf.repair_pilot import eligible_tier, context_for
from mapf.validation import validate_constrained_paths
from mapf.movingai import parse_scen, first_k
from train_override import training_arrays
from override_study import family_ratios, stratified_interval, run_isolated
from verify_downstream import independent_relation


def ev(status='SOLVED', n=5, budget='q32', cost=8):
    return RolloutEvidence(budget, status, n, forced_split_applied=True,
                           solution_cost=cost if status == 'SOLVED' else None,
                           solution_validated=status == 'SOLVED')


def tiny_state():
    grid = GridMap(3, 3)
    starts, goals = [(1, 0), (1, 2), (0, 1)], [(1, 2), (1, 0), (2, 1)]
    paths = [[s, (1, 1), g] for s, g in zip(starts, goals)]
    cs = [ConstraintSet() for _ in starts]
    classified = classify_conflicts(grid, starts, goals, paths, cs)
    h, graph = compute_h4_and_graph(classified)
    node = ICBSNode(cs, paths, 6, h)
    ctx = context_for(node, grid, goals, graph)
    ctx['starts'] = starts
    return node, ctx, eligible_tier(classified), [compute_bfs_h_table(grid, g) for g in goals]


def save_tiny_model(root, intercept=0.):
    data = dict(schema_version='bdro-model-v1', backend='logistic', feature_names=SYMMETRIC_FEATURE_NAMES,
                mean=[0.] * 24, scale=[1.] * 24, coef=[.1] * 24, intercept=intercept, tau=.75)
    data['model_sha256'] = digest(data)
    path = Path(root) / 'model.json'
    write_new(path, data)
    write_new(str(path) + '.metadata.json', dict(model_sha256=sha(path)))
    return path


class EvidenceTests(unittest.TestCase):
    def test_exact_frozen_budgets(self):
        self.assertEqual([(b.name, b.expansion_cap, b.wall_seconds) for b in BUDGETS],
                         [('q32', 32, .35), ('q128', 128, 1.), ('q512', 512, 3.)])

    def test_exact_comparison_both_directions(self):
        self.assertEqual(compare(ev(n=5), ev(n=6)).label, 0)
        self.assertEqual(compare(ev(n=6), ev(n=5)).label, 1)

    def test_exact_tie_excluded(self):
        pair = compare(ev(), ev())
        self.assertEqual(pair.relation, 'TIE')
        self.assertIsNone(pair.label)

    def test_observed_lower_bound_only(self):
        self.assertEqual(compare(ev(n=5), ev('TIME_CAP', 8)).label, 0)
        self.assertEqual(compare(ev('TIME_CAP', 8), ev(n=5)).label, 1)
        self.assertEqual(compare(ev(n=5), ev('TIME_CAP', 5)).relation, 'UNKNOWN')
        self.assertEqual(compare(ev(n=8), ev('TIME_CAP', 2)).relation, 'UNKNOWN')

    def test_two_censored_are_unknown_even_at_different_caps(self):
        self.assertIsNone(compare(ev('EXPANSION_CAP', 32), ev('TIME_CAP', 8)).label)

    def test_complete_cost_and_solvability_disagreement_fails(self):
        with self.assertRaises(IntegrityError):
            compare(ev(cost=8), ev(cost=9))
        with self.assertRaises(IntegrityError):
            check_consistency([ev(), ev('INFEASIBLE')])

    def test_invalid_status_and_negative_work_rejected(self):
        with self.assertRaises(IntegrityError):
            ev('garbage')
        with self.assertRaises(IntegrityError):
            ev(n=-1)

    def test_best_bound_not_largest_nominal_budget(self):
        history = dict(q32=asdict(ev('EXPANSION_CAP', 32)), q128=asdict(ev('TIME_CAP', 9, 'q128')))
        self.assertEqual(best_evidence(history).expanded, 32)

    def test_completed_candidates_are_never_escalated(self):
        h = [dict(q32=asdict(ev(n=5))), dict(q32=asdict(ev('TIME_CAP', 3)))]
        self.assertEqual(needed_candidates(h, BUDGETS[1]), [1])
        h[1]['q128'] = asdict(ev(n=8, budget='q128'))
        self.assertEqual(needed_candidates(h, BUDGETS[2]), [])

    def test_only_unknown_pairs_escalate(self):
        h = [dict(q32=asdict(ev(n=10))), dict(q32=asdict(ev(n=12))),
             dict(q32=asdict(ev('TIME_CAP', 3)))]
        self.assertEqual(needed_candidates(h, BUDGETS[1]), [2])

    def test_missing_prior_stage_rejected(self):
        with self.assertRaises(IntegrityError):
            needed_candidates([{}, {}], BUDGETS[1])

    def test_independent_comparator_agrees_for_all_status_cases(self):
        for a in ('SOLVED', 'EXPANSION_CAP', 'TIME_CAP'):
            for b in ('SOLVED', 'EXPANSION_CAP', 'TIME_CAP'):
                for x, y in ((3, 5), (5, 3), (3, 3)):
                    ea, eb = ev(a, x), ev(b, y)
                    self.assertEqual(compare(ea, eb).relation, independent_relation(asdict(ea), asdict(eb)))


class IntegrationTests(unittest.TestCase):
    def test_pool_replay_stable_and_residual_first(self):
        _, ctx, candidates, _ = tiny_state()
        pool = candidate_pool(candidates, ctx)
        self.assertIs(pool[0], f_aware_residual_selector(candidates, ctx))
        self.assertEqual([c.conflict for c in pool], [c.conflict for c in candidate_pool(candidates, ctx)])

    def test_real_forced_split_solutions_and_constraints(self):
        node, ctx, candidates, ht = tiny_state()
        saved = copy.deepcopy(node.paths)
        costs = []
        for cc in candidates:
            e = rollout_candidate(node, cc, ctx, ht, BUDGETS[0])
            self.assertTrue(e.forced_split_applied)
            self.assertEqual(e.status, 'SOLVED')
            validate_constrained_paths(ctx['grid'], ctx['starts'], ctx['goals'], e.solution_paths, node.constraints)
            costs.append(e.solution_cost)
        self.assertEqual(len(set(costs)), 1)
        self.assertEqual(node.paths, saved)

    def test_unapplied_split_is_integrity_failure(self):
        with patch('mapf.downstream_teacher.icbs_rollout', return_value=dict(forced_split_applied=False)):
            with self.assertRaises(IntegrityError):
                rollout_candidate(None, None, dict(grid=None, starts=[], goals=[]), [], BUDGETS[0])

    def test_swapping_all_ordered_features_is_invariant(self):
        rng = np.random.default_rng(71)
        raw = rng.normal(size=(20, 24))
        swapped = raw.copy()
        for left, right in ORDERED_PAIRS:
            i, j = FEATURE_NAMES.index(left), FEATURE_NAMES.index(right)
            swapped[:, i], swapped[:, j] = raw[:, j], raw[:, i]
        np.testing.assert_array_equal(symmetrize(raw), symmetrize(swapped))
        self.assertEqual(len(set(SYMMETRIC_FEATURE_NAMES)), 24)

    def test_invalid_feature_schema_and_values_fail(self):
        with self.assertRaises(ValueError):
            symmetrize([0] * 23)
        with self.assertRaises(ValueError):
            symmetrize([float('nan')] * 24)
        with self.assertRaises(ValueError):
            symmetrize([0] * 24, list(reversed(FEATURE_NAMES)))

    def test_frozen_legacy_files_remain_byte_identical(self):
        manifest = json.loads(Path('results/recovery/final_study/frozen_source_manifest.json').read_text())
        for path, expected in manifest.items():
            self.assertEqual(sha(path), expected, path)

    def test_cold_worker_plain_residual_and_model_b_solve(self):
        for policy in ('plain', 'residual', 'model_b'):
            row = run_isolated(dict(instance=dict(width=3, height=2, starts=[[0, 0], [0, 2]],
                                                  goals=[[0, 2], [0, 0]]),
                                    policy=policy, seed=71, timeout_seconds=4,
                                    model_paths=dict(model_b='results/recovery/ml_revision/model_b.json')))
            self.assertEqual(row['status'], 'SOLVED', row)
            self.assertEqual(row['cost'], 6)
            self.assertGreater(row['peak_rss_bytes'], 0)

    def test_cold_worker_override_with_transparent_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_tiny_model(tmp, -20.)
            row = run_isolated(dict(instance=dict(width=3, height=3, starts=[[1, 0], [1, 2], [0, 1]],
                                                  goals=[[1, 2], [1, 0], [2, 1]]),
                                    policy='override', seed=71, timeout_seconds=4, model_paths=dict(override=str(path))))
            self.assertEqual(row['status'], 'SOLVED', row)
            self.assertEqual(row['selector_stats']['overrides'], 0)

    def test_end_to_end_hard_cap_is_enforced(self):
        result = run_isolated(dict(instance={}, policy='plain', seed=71, timeout_seconds=.001))
        self.assertEqual(result['status'], 'HARD_TIMEOUT')


class ModelTests(unittest.TestCase):
    def test_actual_logistic_ridge_fit_and_independent_reload(self):
        from train_override import train, verify_models
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = dict(protocol_sha256='p', source_sha256='s', input_sha256='i',
                     smoke=False, gates=GATES, families=dict(validation=['validation/f']))
            (root / 'rollouts.jsonl').write_text('')
            write_new(root / 'quality.json', dict(integrity_passed=True, headroom_gate_passed=True,
                       raw_rollouts_sha256=sha(root / 'rollouts.jsonl'), stage='q32'))
            pairs = []
            for i in range(12):
                pair = dict(query_id=f't{i}', family_id='train/f', split='train',
                            feature_difference=[float(i % 2)] * 24, label=i % 2,
                            relation='CHALLENGER_BETTER' if i % 2 else 'RESIDUAL_BETTER')
                pair['pair_sha256'] = digest(pair)
                pairs.append(pair)
            (root / 'pairs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pairs))
            queries = [dict(query_id=f'v{i}', family_id='validation/f', split='validation',
                            pairs=[dict(feature_difference=[float(i % 2)] * 24, label=i % 2,
                                        candidate_index=1, relation='CHALLENGER_BETTER' if i % 2 else 'RESIDUAL_BETTER')])
                       for i in range(4)]
            write_new(root / 'queries_q32.json', queries)
            with patch('train_override.read_protocol', return_value=p):
                result = train(root)
            self.assertTrue(result['trained'])
            self.assertTrue(verify_models(root, p))
            self.assertFalse(json.loads((root / 'model_logistic.json').read_text())['validation_used_for_fit'])

    def test_probability_gate_at_below_and_above_tau(self):
        candidates = [object(), object(), object()]
        for probabilities, expected in (([.1, .7499], 0), ([.75, .6], 1), ([.8, .9], 2)):
            model = SimpleNamespace(predict_probability=lambda X: np.asarray(probabilities))
            selector = ResidualOverrideSelector(model)
            selector.extractor = object()
            with patch('mapf.override_policy.candidate_pool', return_value=candidates), patch(
                    'mapf.override_policy.symmetric_features', return_value=np.zeros((3, 24))):
                self.assertIs(selector(candidates, {}), candidates[expected])
            self.assertEqual(selector.stats['overrides'], int(expected != 0))
        with self.assertRaises(ValueError):
            ResidualOverrideSelector(model, tau=.6)

    def test_single_candidate_skips_features_and_model(self):
        candidate = object()
        selector = ResidualOverrideSelector(None)
        with patch('mapf.override_policy.candidate_pool', return_value=[candidate]):
            self.assertIs(selector([candidate], {}), candidate)
        self.assertEqual(selector.stats['single_candidate'], 1)

    def test_model_json_roundtrip_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_tiny_model(tmp, .2)
            X = np.arange(48).reshape(2, 24) / 100
            expected = 1 / (1 + np.exp(-(X @ np.full(24, .1) + .2)))
            np.testing.assert_allclose(SavedOverride(path).predict_probability(X), expected, rtol=1e-14)
            path.write_text(path.read_text() + ' ')
            with self.assertRaises(ValueError):
                SavedOverride(path)

    def test_training_only_and_query_weights(self):
        pairs = [dict(query_id=q, family_id='train/a', split='train', feature_difference=[value] * 24,
                      label=y, relation='CHALLENGER_BETTER' if y else 'RESIDUAL_BETTER')
                 for q, value, y in [('q1', 1, 0), ('q1', 2, 1), ('q2', 3, 1)]]
        pairs.append(dict(query_id='v', family_id='validation/b', split='validation',
                          feature_difference=[1e9] * 24, label=1, relation='CHALLENGER_BETTER'))
        train, X, y, weights = training_arrays(pairs)
        self.assertEqual(len(train), 3)
        self.assertEqual(X.max(), 3)
        for query in ('q1', 'q2'):
            self.assertEqual(sum(w for p, w in zip(train, weights) if p['query_id'] == query), 1.)
        pairs[-1]['split'] = 'test'
        with self.assertRaises(ValueError):
            training_arrays(pairs)

    def test_family_leakage_and_unknown_training_fail(self):
        pairs = [dict(query_id='q', family_id='f', split='train', feature_difference=[0] * 24,
                      label=None, relation='UNKNOWN')]
        with self.assertRaises(ValueError):
            training_arrays(pairs)


class ArtifactAndHoldoutTests(unittest.TestCase):
    def test_deterministic_full_holdout_and_no_overwrite(self):
        from prepare_movingai_holdout import prepare
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            maps = {'random-32-32-20': [20, 28], 'room-32-32-4': [16, 24],
                    'warehouse-10-20-10-2-1': [16, 24]}
            benchmark = base / 'data'
            benchmark.mkdir()
            for name in maps:
                (benchmark / (name + '.map')).write_text('type octile\nheight 8\nwidth 8\nmap\n' + ('........\n' * 8))
                for kind in ('random', 'even'):
                    for index in (1, 2, 3):
                        lines = ['version 1']
                        for a in range(32):
                            lines.append(f'0 {name}.map 8 8 {a % 8} {a // 8} {a % 8} {a // 8 + 4} 4')
                        (benchmark / f'{name}-{kind}-{index}.scen').write_text('\n'.join(lines) + '\n')
            p = dict(protocol_sha256='p', source_sha256='s', input_sha256='i', smoke=False,
                     holdout=dict(selection_seed='bdro-holdout-20261101', map_agents=maps))
            empty = base / 'empty'
            empty.mkdir()
            roots = [base / 'a', base / 'b']
            manifests = []
            for root in roots:
                root.mkdir()
                write_new(root / 'validation.json', dict(gate_passed=True))
                (root / 'model_logistic.json').write_text('{}')
                with patch('prepare_movingai_holdout.read_protocol', return_value=p):
                    manifests.append(prepare(root, benchmark, [empty]))
                    with self.assertRaises(FileExistsError):
                        prepare(root, benchmark, [empty])
            self.assertEqual(manifests[0]['instances'], manifests[1]['instances'])
            self.assertEqual(len(manifests[0]['instances']), 24)
            self.assertEqual(len(manifests[0]['families']), 12)

    def test_exclusive_create_and_idempotent_identical_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact(tmp, 'manifest.json', {'x': 1})
            artifact(tmp, 'manifest.json', {'x': 1})
            with self.assertRaises(FileExistsError):
                artifact(tmp, 'manifest.json', {'x': 2})
            with self.assertRaises(FileExistsError):
                write_new(Path(tmp) / 'manifest.json', {})

    def test_candidate_and_content_hashes_deterministic(self):
        self.assertEqual(digest(dict(a=1, b=2)), digest(dict(b=2, a=1)))
        self.assertEqual(candidate_id(dict(conflict=dict(i=1, j=2))), candidate_id(dict(conflict=dict(j=2, i=1))))

    def test_scenario_coordinates_and_first_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.scen'
            path.write_text('version 1\n0 m.map 3 2 2 1 0 0 3.0\n')
            rows = parse_scen(path)
            self.assertEqual(rows[0].start, (1, 2))
            self.assertEqual(first_k(rows, 1, GridMap(3, 2), 'm.map'), ([(1, 2)], [(0, 0)]))
            with self.assertRaises(ValueError):
                first_k(rows, 2, GridMap(3, 2), 'm.map')
            with self.assertRaises(ValueError):
                first_k(rows, 1, GridMap(3, 2), 'wrong.map')
            path.write_text('version 1\n0 m.map 3 2 9 1 0 0 3.0\n')
            with self.assertRaises(ValueError):
                parse_scen(path)

    def test_family_average_precedes_ratio_and_stratified_bootstrap(self):
        rows = []
        for f in ('a', 'b'):
            for k, seconds in ((16, 1), (24, 9)):
                for repeat in (0, 1):
                    for policy, t in (('residual', 1), ('override', seconds)):
                        rows.append(dict(family_id=f, map_name='map', policy=policy, k=k,
                                         repeat=repeat, status='SOLVED', end_to_end_wall_seconds=t))
        families = family_ratios(rows)
        self.assertEqual([r['ratio'] for r in families], [5, 5])
        np.testing.assert_allclose(stratified_interval(families, 100), [5, 5])


if __name__ == '__main__':
    unittest.main()

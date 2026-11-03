"""Evidence checks for replay, censorship and the versioned feature contract."""
import copy
from dataclasses import replace
import json
import unittest

import numpy as np

from mapf.cardinal import ClassifiedConflict
from mapf.conflicts import Conflict
from mapf.grid import GridMap
from mapf.low_level import (ConstraintSet, space_time_astar, get_astar_expansion_count,
                            reset_astar_call_count)
from mapf.multisignal_teacher import (candidate_features, collect_instance, replay_candidate,
                                     compare_evidence, ReplayIntegrityError, FEATURE_NAMES)


class MultiSignalTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = GridMap(4, 4)
        cls.starts = [(0,2), (1,2), (0,0), (2,1), (3,0), (0,3)]
        cls.goals = [(1,0), (2,1), (2,0), (3,0), (0,0), (2,3)]
        cls.collection = collect_instance(cls.grid, cls.starts, cls.goals,
                                           timeout=10, max_expansions=256)
        cls.query = cls.collection['queries'][0]

    def replay(self, index=0, **kwargs):
        return replay_candidate(self.grid, self.starts, self.goals, self.query, index,
                                timeout=10, **kwargs)

    def test_residual_replay_preserves_full_search(self):
        baseline = self.collection['evidence']
        replayed = self.replay()
        self.assertTrue(replayed['intervention_applied'])
        for key in ('status', 'cost', 'paths', 'nodes_expanded', 'll_expansions', 'astar_calls'):
            self.assertEqual(replayed[key], baseline[key], key)
        for key in ('nodes_expanded', 'll_expansions', 'astar_calls'):
            self.assertEqual(replayed[key], replayed['prefix_'+key]+replayed['suffix_'+key])

    def test_challenger_has_valid_same_optimum(self):
        residual, challenger = self.replay(), self.replay(1)
        self.assertTrue(challenger['solution_validated'])
        self.assertEqual(residual['cost'], challenger['cost'])
        self.assertEqual(residual['query_fingerprint'], challenger['query_fingerprint'])
        self.assertIn(compare_evidence(residual, challenger)['relation'],
                      ('CHALLENGER_BETTER', 'RESIDUAL_BETTER', 'TIE'))

    def test_plain_source_replay(self):
        collected = collect_instance(self.grid, self.starts, self.goals, source_policy='plain', timeout=10)
        query = next(q for q in collected['queries'] if q['baseline_index'] is not None)
        replayed = replay_candidate(self.grid, self.starts, self.goals, query, query['baseline_index'], timeout=10)
        for key in ('status', 'cost', 'paths', 'nodes_expanded', 'll_expansions'):
            self.assertEqual(replayed[key], collected['evidence'][key])

    def test_censored_before_target_is_not_preference(self):
        capped = self.replay(max_expansions=1)
        self.assertEqual(capped['status'], 'MAX_EXPANSIONS')
        self.assertFalse(capped['intervention_applied'])
        self.assertEqual(compare_evidence(self.replay(), capped)['relation'], 'NOT_REACHED')

    def test_expansion_cap_retains_observed_post_target_work(self):
        cap = self.query['prefix_nodes_expanded']
        capped = self.replay(max_expansions=cap)
        self.assertTrue(capped['intervention_applied'])
        self.assertEqual(capped['status'], 'MAX_EXPANSIONS')
        self.assertEqual(capped['suffix_nodes_expanded'], 0)
        self.assertGreater(capped['suffix_ll_expansions'], 0)

    def test_changed_prefix_rejected(self):
        query = copy.deepcopy(self.query)
        query['fingerprint'] = 'bad-prefix'
        with self.assertRaises(ReplayIntegrityError):
            replay_candidate(self.grid, self.starts, self.goals, query, 0, timeout=10)

    def test_strict_lower_bound_and_unknown_equality(self):
        complete = dict(status='SOLVED', intervention_applied=True, cost=5,
                        solution_validated=True, query_fingerprint='q', prefix_ll_expansions=10,
                        suffix_ll_expansions=20)
        censored = dict(complete, status='MAX_EXPANSIONS', cost=None, solution_validated=False,
                        suffix_ll_expansions=21)
        self.assertEqual(compare_evidence(complete, censored)['label'], 0)
        self.assertEqual(compare_evidence(censored, complete)['label'], 1)
        censored['suffix_ll_expansions'] = 20
        self.assertEqual(compare_evidence(complete, censored)['relation'], 'UNKNOWN')
        self.assertIsNone(compare_evidence(censored, censored)['label'])

    def test_different_optimum_or_unvalidated_paths_rejected(self):
        a = self.replay()
        b = dict(a, cost=a['cost'] + 1)
        with self.assertRaises(ReplayIntegrityError):
            compare_evidence(a, b)
        with self.assertRaises(ReplayIntegrityError):
            compare_evidence(a, dict(a, solution_validated=False))

    def test_branch_record_symmetry_preserves_relationships(self):
        paths = [[(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2),(0,2)]]
        cc = ClassifiedConflict(Conflict(0,1,2,'VERTEX',(0,2)), 'SEMI_CARDINAL',
                                1, 0, paths[1], paths[1], 3, 3,
                                alt_total_conflicts_i=4, alt_total_conflicts_j=1)
        context = dict(paths=paths, classified=[cc], pair_counts={(0,1):2},
                       per_agent_constraints=[ConstraintSet({(0,1,3)}),ConstraintSet()])
        original = candidate_features([cc], context)
        swapped = replace(cc, conflict=replace(cc.conflict, agent_i=1,agent_j=0),
                          cost_increase_i=cc.cost_increase_j, cost_increase_j=cc.cost_increase_i,
                          alt_path_i=cc.alt_path_j, alt_path_j=cc.alt_path_i,
                          alt_cost_i=cc.alt_cost_j, alt_cost_j=cc.alt_cost_i,
                          alt_total_conflicts_i=cc.alt_total_conflicts_j,
                          alt_total_conflicts_j=cc.alt_total_conflicts_i)
        np.testing.assert_array_equal(original, candidate_features([swapped], dict(context,classified=[swapped])))
        # Same component-wise minima/maxima, but changed associations, must not
        # collapse to the same feature vector (the previous schema did this).
        altered = replace(cc, alt_total_conflicts_i=1, alt_total_conflicts_j=4)
        self.assertFalse(np.array_equal(original, candidate_features([altered], context)))

    def test_infeasible_branch_is_finite_and_separate(self):
        paths = [[(0,0),(0,1)],[(1,0),(0,0)]]
        cc = ClassifiedConflict(Conflict(0,1,1,'VERTEX',(0,0)), 'CARDINAL',
                                float('inf'),1,None,[(1,0),(1,1),(0,1),(0,0)],
                                float('inf'),3,alt_total_conflicts_i=999999,alt_total_conflicts_j=0)
        context = dict(paths=paths,classified=[cc],per_agent_constraints=[ConstraintSet(),ConstraintSet()])
        feature = candidate_features([cc],context)[0]
        self.assertTrue(np.isfinite(feature).all())
        self.assertEqual(feature[FEATURE_NAMES.index('branch_1_infeasible')],1)

    def test_counter_includes_goal_and_resets(self):
        reset_astar_call_count()
        space_time_astar(GridMap(2,1),(0,0),(0,1),ConstraintSet())
        self.assertEqual(get_astar_expansion_count(),2)
        reset_astar_call_count()
        self.assertEqual(get_astar_expansion_count(),0)

    def test_strict_json_roundtrip(self):
        restored = json.loads(json.dumps(self.collection, allow_nan=False))
        replayed = replay_candidate(self.grid,self.starts,self.goals,restored['queries'][0],0,timeout=10)
        self.assertTrue(replayed['solution_validated'])


if __name__ == '__main__':
    unittest.main()

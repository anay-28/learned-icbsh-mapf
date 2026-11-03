"""Independent small-state checks for the new research campaign.

The reference searches below do not call CBS, conflict detection, the solver's
neighbors, or the production validator. They are finite-state exact oracles,
not performance benchmarks. They cover a useful finite set, not a proof that
every solver instance is correct.
"""
from collections import deque
import heapq
import itertools
import random
import unittest

from mapf.baselines import f_aware_residual_selector
from mapf.grid import GridMap
from mapf.high_level_cbs import PlainCBS
from mapf.icbs_h import ICBShSolver
from mapf.low_level import ConstraintSet, space_time_astar


def adjacent(cell, free):
    r, c = cell
    return [p for p in ((r, c), (r - 1, c), (r + 1, c),
                        (r, c - 1), (r, c + 1)) if p in free]


def independent_soc(free, starts, goals):
    """Dijkstra in joint configurations with optional permanent goal locking.

    A still-active agent pays one per time step even when waiting at its goal.
    It may lock for zero cost once at its goal; locked agents remain obstacles.
    Thus temporary visits to a goal do not prematurely stop the cost clock.
    """
    k = len(starts)
    initial = (tuple(starts), 0)
    best = {initial: 0}
    queue = [(0, initial)]
    while queue:
        value, state = heapq.heappop(queue)
        if best[state] != value:
            continue
        positions, locked = state
        if locked == (1 << k) - 1:
            return value
        successors = []
        for i, p in enumerate(positions):
            if p == goals[i] and not locked & (1 << i):
                successors.append(((positions, locked | (1 << i)), 0))
        options = [[p] if locked & (1 << i) else adjacent(p, free)
                   for i, p in enumerate(positions)]
        for moved in itertools.product(*options):
            if len(set(moved)) != k:
                continue
            if any(positions[i] == moved[j] and positions[j] == moved[i]
                   for i, j in itertools.combinations(range(k), 2)):
                continue
            successors.append(((moved, locked), k - locked.bit_count()))
        for nxt, increment in successors:
            candidate = value + increment
            if candidate < best.get(nxt, float("inf")):
                best[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    return None


def independent_path_cost(free, start, goal, vertices, edges):
    """BFS in position/time, with independent future goal occupancy checks."""
    if (*start, 0) in vertices:
        return None
    last = max([0] + [v[2] for v in vertices] + [e[4] for e in edges])
    # At the end of all constraints, any remaining simple path uses <|V| steps.
    horizon = last + len(free)
    queue, seen = deque([(start, 0)]), {(start, 0)}
    while queue:
        position, t = queue.popleft()
        if position == goal:
            future_vertex = any((r, c) == goal and when > t
                                for r, c, when in vertices)
            future_wait = any((r, c) == goal and (rr, cc) == goal and when > t
                              for r, c, rr, cc, when in edges)
            if not future_vertex and not future_wait:
                return t
        if t >= horizon:
            continue
        for nxt in adjacent(position, free):
            state = (nxt, t + 1)
            if ((*nxt, t + 1) in vertices or
                    (*position, *nxt, t + 1) in edges or state in seen):
                continue
            seen.add(state)
            queue.append(state)
    return None


def assert_independent_solution(test, free, starts, goals, result, expected):
    test.assertEqual(result["status"], "SOLVED", result)
    paths = result["paths"]
    test.assertEqual(len(paths), len(starts))
    test.assertEqual(result["cost"], expected)
    test.assertEqual(sum(len(p) - 1 for p in paths), expected)
    for path, start, goal in zip(paths, starts, goals):
        test.assertEqual(path[0], start)
        test.assertEqual(path[-1], goal)
        test.assertTrue(all(cell in free for cell in path))
        test.assertTrue(all(b in adjacent(a, free) for a, b in zip(path, path[1:])))
    at = lambda p, t: p[min(t, len(p) - 1)]
    for t in range(max(map(len, paths)) + 2):
        test.assertEqual(len({at(p, t) for p in paths}), len(paths))
        if t:
            for a, b in itertools.combinations(paths, 2):
                test.assertFalse(at(a, t - 1) == at(b, t) and
                                 at(b, t - 1) == at(a, t))


class IndependentResearchTests(unittest.TestCase):
    def test_obstacle_instances_under_different_conflict_orders(self):
        rng = random.Random(20261104)
        dimensions = [(3, 3, {(1, 1)}), (3, 3, {(2, 2)}),
                      (4, 3, {(1, 1), (1, 2)})]
        checked = 0
        for width, height, obstacles in dimensions:
            free = {(r, c) for r in range(height) for c in range(width)} - obstacles
            grid = GridMap(width, height, obstacles)
            for _ in range(10):
                starts = rng.sample(sorted(free), 3)
                goals = rng.sample(sorted(free), 3)
                optimum = independent_soc(free, starts, goals)
                # The oracle also detects impossible permutations in narrow maps.
                # Do not force a CBS UNSOLVABLE termination claim on these cases.
                if optimum is None:
                    continue
                for selector, bypass in ((None, True),
                                         (f_aware_residual_selector, True),
                                         (lambda cs, _: cs[-1], False)):
                    with self.subTest(width=width, starts=starts, goals=goals,
                                      bypass=bypass, selector=str(selector)):
                        result = ICBShSolver(grid, conflict_selector=selector,
                                            enable_bypass=bypass).solve(
                            starts, goals, timeout=10)
                        assert_independent_solution(self, free, starts, goals,
                                                    result, optimum)
                checked += 1
        self.assertGreaterEqual(checked, 15)

    def test_agent_initially_at_goal_must_leave_to_allow_passage(self):
        obstacles = {(1, 0), (1, 2)}
        free = {(0, 0), (0, 1), (0, 2), (1, 1)}
        starts, goals = [(0, 1), (0, 0)], [(0, 1), (0, 2)]
        optimum = independent_soc(free, starts, goals)
        self.assertEqual(optimum, 4)
        grid = GridMap(3, 2, obstacles)
        for solver in (PlainCBS(grid), ICBShSolver(grid),
                       ICBShSolver(grid, conflict_selector=f_aware_residual_selector)):
            result = solver.solve(starts, goals, timeout=10)
            assert_independent_solution(self, free, starts, goals, result, optimum)
            self.assertGreater(len(result["paths"][0]), 1)

    def test_low_level_against_independent_temporal_bfs(self):
        rng = random.Random(51921)
        grid = GridMap(3, 3, {(1, 1)})
        free = {(r, c) for r in range(3) for c in range(3)} - {(1, 1)}
        cells = sorted(free)
        for case in range(100):
            start, goal = rng.choice(cells), rng.choice(cells)
            vertices = {(*rng.choice(cells), rng.randrange(9)) for _ in range(10)}
            edges = set()
            for _ in range(10):
                source = rng.choice(cells)
                destination = rng.choice(adjacent(source, free))
                edges.add((*source, *destination, rng.randrange(1, 9)))
            expected = independent_path_cost(free, start, goal, vertices, edges)
            with self.subTest(case=case, start=start, goal=goal):
                path = space_time_astar(grid, start, goal,
                                       ConstraintSet(vertices, edges))
                self.assertEqual(None if path is None else len(path) - 1, expected)
                if path is not None:
                    at = lambda t: path[min(t, len(path) - 1)]
                    self.assertTrue(all(at(t) != (r, c) for r, c, t in vertices))
                    self.assertTrue(all((at(t - 1), at(t)) != ((r, c), (rr, cc))
                                        for r, c, rr, cc, t in edges))

    def test_goal_wait_constraints_after_early_arrival(self):
        grid = GridMap(2, 1)
        free = {(0, 0), (0, 1)}
        vertices, edges = set(), {(0, 0, 0, 0, 20)}
        expected = independent_path_cost(free, (0, 0), (0, 0), vertices, edges)
        self.assertEqual(expected, 20)
        path = space_time_astar(grid, (0, 0), (0, 0), ConstraintSet(vertices, edges))
        self.assertEqual(len(path) - 1, expected)
        self.assertNotEqual(path[19], path[20])


if __name__ == "__main__":
    unittest.main()

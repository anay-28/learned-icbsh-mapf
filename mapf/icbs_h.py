"""
ICBS-h implementation (Improved Conflict-Based Search with MVC Heuristic).
Supports:
  1. Conflict classification (cardinal, semi-cardinal, non-cardinal)
  2. Bypassing conflicts
  3. Conflict graph and Minimum Vertex Cover (MVC) admissible heuristic
  4. Instrumenting conflict choices to the ConflictDataLogger for Learning-to-Rank.
"""
"""
High-Level ICBS-h Solver (Boyarski et al., 2015 + Felner et al., 2018) with pluggable
conflict-prioritization hooks, node classification caching, and strict priority-tier invariants.
"""

import heapq
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from mapf.grid import GridMap
from mapf.low_level import (
    ConstraintSet,
    space_time_astar,
    get_path_cost,
    compute_bfs_h_table,
)
from mapf.conflicts import Conflict, detect_all_conflicts
from mapf.cardinal import (
    ClassifiedConflict,
    classify_conflicts,
    build_constraint_for_agent,
)
from mapf.conflict_graph import CardinalConflictGraph, compute_h4_and_graph


class ICBSNode:
    def __init__(
        self,
        constraints: List[ConstraintSet],
        paths: List[List[Tuple[int, int]]],
        g: int,
        h: int,
        depth: int = 0,
        pair_counts: Optional[Dict[Tuple[int, int], int]] = None,
        node_id: int = 0,
        conflicts_cache: Optional[List[Conflict]] = None,
        classified_cache: Optional[List[ClassifiedConflict]] = None,
        g_cf_cache: Optional[CardinalConflictGraph] = None,
    ):
        self.constraints = constraints
        self.paths = paths
        self.g = g
        self.h = h
        self.f = g + h
        self.depth = depth
        self.pair_counts = dict(pair_counts) if pair_counts else {}
        self.node_id = node_id
        self.conflicts_cache = conflicts_cache
        self.classified_cache = classified_cache
        self.g_cf_cache = g_cf_cache

    def __lt__(self, other: "ICBSNode"):
        if self.f != other.f:
            return self.f < other.f
        if self.h != other.h:
            return self.h < other.h
        # Tie-break toward fewer remaining conflicts if cached
        c_self = len(self.conflicts_cache) if self.conflicts_cache is not None else 999999
        c_other = len(other.conflicts_cache) if other.conflicts_cache is not None else 999999
        if c_self != c_other:
            return c_self < c_other
        return self.node_id < other.node_id


class ICBSHSolver:
    def __init__(
        self,
        grid: GridMap,
        conflict_selector: Optional[
            Callable[[List[ClassifiedConflict], Dict[str, Any]], ClassifiedConflict]
        ] = None,
        enable_bypass: bool = True,
        regime: str = "B",
        strict_tiering: bool = True,
    ):
        """
        conflict_selector: callable taking (candidates, context_dict) -> chosen ClassifiedConflict.
        strict_tiering: if True (default), always filters candidates to the highest non-empty
                        ICBS priority tier (CARDINAL > SEMI_CARDINAL > NON_CARDINAL) before
                        invoking conflict_selector. Prevents Regime B from ever ranking a
                        non-cardinal conflict above a cardinal conflict.
        """
        self.grid = grid
        self.conflict_selector = conflict_selector
        self.enable_bypass = enable_bypass
        self.regime = regime
        self.strict_tiering = strict_tiering

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        timeout: float = 60.0,
        max_expansions: Optional[int] = None,
        on_node_callback: Optional[Callable[[ICBSNode, List[ClassifiedConflict], CardinalConflictGraph], None]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        k = len(starts)

        # Precompute exact 2D Reverse-BFS heuristics once per agent
        h_tables = [compute_bfs_h_table(self.grid, goals[i]) for i in range(k)]
        for i in range(k):
            if starts[i] not in h_tables[i]:
                return {
                    "status": "NO_ROOT_PATH",
                    "cost": None,
                    "nodes_expanded": 0,
                    "runtime": time.perf_counter() - start_time,
                }

        max_base_dist = max(h_tables[i][starts[i]] for i in range(k)) if k > 0 else 0
        max_t = max(max_base_dist + 80, self.grid.width + self.grid.height + 50)

        # Per-solve low-level path cache: (agent_id, constraints_key) -> path
        path_cache: Dict[Tuple[int, Tuple[frozenset, frozenset]], Optional[List[Tuple[int, int]]]] = {}

        root_constraints = [ConstraintSet() for _ in range(k)]
        root_paths = []
        for i in range(k):
            p = space_time_astar(
                self.grid, starts[i], goals[i], root_constraints[i], max_timesteps=max_t, h_table=h_tables[i]
            )
            if p is None:
                return {
                    "status": "NO_ROOT_PATH",
                    "cost": None,
                    "nodes_expanded": 0,
                    "runtime": time.perf_counter() - start_time,
                }
            path_cache[(i, root_constraints[i].freeze_key())] = p
            root_paths.append(p)

        root_g = sum(get_path_cost(p) for p in root_paths)
        root_conflicts = detect_all_conflicts(root_paths)
        if not root_conflicts:
            return {
                "status": "OPTIMAL",
                "cost": root_g,
                "paths": root_paths,
                "nodes_expanded": 1,
                "Bypass_count": 0,
                "decision_nodes": 0,
                "runtime": time.perf_counter() - start_time,
            }

        classified_root, _ = classify_conflicts(
            self.grid,
            root_paths,
            starts,
            goals,
            root_constraints,
            root_conflicts,
            enable_bypass=False,
            max_timesteps=max_t,
            h_tables=h_tables,
            path_cache=path_cache,
        )
        root_h, root_g_cf = compute_h4_and_graph(classified_root)

        node_counter = 0
        root = ICBSNode(
            root_constraints,
            root_paths,
            root_g,
            root_h,
            depth=0,
            node_id=node_counter,
            conflicts_cache=root_conflicts,
            classified_cache=classified_root,
            g_cf_cache=root_g_cf,
        )
        open_list = [root]

        nodes_expanded = 0
        bypass_count = 0
        decision_nodes = 0

        while open_list:
            if time.perf_counter() - start_time > timeout:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "Bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "runtime": time.perf_counter() - start_time,
                }

            if max_expansions is not None and nodes_expanded >= max_expansions:
                return {
                    "status": "MAX_EXPANSIONS",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "Bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "runtime": time.perf_counter() - start_time,
                }

            node = heapq.heappop(open_list)
            nodes_expanded += 1

            conflicts = (
                node.conflicts_cache
                if node.conflicts_cache is not None
                else detect_all_conflicts(node.paths)
            )
            if not conflicts:
                return {
                    "status": "OPTIMAL",
                    "cost": node.g,
                    "paths": node.paths,
                    "nodes_expanded": nodes_expanded,
                    "Bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "runtime": time.perf_counter() - start_time,
                }

            # Classify conflicts and check for immediate bypass if not already cached
            if node.classified_cache is not None and not self.enable_bypass:
                classified = node.classified_cache
                bypass_info = None
            else:
                classified, bypass_info = classify_conflicts(
                    self.grid,
                    node.paths,
                    starts,
                    goals,
                    node.constraints,
                    conflicts,
                    enable_bypass=self.enable_bypass,
                    max_timesteps=max_t,
                    h_tables=h_tables,
                    path_cache=path_cache,
                )

            if bypass_info is not None:
                bp_agent, bp_path, bp_conflict = bypass_info
                new_constraints = [c.copy() for c in node.constraints]
                new_constraints[bp_agent] =

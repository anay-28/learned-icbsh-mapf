"""
Feature extraction module for Learning-to-Rank conflict prioritization in CBS/ICBS-h.
Covers all six statement-named feature families:

1. Corridor Structure:
   Option (c) Distance-transform value was chosen.
   - Justification / Rationale:
     Option (c) directly measures the distance from the collision cell to the nearest obstacle.
     Low values (e.g. 1) indicate tight bottlenecks/corridors with no lateral bypass routes,
     whereas high values indicate open rooms where agents easily circumvent one another.
     Alternative (a) [degree-3+ cells] only measures local graph branching without metric geometry.
     Alternative (b) [alternative shortest path counts] is computationally expensive (O(V+E) per check).
     Option (c) is O(1) at evaluation time (precomputed per map grid), highly interpretable, and clipped to [0, 5].

2. Congestion:
   - local_density_i, local_density_j: Number of other agents whose space-time trajectories
     pass within Manhattan radius <= 2 of the conflict vertex at time t.

3. Conflict Frequency:
   - pair_conflict_count: Running count of conflicts previously detected and resolved between
     agent i and agent j along this branch of the search tree.

4. Goal Distance:
   - goal_dist_i, goal_dist_j: True static shortest-path distance from conflict vertex to agent's goal.

5. Number of Affected Agents:
   - num_agents_in_conflict_graph: Total agents in V(G_CF).
   - cardinal_degree_i, cardinal_degree_j: Degree of agent i and agent j in G_CF.

6. Predicted Downstream Conflicts:
   - downstream_conflict_count_i, downstream_conflict_count_j: One-step lookahead conflict count
     in the re-planned paths found during cardinal classification.

Kept Baseline Features:
   - conflict_type: 2.0 (cardinal), 1.0 (semi-cardinal), 0.0 (non-cardinal)
   - mvc_size: size of MVC(G_CF)
   - conflict_time: timestep t
   - is_edge_conflict: 1.0 if edge, 0.0 if vertex
   - agent_i_path_length, agent_j_path_length
"""
"""
Feature extraction for candidate conflicts at a CBS / ICBS-h high-level node.
Provides both the standard 18-feature suite (covering all 6 statement-named families)
and the 24-feature Cached-Lookahead suite (CL-ICBS-h) at zero additional A* cost.
"""

from collections import deque
from typing import List, Tuple, Dict, Optional
import numpy as np

from mapf.grid import GridMap
from mapf.low_level import get_location, get_path_cost, compute_bfs_h_table
from mapf.conflicts import detect_all_conflicts
from mapf.cardinal import ClassifiedConflict
from mapf.conflict_graph import CardinalConflictGraph

# Exact 18 concrete features covering all 6 required statement families + baseline context
FEATURE_NAMES = [
    # Family 1: Corridor structure (Option c: Distance-Transform Value to nearest obstacle, clipped [0, 5])
    "corridor_score_i",
    "corridor_score_j",
    # Family 2: Congestion (Number of other agents within Manhattan radius <= 2 at timestep t)
    "local_density_i",
    "local_density_j",
    # Family 3: Conflict frequency (Historical collision count for pair (i, j) along CT branch)
    "pair_conflict_count",
    # Family 4: Goal distance (Exact shortest-path distance from conflict cell to each agent's goal)
    "goal_dist_i",
    "goal_dist_j",
    # Family 5: Affected agents (Vertices in G_CF and degrees of agent_i, agent_j in G_CF)
    "num_agents_in_conflict_graph",
    "cardinal_degree_i",
    "cardinal_degree_j",
    # Family 6: Downstream conflicts (Conflicts involving replanned agent in 1-step lookahead)
    "downstream_conflict_count_i",
    "downstream_conflict_count_j",
    # Baseline context features
    "conflict_type",          # 2 = cardinal, 1 = semi-cardinal, 0 = non-cardinal
    "mvc_size",               # Current node h(N) = MVC(G_CF)
    "conflict_time",          # Timestep t of conflict
    "is_edge_conflict",       # 1 if EDGE conflict, 0 if VERTEX conflict
    "agent_i_path_length",    # Cost of agent_i current path
    "agent_j_path_length",    # Cost of agent_j current path
]

# Extended 24-feature vector for CL-ICBS-h (adds 6 zero-cost Cached-Lookahead features)
EXTENDED_FEATURE_NAMES = FEATURE_NAMES + [
    "delta_g_min",                # min(cost_increase_i, cost_increase_j) from cardinal.py
    "delta_g_max",                # max(cost_increase_i, cost_increase_j) from cardinal.py
    "residual_f_gain_min",        # Guaranteed 1-step f increase on cheaper branch using G_CF \ {a}
    "residual_f_gain_max",        # Guaranteed 1-step f increase on harder branch using G_CF \ {a}
    "marginal_conflict_delta_min",# min(marginal_conflict_delta_i, marginal_conflict_delta_j)
    "marginal_conflict_delta_max",# max(marginal_conflict_delta_i, marginal_conflict_delta_j)
]

FAMILY_MAP = {
    "corridor": ["corridor_score_i", "corridor_score_j"],
    "congestion": ["local_density_i", "local_density_j"],
    "conflict_frequency": ["pair_conflict_count"],
    "goal_distance": ["goal_dist_i", "goal_dist_j"],
    "affected_agents": ["num_agents_in_conflict_graph", "cardinal_degree_i", "cardinal_degree_j"],
    "downstream": ["downstream_conflict_count_i", "downstream_conflict_count_j"],
    "cached_lookahead": [
        "delta_g_min",
        "delta_g_max",
        "residual_f_gain_min",
        "residual_f_gain_max",
        "marginal_conflict_delta_min",
        "marginal_conflict_delta_max",
    ],
}


class FeatureExtractor:
    def __init__(self, grid: GridMap, goals: List[Tuple[int, int]]):
        self.grid = grid
        self.goals = goals
        self.dist_transform = self._compute_obstacle_distance_transform()
        self.goal_dist_tables: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        for g in goals:
            if g not in self.goal_dist_tables:
                self.goal_dist_tables[g] = compute_bfs_h_table(self.grid, g)

    def _compute_obstacle_distance_transform(self) -> np.ndarray:
        """
        Option (c) Corridor Feature: Multi-source BFS from all obstacle cells and map borders.
        Clipped to [0, 5]. Narrow 1-cell corridors get value 1; open areas get up to 5.
        """
        H, W = self.grid.height, self.grid.width
        dist = np.full((H, W), 5, dtype=np.int32)
        queue = deque()

        for r in range(H):
            for c in range(W):
                if not self.grid.is_passable(r, c):
                    dist[r, c] = 0
                    queue.append((r, c))
                elif r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    dist[r, c] = 1
                    queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            if d >= 5:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))

        return np.clip(dist, 0, 5)

    def _get_goal_dist(self, agent_idx: int, pos: Tuple[int, int]) -> int:
        g = self.goals[agent_idx]
        table = self.goal_dist_tables.get(g, {})
        return table.get(pos, abs(pos[0] - g[0]) + abs(pos[1] - g[1]))

    def extract_conflict_features(
        self,
        classified: List[ClassifiedConflict],
        paths: List[List[Tuple[int, int]]],
        G_CF: CardinalConflictGraph,
        mvc_size: int,
        pair_counts: Dict[Tuple[int, int], int],
        include_extended: bool = False,
    ) -> np.ndarray:
        """
        Extracts feature matrix of shape (len(classified), 18) if include_extended=False,
        or (len(classified), 24) if include_extended=True.
        """
        n_cols = len(EXTENDED_FEATURE_NAMES) if include_extended else len(FEATURE_NAMES)
        if not classified:
            return np.zeros((0, n_cols), dtype=np.float32)

        k = len(paths)
        num_cf_agents = G_CF.num_vertices()
        rows = []

        # Cache residual MVC per vertex in G_CF so we compute it at most once per agent
        res_mvc_cache: Dict[int, int] = {}

        for cc in classified:
            c = cc.conflict
            i, j = c.agent_i, c.agent_j
            t = c.time
            pos_i = c.loc1
            pos_j = c.loc2 if c.conflict_type == "EDGE" else c.loc1

            # 1. Corridor structure
            corridor_i = float(self.dist_transform[pos_i[0], pos_i[1]])
            corridor_j = float(self.dist_transform[pos_j[0], pos_j[1]])

            # 2. Congestion: other agents within Manhattan radius <= 2 at timestep t
            density_i = 0
            density_j = 0
            for a in range(k):
                if a == i or a == j:
                    continue
                pos_a = get_location(paths[a], t)
                if abs(pos_a[0] - pos_i[0]) + abs(pos_a[1] - pos_i[1]) <= 2:
                    density_i += 1
                if abs(pos_a[0] - pos_j[0]) + abs(pos_a[1] - pos_j[1]) <= 2:
                    density_j += 1

            # 3. Conflict frequency
            pair_key = (min(i, j), max(i, j))
            pair_freq = float(pair_counts.get(pair_key, 0))

            # 4. Goal distance
            g_dist_i = float(self._get_goal_dist(i, pos_i))
            g_dist_j = float(self._get_goal_dist(j, pos_j))

            # 5. Affected agents
            deg_i = float(G_CF.get_degree(i))
            deg_j = float(G_CF.get_degree(j))

            # 6. Downstream conflicts (use cached values from cardinal.py if available)
            if cc.alt_total_conflicts_i < 999999:
                downstream_i = float(cc.alt_total_conflicts_i)
            elif cc.alt_path_i is not None:
                temp_paths_i = list(paths)
                temp_paths_i[i] = cc.alt_path_i
                downstream_i = float(len(detect_all_conflicts(temp_paths_i)))
            else:
                downstream_i = 50.0

            if cc.alt_total_conflicts_j < 999999:
                downstream_j = float(cc.alt_total_conflicts_j)
            elif cc.alt_path_j is not None:
                temp_paths_j = list(paths)
                temp_paths_j[j] = cc.alt_path_j
                downstream_j = float(len(detect_all_conflicts(temp_paths_j)))
            else:
                downstream_j = 50.0

            # Context features
            if cc.classification == "CARDINAL":
                ctype_val = 2.0
            elif cc.classification == "SEMI_CARDINAL":
                ctype_val = 1.0
            else:
                ctype_val = 0.0

            is_edge = 1.0 if c.conflict_type == "EDGE" else 0.0
            len_i = float(get_path_cost(paths[i]))
            len_j = float(get_path_cost(paths[j]))

            row = [
                corridor_i,
                corridor_j,
                float(density_i),
                float(density_j),
                pair_freq,
                g_dist_i,
                g_dist_j,
                float(num_cf_agents),
                deg_i,
                deg_j,
                downstream_i,
                downstream_j,
                ctype_val,
                float(mvc_size),
                float(t),
                is_edge,
                len_i,
                len_j,
            ]

            if include_extended:
                dg_i = min(float(cc.cost_increase_i), 25.0)
                dg_j = min(float(cc.cost_increase_j), 25.0)
                dg_min = min(dg_i, dg_j)
                dg_max = max(dg_i, dg_j)

                if i not in res_mvc_cache:
                    res_mvc_cache[i] = G_CF.residual_mvc_size(i) if deg_i > 0 else mvc_size
                if j not in res_mvc_cache:
                    res_mvc_cache[j] = G_CF.residual_mvc_size(j) if deg_j > 0 else mvc_size

                dh_res_i = float(res_mvc_cache[i] - mvc_size)
                dh_res_j = float(res_mvc_cache[j] - mvc_size)

                rf_i = dg_i + dh_res_i
                rf_j = dg_j + dh_res_j
                rf_min = min(rf_i, rf_j)
                rf_max = max(rf_i, rf_j)

                mc_i = min(float(cc.marginal_conflict_delta_i), 25.0) if cc.marginal_conflict_delta_i < 999999 else 25.0
                mc_j = min(float(cc.marginal_conflict_delta_j), 25.0) if cc.marginal_conflict_delta_j < 999999 else 25.0
                mc_min = min(mc_i, mc_j)
                mc_max = max(mc_i, mc_j)

                row.extend([dg_min, dg_max, rf_min, rf_max, mc_min, mc_max])

            rows.append(row)

        return np.array(rows, dtype=np.float32)

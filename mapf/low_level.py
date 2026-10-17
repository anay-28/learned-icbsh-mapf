"""
Low-level Space-Time A* planner for single-agent pathfinding under constraints.
Includes exact Reverse-BFS 2D heuristic precomputation to eliminate local minima
on complex maps (e.g., Paris_1_256, room-32-32-4).
"""

import heapq
from collections import deque
from typing import List, Tuple, Set, Optional, Dict
from mapf.grid import GridMap

# Constraint types:
# Vertex constraint: (agent_id, r, c, t)
# Edge constraint: (agent_id, r1, c1, r2, c2, t) -> cannot move from (r1,c1) to (r2,c2) arriving at t


class ConstraintSet:
    def __init__(
        self,
        vertex_constraints: Optional[Set[Tuple[int, int, int]]] = None,
        edge_constraints: Optional[Set[Tuple[int, int, int, int, int]]] = None,
    ):
        # Stored per agent: (r, c, t)
        self.vertex_constraints: Set[Tuple[int, int, int]] = (
            set(vertex_constraints) if vertex_constraints else set()
        )
        # Stored per agent: (r1, c1, r2, c2, t)
        self.edge_constraints: Set[Tuple[int, int, int, int, int]] = (
            set(edge_constraints) if edge_constraints else set()
        )

    def add_vertex_constraint(self, r: int, c: int, t: int):
        self.vertex_constraints.add((r, c, t))

    def add_edge_constraint(self, r1: int, c1: int, r2: int, c2: int, t: int):
        self.edge_constraints.add((r1, c1, r2, c2, t))

    def copy(self) -> "ConstraintSet":
        return ConstraintSet(self.vertex_constraints, self.edge_constraints)

    def freeze_key(self) -> Tuple[frozenset, frozenset]:
        """Hashable key for memoizing low-level A* queries."""
        return (frozenset(self.vertex_constraints), frozenset(self.edge_constraints))


def compute_bfs_h_table(grid: GridMap, goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """
    Computes exact 2D shortest-path distances from every reachable passable cell
    to `goal` via Reverse BFS. Completely eliminates Manhattan local minima in
    rooms and around large obstacles.
    """
    if not grid.is_passable(goal[0], goal[1]):
        return {}
    h_table: Dict[Tuple[int, int], int] = {goal: 0}
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        next_d = h_table[(r, c)] + 1
        for nr, nc in grid.get_neighbors(r, c):
            if (nr, nc) not in h_table:
                h_table[(nr, nc)] = next_d
                queue.append((nr, nc))
    return h_table


def get_location(path: List[Tuple[int, int]], t: int) -> Tuple[int, int]:
    """Returns agent's position at timestep t (persisting at goal after path ends)."""
    if t < len(path):
        return path[t]
    return path[-1]


def get_path_cost(path: List[Tuple[int, int]]) -> int:
    """Standard MAPF sum-of-costs metric: number of timesteps until goal is reached and held."""
    if not path:
        return 0
    return len(path) - 1


def space_time_astar(
    grid: GridMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    constraints: ConstraintSet,
    max_timesteps: Optional[int] = None,
    h_table: Optional[Dict[Tuple[int, int], int]] = None,
    conflict_avoidance_table: Optional[Dict[Tuple[int, int, int], int]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds optimal single-agent path from start to goal respecting vertex and edge constraints.
    Goal persistence: once agent reaches goal at t_g, it stays there for all t >= t_g,
    so no vertex constraint at (goal_r, goal_c, t) for t >= t_g can be violated.
    If `conflict_avoidance_table` is provided, breaks f-ties toward paths with fewer
    collisions with other agents' current trajectories.
    """
    if not grid.is_passable(start[0], start[1]) or not grid.is_passable(goal[0], goal[1]):
        return None

    # Auto-compute exact 2D BFS heuristic if not supplied for large/complex grids
    if h_table is None:
        h_table = compute_bfs_h_table(grid, goal)

    if start not in h_table:
        return None

    v_constraints = constraints.vertex_constraints
    e_constraints = constraints.edge_constraints

    if (start[0], start[1], 0) in v_constraints:
        return None

    # Determine the latest timestep a constraint affects the goal cell
    gr, gc = goal
    max_goal_constraint_t = -1
    max_any_constraint_t = 0
    for r, c, t in v_constraints:
        if t > max_any_constraint_t:
            max_any_constraint_t = t
        if r == gr and c == gc and t > max_goal_constraint_t:
            max_goal_constraint_t = t

    if max_timesteps is None:
        base_h = h_table[start]
        max_timesteps = max(base_h + max_any_constraint_t + 60, grid.width + grid.height + 40)

    start_h = h_table[start]
    # Priority queue entries: (f, cat_hits, h, g, r, c, counter)
    counter = 0
    open_list = [(start_h, 0, start_h, 0, start[0], start[1], counter)]
    best_g: Dict[Tuple[int, int, int], int] = {(start[0], start[1], 0): 0}
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    max_expansions = 150000
    expansions = 0

    while open_list:
        f, cat_hits, h, t, r, c, _ = heapq.heappop(open_list)
        if best_g.get((r, c, t), float("inf")) < t:
            continue

        expansions += 1
        if expansions > max_expansions:
            return None

        # Goal condition: at goal and past all goal vertex constraints
        if r == gr and c == gc and t > max_goal_constraint_t:
            path = []
            curr = (r, c, t)
            while curr in came_from:
                path.append((curr[0], curr[1]))
                curr = came_from[curr]
            path.append((curr[0], curr[1]))
            path.reverse()
            return path

        if t + 1 > max_timesteps:
            continue

        nt = t + 1
        # Wait in place + 4 cardinal moves
        neighbors = grid.get_neighbors(r, c)
        neighbors.append((r, c))

        for nr, nc in neighbors:
            nh = h_table.get((nr, nc))
            if nh is None:
                continue
            if (nr, nc, nt) in v_constraints:
                continue
            if (r, c, nr, nc, nt) in e_constraints:
                continue

            next_state = (nr, nc, nt)
            if best_g.get(next_state, 999999999) <= nt:
                continue

            best_g[next_state] = nt
            came_from[next_state] = (r, c, t)
            nf = nt + nh
            ncat = cat_hits
            if conflict_avoidance_table is not None:
                ncat += conflict_avoidance_table.get(next_state, 0)
            counter += 1
            heapq.heappush(open_list, (nf, ncat, nh, nt, nr, nc, counter))

    return None

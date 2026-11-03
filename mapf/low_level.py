"""
Low-level Space-Time A* planner for single-agent pathfinding under constraints.
Includes exact Reverse-BFS 2D heuristic precomputation to eliminate local minima
on complex maps (e.g., Paris_1_256, room-32-32-4).
"""

import heapq
from collections import deque
from typing import List, Tuple, Set, Optional, Dict, Union, Iterable
from mapf.grid import GridMap
from mapf.conflicts import Constraint
from mapf.search_limits import check_deadline, HorizonExceeded

# -- Module-level A* call counter (Task 1.1) --
_ASTAR_CALL_COUNT: int = 0
_ASTAR_EXPANSION_COUNT: int = 0


def reset_astar_call_count() -> None:
    global _ASTAR_CALL_COUNT, _ASTAR_EXPANSION_COUNT
    _ASTAR_CALL_COUNT = 0
    _ASTAR_EXPANSION_COUNT = 0


def get_astar_call_count() -> int:
    return _ASTAR_CALL_COUNT


def get_astar_expansion_count() -> int:
    """Number of non-stale A* states popped, including terminal goal states.

    Reset together with the call counter at each top-level solve. This records
    observed work on interrupted searches too; it excludes reverse-BFS work.
    """
    return _ASTAR_EXPANSION_COUNT



class ConstraintSet:
    def __init__(
        self,
        vertex_constraints: Optional[Set[Tuple[int, int, int]]] = None,
        edge_constraints: Optional[Set[Tuple[int, int, int, int, int]]] = None,
    ):
        self.vertex_constraints: Set[Tuple[int, int, int]] = (
            set(vertex_constraints) if vertex_constraints else set()
        )
        self.edge_constraints: Set[Tuple[int, int, int, int, int]] = (
            set(edge_constraints) if edge_constraints else set()
        )

    def add_vertex_constraint(self, r: int, c: int, t: int):
        self.vertex_constraints.add((r, c, t))

    def add_edge_constraint(self, r1: int, c1: int, r2: int, c2: int, t: int):
        self.edge_constraints.add((r1, c1, r2, c2, t))

    def add_constraint(self, con: Constraint):
        if con.conflict_type.upper() == "VERTEX":
            self.vertex_constraints.add((con.loc1[0], con.loc1[1], con.time))
        else:
            loc2 = con.loc2 if con.loc2 is not None else con.loc1
            self.edge_constraints.add((con.loc1[0], con.loc1[1], loc2[0], loc2[1], con.time))

    def copy(self) -> "ConstraintSet":
        return ConstraintSet(self.vertex_constraints, self.edge_constraints)

    def freeze_key(self) -> Tuple[frozenset, frozenset]:
        """Hashable key for memoizing low-level A* queries."""
        return (frozenset(self.vertex_constraints), frozenset(self.edge_constraints))


def coerce_constraint_set(
    constraints: Optional[Union[ConstraintSet, Iterable[Constraint]]],
    agent_id: Optional[int] = None,
) -> ConstraintSet:
    if constraints is None:
        return ConstraintSet()
    if isinstance(constraints, ConstraintSet):
        return constraints
    cs = ConstraintSet()
    for item in constraints:
        if isinstance(item, Constraint):
            if agent_id is None or item.agent == agent_id:
                cs.add_constraint(item)
        elif isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], tuple):
            # Legacy (agent_id, 'vertex'/'edge', data)
            ag, ctype, data = item
            if agent_id is None or ag == agent_id:
                if ctype == "vertex":
                    cs.add_vertex_constraint(data[0], data[1], data[2])
                else:
                    cs.add_edge_constraint(data[0], data[1], data[2], data[3], data[4])
    return cs


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
        check_deadline()
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
    constraints: Optional[Union[ConstraintSet, Iterable[Constraint]]] = None,
    max_timesteps: Optional[int] = None,
    h_table: Optional[Dict[Tuple[int, int], int]] = None,
    conflict_avoidance_table: Optional[Dict[Tuple[int, int, int], int]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds optimal single-agent path from start to goal respecting vertex and edge constraints.
    """
    global _ASTAR_CALL_COUNT, _ASTAR_EXPANSION_COUNT
    _ASTAR_CALL_COUNT += 1
    if not grid.is_passable(start[0], start[1]) or not grid.is_passable(goal[0], goal[1]):
        return None

    if h_table is None:
        h_table = compute_bfs_h_table(grid, goal)

    if start not in h_table:
        return None

    cs = coerce_constraint_set(constraints)
    v_constraints = cs.vertex_constraints
    e_constraints = cs.edge_constraints

    if (start[0], start[1], 0) in v_constraints:
        return None

    gr, gc = goal
    max_goal_constraint_t = -1
    max_any_constraint_t = 0
    for r, c, t in v_constraints:
        if t > max_any_constraint_t:
            max_any_constraint_t = t
        if r == gr and c == gc and t > max_goal_constraint_t:
            max_goal_constraint_t = t

    max_goal_wait_t = -1
    for r, c, nr, nc, t in e_constraints:
        max_any_constraint_t = max(max_any_constraint_t, t)
        if (r, c) == goal and (nr, nc) == goal:
            max_goal_wait_t = max(max_goal_wait_t, t)
    # After the last constraint, a reachable state needs a simple path of at
    # most |V|-1 steps to the goal. A smaller user horizon is not complete.
    complete_horizon = max_any_constraint_t + len(h_table) - 1
    limited = max_timesteps is not None and max_timesteps < complete_horizon
    if max_timesteps is None:
        max_timesteps = complete_horizon

    start_h = h_table[start]
    counter = 0
    open_list = [(start_h, 0, start_h, 0, start[0], start[1], counter)]
    best_g: Dict[Tuple[int, int, int], int] = {(start[0], start[1], 0): 0}
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}


    while open_list:
        check_deadline()
        f, cat_hits, h, t, r, c, _ = heapq.heappop(open_list)
        if best_g.get((r, c, t), 999999999) < t:
            continue
        _ASTAR_EXPANSION_COUNT += 1


        if r == gr and c == gc and t > max_goal_constraint_t and t >= max_goal_wait_t:
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

    if limited:
        raise HorizonExceeded(f'No path within horizon {max_timesteps}; complete bound {complete_horizon}')
    return None


class SpaceTimeAStar:
    """Legacy wrapper for Phase 0/1 scripts."""
    def __init__(self, grid: GridMap):
        self.grid = grid
        self._h_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        vertex_constraints: Set[Tuple[int, int, int]],
        edge_constraints: Set[Tuple[int, int, int, int, int]],
        max_timesteps: Optional[int] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        if goal not in self._h_cache:
            self._h_cache[goal] = compute_bfs_h_table(self.grid, goal)
        cs = ConstraintSet(vertex_constraints, edge_constraints)
        return space_time_astar(
            self.grid, start, goal, cs, max_timesteps=max_timesteps, h_table=self._h_cache[goal]
        )

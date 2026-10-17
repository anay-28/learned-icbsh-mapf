"""
Low-level space-time A* pathfinding for individual agents with dynamic constraints.
"""
import heapq
from typing import List, Tuple, Set, Dict, Optional
from mapf.grid import Grid
from collections import deque

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def compute_heuristic_map(grid: Grid, goal: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    """
    Reverse BFS from goal to compute exact shortest distance to goal ignoring dynamic obstacles.
    """
    dist: Dict[Tuple[int, int], int] = {goal: 0}
    queue = [goal]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        d = dist[curr]
        for nxt in grid.get_neighbors(curr[0], curr[1]):
            if nxt not in dist:
                dist[nxt] = d + 1
                queue.append(nxt)
    return dist



def compute_bfs_h_table(grid, goal: Tuple[int, int]) -> dict:
    """Exact obstacle-aware shortest path distance from every passable cell to goal."""
    h_table = {goal: 0}
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        d = h_table[(r, c)]
        for nr, nc in grid.get_neighbors(r, c):
            if (nr, nc) not in h_table:
                h_table[(nr, nc)] = d + 1
                queue.append((nr, nc))
    return h_table
    
class SpaceTimeAStar:
    def __init__(self, grid: Grid):
        self.grid = grid
        self._h_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}

    def get_h(self, loc: Tuple[int, int], goal: Tuple[int, int]) -> int:
        if goal not in self._h_cache:
            self._h_cache[goal] = compute_heuristic_map(self.grid, goal)
        return self._h_cache[goal].get(loc, manhattan_distance(loc, goal))

    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        vertex_constraints: Set[Tuple[int, int, int]],  # (x, y, timestep)
        edge_constraints: Set[Tuple[int, int, int, int, int]],  # (from_x, from_y, to_x, to_y, timestep)
        max_timestep: int = 1024
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Space-time A* search for a single agent.
        Path is a list of positions: [pos_at_t0, pos_at_t1, ...].
        Cost of path is its length - 1 (number of actions).
        """
        if (start[0], start[1], 0) in vertex_constraints:
            return None

        # Find maximum constraint time to ensure agent stays at goal if constraints exist later
        max_c_time = 0
        for (_, _, t) in vertex_constraints:
            if t > max_c_time:
                max_c_time = t
        for (_, _, _, _, t) in edge_constraints:
            if t > max_c_time:
                max_c_time = t

        # open_set item: (f_score, g_score, (x, y, timestep), path)
        h0 = self.get_h(start, goal)
        open_set = [(h0, 0, (start[0], start[1], 0), [start])]
        closed_set: Set[Tuple[int, int, int]] = set()

        while open_set:
            f, g, (x, y, t), path = heapq.heappop(open_set)

            if (x, y) == goal and t >= max_c_time:
                return path

            if (x, y, t) in closed_set:
                continue
            closed_set.add((x, y, t))

            if t >= max_timestep:
                continue

            # Possible moves: 4 neighbors + wait in place
            possible_moves = self.grid.get_neighbors(x, y) + [(x, y)]
            next_t = t + 1

            for nx, ny in possible_moves:
                # Check vertex constraint at next_t
                if (nx, ny, next_t) in vertex_constraints:
                    continue
                # Check edge constraint (traverse (x,y) -> (nx,ny) at next_t)
                if (x, y, nx, ny, next_t) in edge_constraints:
                    continue

                if (nx, ny, next_t) in closed_set:
                    continue

                next_g = g + 1
                next_h = self.get_h((nx, ny), goal)
                next_f = next_g + next_h
                heapq.heappush(open_set, (next_f, next_g, (nx, ny, next_t), path + [(nx, ny)]))

        return None

"""
Cardinal Conflict Graph G_CF and exact Minimum Vertex Cover (MVC) for the h4 heuristic (Felner et al., 2018).
Includes Residual MVC analysis for Cached-Lookahead conflict prioritization (CL-ICBS-h).
"""

from typing import List, Set, Dict, Tuple
from mapf.cardinal import ClassifiedConflict


class CardinalConflictGraph:
    def __init__(self, classified_conflicts: List[ClassifiedConflict]):
        self.adj: Dict[int, Set[int]] = {}
        self.edges: Set[Tuple[int, int]] = set()

        for cc in classified_conflicts:
            if cc.classification == "CARDINAL":
                u = cc.conflict.agent_i
                v = cc.conflict.agent_j
                if u > v:
                    u, v = v, u
                self.edges.add((u, v))
                if u not in self.adj:
                    self.adj[u] = set()
                if v not in self.adj:
                    self.adj[v] = set()
                self.adj[u].add(v)
                self.adj[v].add(u)

    def num_vertices(self) -> int:
        return len(self.adj)

    def get_degree(self, agent_id: int) -> int:
        return len(self.adj.get(agent_id, set()))

    def exact_mvc_size(self) -> int:
        """
        Computes the exact Minimum Vertex Cover size of G_CF using branch-and-bound.
        Admissible lower bound h(N) on future cost increases.
        """
        if not self.edges:
            return 0
        edges_list = list(self.edges)
        return self._mvc_Provide(edges_list)

    def residual_mvc_size(self, removed_agent: int) -> int:
        """
        Computes exact MVC size of G_CF after removing `removed_agent` and its incident edges.
        Since replanning `removed_agent` leaves all other agents' trajectories untouched,
        all cardinal conflicts among V \\ {removed_agent} remain cardinal in the child node!
        Thus residual_mvc_size(i) is a provable, zero-A*-cost lower bound on h(Child_i).
        """
        if not self.edges:
            return 0
        rem_edges = [(u, v) for (u, v) in self.edges if u != removed_agent and v != removed_agent]
        if not rem_edges:
            return 0
        return self._mvc_Provide(rem_edges)

    def _mvc_Provide(self, edges_list: List[Tuple[int, int]]) -> int:
        best = [len(self.adj)]

        def solve(remaining_edges: List[Tuple[int, int]], current_size: int):
            if not remaining_edges:
                if current_size < best[0]:
                    best[0] = current_size
                return
            if current_size >= best[0]:
                return

            # Greedy matching lower bound for pruning
            matched = set()
            matching_lb = 0
            for u, v in remaining_edges:
                if u not in matched and v not in matched:
                    matched.add(u)
                    matched.add(v)
                    matching_lb += 1

            if current_size + matching_lb >= best[0]:
                return

            # Pick vertex with highest degree in remaining edges
            deg: Dict[int, int] = {}
            for u, v in remaining_edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            best_v = max(deg, key=deg.get)

            # Branch 1: Include best_v in cover
            rem_without_v = [(u, v) for (u, v) in remaining_edges if u != best_v and v != best_v]
            solve(rem_without_v, current_size + 1)

            # Branch 2: Exclude best_v -> must include all its current neighbors
            neighbors = {v if u == best_v else u for (u, v) in remaining_edges if u == best_v or v == best_v}
            if current_size + len(neighbors) < best[0]:
                rem_without_neighbors = [
                    (u, v) for (u, v) in remaining_edges if u not in neighbors and v not in neighbors
                ]
                solve(rem_without_neighbors, current_size + len(neighbors))

        solve(edges_list, 0)
        return best[0]


def compute_h4_and_graph(
    classified_conflicts: List[ClassifiedConflict],
) -> Tuple[int, CardinalConflictGraph]:
    graph = CardinalConflictGraph(classified_conflicts)
    return graph.exact_mvc_size(), graph

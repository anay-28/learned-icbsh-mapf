"""
Conflict classification (Cardinal, Semi-Cardinal, Non-Cardinal) and Bypass logic (Boyarski et al., 2015).
Includes memoized path caching and exact marginal conflict tracking for CL-ICBS-h.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from mapf.grid import GridMap
from mapf.low_level import ConstraintSet, space_time_astar, get_path_cost
from mapf.conflicts import Conflict, detect_all_conflicts


@dataclass
class ClassifiedConflict:
    conflict: Conflict
    classification: str  # 'CARDINAL', 'SEMI_CARDINAL', 'NON_CARDINAL'
    cost_increase_i: int
    cost_increase_j: int
    alt_path_i: Optional[List[Tuple[int, int]]]
    alt_path_j: Optional[List[Tuple[int, int]]]
    alt_cost_i: float
    alt_cost_j: float
    # Zero-cost cached lookahead conflict metrics
    alt_total_conflicts_i: int = 999999
    alt_total_conflicts_j: int = 999999
    marginal_conflict_delta_i: int = 999999
    marginal_conflict_delta_j: int = 999999


def build_constraint_for_agent(
    conflict: Conflict, agent_id: int, base_constraints: ConstraintSet
) -> ConstraintSet:
    """Returns a new ConstraintSet with the branch constraint added for agent_id."""
    new_c = base_constraints.copy()
    t = conflict.time
    if conflict.conflict_type == "VERTEX":
        r, c = conflict.loc1
        new_c.add_vertex_constraint(r, c, t)
    else:
        # Edge conflict: loc1=(r1,c1) of agent_i at t-1, loc2=(r2,c2) of agent_i at t
        r1, c1 = conflict.loc1
        r2, c2 = conflict.loc2
        if agent_id == conflict.agent_i:
            new_c.add_edge_constraint(r1, c1, r2, c2, t)
        else:
            new_c.add_edge_constraint(r2, c2, r1, c1, t)
    return new_c


def _count_agent_involved_conflicts(
    conflicts: List[Conflict], agent_id: int
) -> int:
    """Counts how many conflicts in a conflict list directly involve `agent_id`."""
    cnt = 0
    for c in conflicts:
        if c.agent_i == agent_id or c.agent_j == agent_id:
            cnt += 1
    return cnt


def classify_conflicts(
    grid: GridMap,
    paths: List[List[Tuple[int, int]]],
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    constraints: List[ConstraintSet],
    conflicts: Optional[List[Conflict]] = None,
    enable_bypass: bool = True,
    max_timesteps: Optional[int] = None,
    h_tables: Optional[List[Dict[Tuple[int, int], int]]] = None,
    path_cache: Optional[Dict[Tuple[int, Tuple[frozenset, frozenset]], Optional[List[Tuple[int, int]]]]] = None,
) -> Tuple[List[ClassifiedConflict], Optional[Tuple[int, List[Tuple[int, int]], Conflict]]]:
    """
    Classifies all conflicts into CARDINAL, SEMI_CARDINAL, or NON_CARDINAL.
    Caches exact alternative paths, cost increases, and marginal conflict deltas.
    If enable_bypass is True and a valid bypass is found (alt_cost == old_cost and
    total conflicts strictly decrease), returns bypass info: (agent_id, new_path, conflict).
    """
    if conflicts is None:
        conflicts = detect_all_conflicts(paths)

    total_conflicts_before = len(conflicts)
    # Count per-agent conflicts in current paths
    agent_base_conflicts: Dict[int, int] = {}
    for c in conflicts:
        agent_base_conflicts[c.agent_i] = agent_base_conflicts.get(c.agent_i, 0) + 1
        agent_base_conflicts[c.agent_j] = agent_base_conflicts.get(c.agent_j, 0) + 1

    classified_list: List[ClassifiedConflict] = []

    for c in conflicts:
        ai = c.agent_i
        aj = c.agent_j
        cost_i = get_path_cost(paths[ai])
        cost_j = get_path_cost(paths[aj])

        # Replan agent_i with constraint from c
        ci = build_constraint_for_agent(c, ai, constraints[ai])
        hi = h_tables[ai] if h_tables is not None else None
        if path_cache is not None:
            key_i = (ai, ci.freeze_key())
            if key_i not in path_cache:
                path_cache[key_i] = space_time_astar(
                    grid, starts[ai], goals[ai], ci, max_timesteps=max_timesteps, h_table=hi
                )
            alt_path_i = path_cache[key_i]
        else:
            alt_path_i = space_time_astar(
                grid, starts[ai], goals[ai], ci, max_timesteps=max_timesteps, h_table=hi
            )
        alt_cost_i = get_path_cost(alt_path_i) if alt_path_i is not None else float("inf")

        # Replan agent_j with constraint from c
        cj = build_constraint_for_agent(c, aj, constraints[aj])
        hj = h_tables[aj] if h_tables is not None else None
        if path_cache is not None:
            key_j = (aj, cj.freeze_key())
            if key_j not in path_cache:
                path_cache[key_j] = space_time_astar(
                    grid, starts[aj], goals[aj], cj, max_timesteps=max_timesteps, h_table=hj
                )
            alt_path_j = path_cache[key_j]
        else:
            alt_path_j = space_time_astar(
                grid, starts[aj], goals[aj], cj, max_timesteps=max_timesteps, h_table=hj
            )
        alt_cost_j = get_path_cost(alt_path_j) if alt_path_j is not None else float("inf")

        inc_i = (alt_cost_i - cost_i) if alt_path_i is not None else float("inf")
        inc_j = (alt_cost_j - cost_j) if alt_path_j is not None else float("inf")

        if inc_i > 0 and inc_j > 0:
            ctype = "CARDINAL"
        elif inc_i > 0 or inc_j > 0:
            ctype = "SEMI_CARDINAL"
        else:
            ctype = "NON_CARDINAL"

        # Compute exact conflict counts along tentative alternative paths
        base_c_i = agent_base_conflicts.get(ai, 1)
        base_c_j = agent_base_conflicts.get(aj, 1)

        if alt_path_i is not None:
            temp_paths_i = list(paths)
            temp_paths_i[ai] = alt_path_i
            confs_i = detect_all_conflicts(temp_paths_i)
            alt_tot_i = len(confs_i)
            agent_c_i = _count_agent_involved_conflicts(confs_i, ai)
            marg_delta_i = agent_c_i - base_c_i
        else:
            alt_tot_i = 999999
            marg_delta_i = 999999

        if alt_path_j is not None:
            temp_paths_j = list(paths)
            temp_paths_j[aj] = alt_path_j
            confs_j = detect_all_conflicts(temp_paths_j)
            alt_tot_j = len(confs_j)
            agent_c_j = _count_agent_involved_conflicts(confs_j, aj)
            marg_delta_j = agent_c_j - base_c_j
        else:
            alt_tot_j = 999999
            marg_delta_j = 999999

        # Check Boyarski Bypass Condition (zero cost increase + strictly fewer total conflicts)
        if enable_bypass and ctype != "CARDINAL":
            if inc_i == 0 and alt_path_i is not None and alt_tot_i < total_conflicts_before:
                return classified_list, (ai, alt_path_i, c)
            if inc_j == 0 and alt_path_j is not None and alt_tot_j < total_conflicts_before:
                return classified_list, (aj, alt_path_j, c)

        classified_list.append(
            ClassifiedConflict(
                conflict=c,
                classification=ctype,
                cost_increase_i=int(inc_i) if inc_i != float("inf") else 999999,
                cost_increase_j=int(inc_j) if inc_j != float("inf") else 999999,
                alt_path_i=alt_path_i,
                alt_path_j=alt_path_j,
                alt_cost_i=alt_cost_i,
                alt_cost_j=alt_cost_j,
                alt_total_conflicts_i=alt_tot_i,
                alt_total_conflicts_j=alt_tot_j,
                marginal_conflict_delta_i=marg_delta_i,
                marginal_conflict_delta_j=marg_delta_j,
            )
        )

    return classified_list, None

"""Bounded interventional continuations; censored work is never an exact target."""
from dataclasses import asdict, dataclass, field
from typing import Optional

from mapf.repair_pilot import icbs_rollout


class IntegrityError(ValueError):
    """Evidence cannot support a scientific comparison."""


@dataclass(frozen=True)
class RolloutBudget:
    name: str
    expansion_cap: int
    wall_seconds: float


BUDGETS = (RolloutBudget('q32', 32, .35), RolloutBudget('q128', 128, 1.),
           RolloutBudget('q512', 512, 3.))


@dataclass
class RolloutEvidence:
    budget: str
    status: str
    expanded: int
    astar_calls: Optional[int] = None
    forced_split_applied: bool = False
    wall_seconds: float = 0.
    cpu_seconds: float = 0.
    solution_cost: Optional[int] = None
    solution_validated: bool = False
    solution_paths: Optional[list] = None

    def __post_init__(self):
        if self.status not in ('SOLVED', 'INFEASIBLE', 'EXPANSION_CAP', 'TIME_CAP', 'HORIZON_LIMIT'):
            raise IntegrityError(f'Unknown continuation status: {self.status}')
        if type(self.expanded) is not int or self.expanded < 0:
            raise IntegrityError('Invalid observed expansion count')

    @property
    def is_complete(self):
        return self.status in ('SOLVED', 'INFEASIBLE')

    @property
    def lower_bound(self):
        return self.expanded


@dataclass(frozen=True)
class PairEvidence:
    label: Optional[int]
    relation: str
    budget: str
    residual_lb: int
    challenger_lb: int


def check_consistency(evidence):
    complete = [e for e in evidence if e.is_complete]
    if len({e.status for e in complete}) > 1:
        raise IntegrityError('Completed continuations disagree on solvability')
    if len({e.solution_cost for e in complete if e.status == 'SOLVED'}) > 1:
        raise IntegrityError('Completed continuations disagree on optimal cost')


def compare(residual, challenger):
    """Strict exact/lower-bound comparisons only; equality to a bound is unknown."""
    check_consistency([residual, challenger])
    relation = 'UNKNOWN'
    if residual.is_complete and challenger.is_complete:
        relation = ('RESIDUAL_BETTER' if residual.expanded < challenger.expanded else
                    'CHALLENGER_BETTER' if challenger.expanded < residual.expanded else 'TIE')
    elif residual.is_complete and residual.expanded < challenger.lower_bound:
        relation = 'RESIDUAL_BETTER'
    elif challenger.is_complete and challenger.expanded < residual.lower_bound:
        relation = 'CHALLENGER_BETTER'
    budget = max((residual.budget, challenger.budget), key=lambda n: [b.name for b in BUDGETS].index(n))
    return PairEvidence({'CHALLENGER_BETTER': 1, 'RESIDUAL_BETTER': 0}.get(relation),
                        relation, budget, residual.lower_bound, challenger.lower_bound)


def best_evidence(history):
    values = [v if isinstance(v, RolloutEvidence) else RolloutEvidence(**v) for v in history.values()]
    if not values:
        return None
    check_consistency(values)
    complete = [v for v in values if v.is_complete]
    if complete and len({v.expanded for v in complete}) != 1:
        raise IntegrityError('Completed work changed with budget')
    if complete and any(v.expanded > complete[0].expanded for v in values):
        raise IntegrityError('Censored work exceeds known exact work')
    return complete[0] if complete else max(values, key=lambda v: v.lower_bound)


def needed_candidates(histories, budget):
    """Escalate only unresolved pairs, reusing exact completion at any budget."""
    if budget == BUDGETS[0]:
        return [i for i, h in enumerate(histories) if budget.name not in h]
    previous = BUDGETS[[b.name for b in BUDGETS].index(budget.name) - 1].name
    if any(not h for h in histories):
        raise IntegrityError('Run Q32 for every candidate first')
    needed = set()
    residual = best_evidence(histories[0])
    for i in range(1, len(histories)):
        challenger = best_evidence(histories[i])
        if compare(residual, challenger).relation != 'UNKNOWN':
            continue
        for j, evidence in ((0, residual), (i, challenger)):
            if not evidence.is_complete and budget.name not in histories[j]:
                if previous not in histories[j]:
                    raise IntegrityError(f'Unresolved candidate skipped {previous}')
                needed.add(j)
    return sorted(needed)


def rollout_candidate(node, cc, context, h_tables, budget):
    raw = icbs_rollout(node, cc, context['grid'], context['starts'], context['goals'],
                       h_tables, budget.expansion_cap, budget.wall_seconds)
    if not raw['forced_split_applied']:
        raise IntegrityError('Requested first split was not applied')
    if raw['status'] == 'SOLVED' and not raw.get('solution_validated'):
        raise IntegrityError('Unvalidated solution')
    fields = RolloutEvidence.__dataclass_fields__
    return RolloutEvidence(budget=budget.name, **{k: v for k, v in raw.items() if k in fields})

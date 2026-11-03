# Independent solver and research audit

Audit date: 2026-11-04. Baseline inspected: `fb24ecc` on
`bdro-two-day-study`. This audit is distinct from the new campaign's empirical
performance results.

## Independent finite checks

Run from the repository root:

```powershell
python -m unittest -v test_independent_research
```

The four test methods passed on the local Python 3.11 environment. They include:

- 30 reproducibly sampled three-agent instances on three obstacle maps. An
  independent joint-state Dijkstra oracle found 19 solvable instances and 11
  impossible configurations. Each solvable instance was checked under baseline
  selection, residual selection, and last-eligible-conflict selection with bypass
  disabled: 57 solver comparisons. Every returned cost matched the exact oracle,
  and every path passed independent movement, collision, endpoint, and persistent
  goal checks. The test deliberately makes no CBS termination assertion for the
  11 impossible configurations.
- 100 random constrained single-agent instances compared with a separate temporal
  BFS, including vertex and directed edge restrictions. Returned path lengths or
  infeasibility agreed in all cases, and paths respected constraints after arrival.
- A T-shaped instance in which an agent starting at its own goal must temporarily
  leave to let another pass. CBS and both ICBS-h selectors returned the exact SOC
  of four.
- A goal waiting-edge restriction at time 20, requiring departure despite the
  start already equaling the goal. The exact arrival-and-remain cost is 20.

The new reference searches do not call production CBS, conflict detection,
neighbor generation, or solution validation. These are finite regression checks;
they are not a general proof of optimality. The repository also contains earlier
exhaustive 2x2 and sampled open-grid tests in `mapf/test_recovery.py`.

The combined supported suite was also run during implementation:

```powershell
python -m unittest -v mapf.test_recovery mapf.test_day3 mapf.test_repair mapf.test_ml_revision test_final_study test_counterfactual_override test_independent_research
```

It ran 72 tests in 20.511 seconds: 70 passed and two historical source-freeze
assertions failed. `test_approved_source_and_models_unchanged` compares the
modified source tree with the historical frozen digest;
`test_frozen_legacy_files_remain_byte_identical` first reports a differing byte
hash for the git-unmodified `mapf/benchmark_worker.py`. This Windows checkout
uses `core.autocrlf=true`, so filesystem byte identity with Linux artifacts is
not guaranteed. These failures were retained, not rewritten as passes; no
algorithmic or integration test failed in that run. The full run log is saved
in the task workspace's `work/supported-tests-audit.log`.

## Scientific distinctions

The supplied paper, *Learning Node-Selection Strategies in Bounded-Suboptimal
Conflict-Based Search for Multi-Agent Path Finding* (Huang, Dilkina, and Koenig,
AAMAS 2021, pp. 611-619), learns which CT node to expand from ECBS's focal list.
This repository instead changes conflict selection inside optimal ICBS-h. The
paper is related work, not a direct reproduction target or a matched baseline.

The existing `initial_node` continuation discards the global OPEN frontier,
recomputes root heuristics, and does not restore the saved node's `bypass_steps`.
Its outcomes measure a restarted descendant problem, not the remaining runtime
of the original full search. They must not be represented as whole-solver
counterfactual outcomes. A deterministic replay intervention inside the existing
solve loop can preserve the frontier and bypass state; prefix replay cost and
suffix intervention cost must be recorded separately.

The inherited solver exposes A* **calls**, not low-level node expansions. These
are different quantities. Deadline interruptions can return null high-level
expansions unless the caller also keeps the cooperative `progress` dictionary.
Any new expansion instrumentation must explicitly count low-level node pops and
retain partial counts on interruption.

The historical path-cache key formats differ: classification uses
`(agent, frozen_constraints, max_timesteps)`, whereas ordinary replanning uses
`(agent, frozen_constraints)`. This can miss reuse; no invalid-solution effect was
found. Changing cache behavior during a study changes its implementation and
requires a new frozen code identity before timing comparisons.

## Interpretation limits

- Preserving complete two-way CBS splitting, exact constrained low-level search,
  admissible high-level bounds, and best-first node ordering is essential; a
  learned conflict selector should only change the eligible conflict choice.
- A solved run must retain its paths for independent verification. Agreement
  between two variants of the same solver is weaker evidence than comparison
  with the independent joint-state reference.
- Few remaining conflicts, a small OPEN list, or a larger cost lower bound is
  not proof of lower runtime. These can be auxiliary signals only.
- One fixed 8x8 benchmark map supports conclusions about held-out scenarios on
  that map. It does not establish transfer to new map topologies or superiority
  to current state-of-the-art MAPF solvers.
- Unfinished runs and unresolved candidate comparisons remain censored evidence;
  they must not be relabeled as exact losses or successful solutions.

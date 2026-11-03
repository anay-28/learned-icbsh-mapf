# Full-solver multi-signal study on empty-8-8

Protocol version: MS8-v1. Implementation and local computation authorized by the
current user request. This is a new study, not a change to any historical gate.

## Research claim and limits

Question: can a small learned conflict selector reduce total runtime of this
Python optimal ICBS-h implementation on new scenarios of one known map?
Learning conflict selection itself is not new: Huang, Dilkina and Koenig,
AAAI 2021, https://arxiv.org/abs/2012.06005. The supplied AAMAS 2021 paper,
*Learning Node-Selection Strategies in Bounded-Suboptimal Conflict-Based Search*,
instead learns ECBS node ordering. We do not claim to reproduce it.

The defensible contribution is an auditable experiment comparing single-work
and multiple-work-signal supervision using complete-search interventions,
explicit censoring, and deployment runtime. A positive result is not promised.
This small study cannot establish state of the art, unseen-map transfer,
warehouse performance, or statistical independence of agent-count prefixes.

## Preserve and repair

Use anay-28/learned-icbsh-mapf at fb24eccc as the additive base. All 233 files
from AvniMittal07 main at 625cda82 are already present byte-identically.
Preserve Model B's inconclusive historical result and BDRO's failed 10% gate.
New code and results use multisignal names. Retain the old 24-feature Model B
unchanged; new features use coherent canonical branch records, finite-value
checks, explicit infeasibility, and consistently defined cached quantities.

## Task and independent correctness

Use the official MovingAI empty-8-8 map (64 free cells), four-neighbor movement
plus waiting, vertex and swap conflicts, persistent goal occupancy, and optimal
sum of path costs. Verify downloads and SHA-256 hashes, strict map dimensions,
scenario coordinate conversion, uniqueness, reachability and cross-file duplicate
instance fingerprints. Scenario distances use the benchmark convention and are
not treated as four-neighbor optimal costs.

Check all returned solutions independently and compare costs across successful
methods. Extend existing small-instance optimality checks with an independent
joint-state shortest-path oracle and constrained single-agent BFS. Learning only
selects an eligible conflict; it cannot change bounds, prune children or relax
constraints. Finite tests are evidence, not a universal correctness proof.

## Splits and pilot

Official inventory: 25 even and 25 random files, each supporting up to 32 agents.
Fixed seed 20261104; stratify by type. Allocate 4 pilot, 26 train, 8 validation,
12 test files. All prefixes, source policies, states, labels and repetitions
from a file stay together. Freeze the split before solving. Test files are
available only for provenance/geometry checks until final model freeze.

Pilot: counts 4, 8, 12, 16, 24; plain ICBS-h and residual; 8-second caps;
at most 10 minutes elapsed. Select two useful counts using pilot results only:
prefer counts with nontrivial median runtime and at least half completed runs.
If too few satisfy this, retain the most informative available counts and
explicitly report feasibility limitations. No hidden retuning using test data.

## Teacher and data generation

Sample fixed decision ordinals (initially 0, 3, 15) on residual and plain
trajectories. Use the existing residual-first pool of at most six eligible
conflicts, chosen before outcomes. Preserve collection failures, empty-query
instances and unresolved labels. Each query stores source policy, state and
candidate fingerprint, decision-time features and provenance.

For each candidate, replay the entire original solve deterministically and
change that single selection. Verify the target fingerprint before applying.
This reconstructs the real OPEN frontier, caches and bypass history. A local
subtree restart is not used as ground truth. Store prefix/suffix accounting,
full elapsed wall/CPU, HL expansions, true LL expanded states, A* calls, status,
intervention applied, cost, paths and validation result separately.

Collection budgets: up to 4096 HL expansions and 15 seconds per replay,
4 workers initially, each with one numerical thread. Initial scope is all
train/validation families at two pilot-selected counts and both source policies.
Reserve up to 4 elapsed hours for collection, with resumable per-job outputs;
this is an operational ceiling, not a promise that every label will resolve.
Explicit interruption leaves the dataset incomplete. Completed outputs are
never silently rerun or replaced. Wall-time censoring under contention remains
visible; expansion counts are the teacher's main evidence.

Primary labels: strict LL work comparisons when both outcomes complete, or one
completed value is strictly below another run's observed lower bound. Require
actual target application and validated equal-cost solutions where available.
Ties and overlapping censored bounds are unknown. HL work is a distinct
auxiliary signal, added only for otherwise unresolved LL comparisons at fixed
weight 0.2. No arbitrary sum of time, frontier size and conflicts is called
ground truth. This is a multi-work-signal teacher, not a multi-bound oracle.
Do not invent an upper bound, a completion probability or a speedup label.

## Fitting and validation

Compare confirmed LL supervision against confirmed LL plus auxiliary HL
supervision with identical features, model capacity, splits and data. Fit one
linear shared candidate score using pairwise logistic loss, no intercept,
training-only scale normalization and family-balanced weights. A symmetry
test covers whole branch records. Serialize plain JSON parameters and hashes.

Require at least 10 strict examples of each original label direction and three
informative training families before fitting. Auxiliary labels cannot satisfy
this gate. A failed gate produces an explicit residual fallback artifact.
Thresholds 0.5, 0.6, 0.7, 0.8, 0.9 and disabled (1.01) are selected using
complete-solver capped runtime on validation families, serially. Include a
matched random-pool control and historical Model B. Record harmful/beneficial
teacher overrides separately from final runtime effects. Local pair accuracy
is not evidence of deployment speed.

## Frozen final benchmark

Before test, freeze source archive/hash, split, model bytes, thresholds,
schedule, analysis and resource limits. Stop collection and training. Run
fresh processes serially, one numerical thread, randomized balanced order:
12 families x 2 counts x 6 policies x 2 repeats x 20 seconds = 5760 nominal
solver seconds (96 minutes), within a two-hour elapsed final ceiling. Policies:
plain ICBS-h, residual, matched pool random, Model B, confirmed, confirmed+aux.
Include feature extraction/inference and solver initialization in runtime;
record process startup separately. Enforce an external timeout and 3 GiB
per-worker memory ceiling. Retain crashes and timeouts as capped failures.

Primary estimator: geometric mean of scenario-family ratios of mean capped
runtime (new confirmed+aux / residual). Aggregate repeats and agent counts
inside each family; stratified family bootstrap 10,000 replicates, fixed seed.
Report solved counts, valid equal costs, paired expansions, overhead, all six
comparisons and confirmed+aux versus confirmed ablation. A positive primary
claim requires upper 95% interval below 1 and no lower solved count. Secondary
comparisons are descriptive, without multiple-testing victory claims. If any
jobs are missing, publish incomplete coverage and no confirmatory claim.

## Evidence and delivery

Each job has a deterministic ID, source/config/input hashes, seed, limits,
status, raw metrics, timestamps, exit code and log. Resume verifies identity.
Verify duplicates, completeness, leakage, valid solutions and cost agreement.
Reports and tables are generated from raw artifacts. Deliver this revised plan,
source, official input manifest, real generated data, fitted/fallback models,
serial benchmark, automated integrity checks and an honest results report.
Never fabricate missing data or relabel an unresolved rollout as a win.

## Reconciliation with the supplied 76-section MSCO plan

The full earlier plan was supplied after initial implementation. It is a
reference design, not evidence that its components have already been built.
The current implementation is the first controlled work-signal study below;
it is not the full three-oracle MSCO system.

Retain: unknown versus bad, separate raw evidence and derived labels, fixed
family splits, cheap deployed features, shared pairwise scoring, worker-level
parallelism, independent correctness checks and mandatory serial runtime tests.

Revise local-only snapshots to complete-solver replay for the initial study.
This costs repeated prefix work but measures the actual intervention while
preserving OPEN and bypass history. Measure prefix overhead explicitly. An
eventual serialized full-frontier snapshot can optimize this without changing
the estimand. Local-subtree intrinsic difficulty remains a separate ablation.

Keep highest-tier candidate selection in this first comparison. All-conflict
eligibility changes both coverage and bypass interactions and must be a
separate versioned experiment; it cannot be attributed to better labels.
The existing pool is not invariant to all agent renamings; only the new feature
records have the tested branch-symmetry property.

The initial teacher uses LL work as primary and HL work as auxiliary. This is
an explicit departure from the proposed HL-primary teacher, motivated by
variable replanning cost. Both are measured compute proxies. The proposal that
label quality is the main bottleneck remains a hypothesis, not a conclusion.

Next research stage, before claiming full MSCO: instrument a single search trace
at exact post-split horizons 8/16/32/64/128/256; prove checkpoint consistency;
compare coverage and agreement against actually completed deep outcomes; retain
unresolved deep results as unresolved. Report the requested horizon/coverage
plots and oracle ablation table. A stable frontier heuristic can be consistently
wrong, so stability alone does not establish reliability.

Only then add D_phi harvesting, with recorded parent/bypass relationships,
co-occurring frontier pairs and cross-fitting by scenario family. Distance to
an observed solution is an upper bound on the nearest possible solution-leaf
distance unless the relevant tree is exhausted; never call it exact shortest
distance merely because one leaf was harvested. The paper-style 54 features
and bucket boundaries are a baseline requiring independent validation.
Confidence constants such as 0.75 are loss weights, not measured probabilities.

DAgger and curriculum follow a demonstrated teacher benefit on development
instances. Their own datasets, rounds and fixed student comparisons are needed
before claiming reduced distribution shift. No D_phi, multi-horizon fusion,
direct composite-oracle result or DAgger result is currently claimed.

Input-audit amendment before pilot: official 32-agent files include duplicate
assignment sets across even/random files. Exclude 32 entirely in this version;
all remaining 4/8/12/16/24 fingerprints are unique. This avoids moving related
cases across the frozen splits. On this laptop only about 1.5 GiB RAM was free
at inspection, despite 14 physical cores; start with two workers, not ten.

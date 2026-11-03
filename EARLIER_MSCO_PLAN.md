 Implementation Plan: Multi-Signal Oracle for Learned Conflict Selection in ICBS-h

## 0. Research objective

The existing Model B showed that a model can imitate a one-step teacher reasonably well while still failing to improve ICBS-h globally.

The new hypothesis is:

\[
\boxed{
\text{The main bottleneck is label quality, not merely model capacity.}
}
\]

Instead of teaching the ranker:

\[
\text{"which conflict gives better immediate children?"}
\]

we want to approximate:

\[
\boxed{
Q(N,c)
=
\text{future CT search effort caused by resolving conflict }c
}
\]

at current CT node \(N\).

The ideal but computationally expensive oracle would be:

\[
Q_\infty(N,c)
=
\text{number of CT expansions required to reach an optimal solution}
\]

after forcing \(c\) and then continuing ICBS-h.

Because full rollouts are often infeasible, we construct a **Multi-Signal Counterfactual Oracle (MSCO)** using three sources of evidence:

\[
O_{\text{MH}}
=
\text{multi-horizon search progress}
\]

\[
O_{\text{roll}}
=
\text{exact/censored rollout-to-solution}
\]

\[
O_{\text{dist}}
=
\text{learned distance-to-solution signal }D_\phi.
\]

These are not averaged blindly.

They are fused hierarchically according to reliability.

---

# 1. Fixed design decisions

These should be frozen before implementation begins.

### 1.1 Counterfactual scope

For current CT node \(N\) and candidate conflict \(c\):

- clone only \(N\);
- force \(c\);
- generate its two CBS children;
- create a new **local OPEN** containing only those children;
- completely ignore the original solver's global OPEN;
- continue normal ICBS-h inside this local subtree.

Therefore the oracle answers:

\[
\boxed{
\text{"How intrinsically difficult is the subtree produced by choosing }c\text{?"}
}
\]

rather than:

\[
\text{"What would happen to the entire global CBS search right now?"}
\]

This makes the label much more reusable across states and substantially simplifies parallelization.

---

## 1.2 Expansion definition

A rollout budget \(B\) means:

\[
\boxed{
B=\text{total high-level CT nodes expanded after forcing the conflict}
}
\]

not \(B\) per child.

Immediately after splitting:

```text
                  N
                  |
             force conflict c
                /   \
              L       R
               \     /
             local OPEN

expansions = 0
```

When one node is removed from local OPEN and processed:

\[
expansions \leftarrow expansions+1.
\]

Checkpoints are recorded after exactly:

\[
8,16,32,64,128,256,\ldots
\]

total local CT expansions.

If a solution is popped after 47 expansions:

\[
E(c)=47.
\]

---

## 1.3 Final target

Primary teacher quantity:

\[
\boxed{\text{high-level CT expansions}}
\]

Final real-world evaluation quantity:

\[
\boxed{\text{wall-clock runtime}}
\]

Additional diagnostics:

\[
\text{generated CT nodes}
\]

\[
\text{low-level A* expansions}
\]

\[
\text{solution depth}
\]

\[
\text{MVC computations}
\]

\[
\text{feature/model inference overhead}.
\]

Runtime should not be used as the primary teacher because it is noisier than deterministic expansion counts.

---

## 1.4 Conflict eligibility

At deployment:

\[
\boxed{
C(N)=\text{all detected conflicts at }N.
}
\]

We are no longer restricting the ranker to ICBS-h's highest conflict-priority class.

That is a significant experimental change.

ICBS traditionally prioritizes cardinal conflicts because resolving them immediately raises cost and often shrinks the CT; bypassing is applicable to semi-/non-cardinal conflicts rather than cardinal ones.

Therefore:

- high-level node ordering remains unchanged;
- \(g,h,f\) remain unchanged;
- admissible ICBS-h heuristic remains unchanged;
- splitting remains unchanged;
- low-level planner remains unchanged;
- bypass correctness remains unchanged;
- only `choose_conflict()` is replaced.

If the learned model chooses a conflict whose cardinality has not yet been computed, classify **that selected conflict** before applying bypass logic.

Do not require expensive cardinality classification for every candidate merely to feed the ML model unless your existing solver already has it cached.

---

# 2. Overall architecture

The complete system should look like:

```text
                         ICBS-h state N
                               |
                               v
                    detect ALL conflicts
                     C(N) = {c1,...,ck}
                               |
                        cheap features
                               |
                               v
                   oracle-query candidates
                   (training only; subset if
                    conflict count is huge)
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
       force candidate c                  state metadata
              |
              v
       local ICBS-h subtree
              |
    +---------+----------+----------------+
    |                    |                |
    v                    v                v
multi-horizon        solution found     frontier nodes
checkpoints          / censored               |
O_MH                 O_roll                   |
                                             v
                                      node model D_phi
                                             |
                                             v
                                          O_dist
    |                    |                    |
    +--------------------+--------------------+
                         |
                         v
               confidence-aware fusion
                         |
                +--------+--------+
                |        |        |
                v        v        v
              ci>cj     tie     unknown
                                  |
                                  v
                         adaptive deepening
                                  |
                                  v
                          pairwise labels
                                  |
                                  v
                           conflict ranker
                                  |
                                  v
                            production
                              ICBS-h
```

A major engineering goal is:

\[
\boxed{
\text{Do not implement three completely separate rollout engines.}
}
\]

Oracle 1 and Oracle 2 should be outputs from the **same search trace**.

---

# 3. Repository/module architecture

I recommend keeping the baseline solver untouched as much as possible.

A clean structure would be approximately:

```text
src/
    icbsh/
        solver.*
        high_level.*
        low_level.*
        conflict.*
        heuristics.*

    oracle/
        oracle_snapshot.*
        local_subtree_runner.*
        checkpoint.*
        rollout_result.*
        candidate_sampler.*
        dphi_runtime.*
        label_fusion.*

    ml/
        conflict_features.*
        model_runtime.*

tools/
    oracle_worker.cpp
    validate_snapshot.cpp
    benchmark_solver.cpp

scripts/
    build_state_manifest.py
    build_oracle_manifest.py
    run_oracle_jobs.py
    merge_oracle_results.py

    build_dphi_dataset.py
    train_dphi.py
    evaluate_dphi.py

    fuse_oracle_labels.py
    analyze_horizons.py
    train_conflict_ranker.py

    run_benchmark.py
    analyze_benchmark.py

configs/
    oracle_pilot.yaml
    oracle_production.yaml
    benchmark.yaml

data/
    snapshots/
    oracle_raw/
    oracle_merged/
    dphi/
    labels/
    models/
    benchmark/
```

Do not mix oracle-specific behavior into the production ICBS-h implementation.

The production solver should ideally know only:

```text
score_conflict(features)
```

and nothing about rollout generation.

---

# 4. Phase 0 — freeze and verify the baseline

Before touching the oracle, freeze the current ICBS-h implementation.

Create a baseline tag/commit such as:

```text
baseline/icbsh-pre-multioracle
```

Record for every benchmark instance:

```text
instance_id
map
scenario
agent_count
solved
solution_SOC
runtime_ms
CT_expanded
CT_generated
LL_expanded
peak_memory
```

## Determinism test

Run representative instances at least several times.

With deterministic tie-breaking you should obtain identical:

\[
\text{solution SOC},
\]

\[
CT_{\text{expanded}},
\]

\[
CT_{\text{generated}}.
\]

Runtime may change slightly.

If CT expansions change between runs, fix nondeterminism before collecting data.

Potential causes include:

- unordered-map iteration;
- random tie breaking;
- unstable priority-queue ties;
- uninitialized values;
- parallel low-level planner behavior.

Every CT node should have a deterministic unique insertion ID used as the final priority-queue tie breaker.

---

# 5. Phase 1 — serializable CT-node snapshots

Because the oracle operates only on the subtree rooted at \(N\), you do **not** need to serialize the entire global OPEN list.

Create:

```cpp
struct OracleSnapshot {
    string map_id;
    string scenario_id;
    int agent_count;

    vector<Constraint> constraints;
    vector<Path> paths;

    int g;
    int h;
    int f;
    int depth;

    vector<Conflict> conflicts;

    uint64_t state_hash;
};
```

You may also store:

```text
root conflict graph statistics
unconstrained individual path sum S0
MVC value
agent path lengths
```

if expensive to recompute.

Do **not** serialize raw pointers, MDD pointers, priority queue iterators or memory addresses.

Caches should either:

1. be reconstructable; or
2. be explicitly serialized in a platform-independent structure.

For the first implementation, recomputation is safer.

---

# 6. Snapshot correctness test

For randomly chosen nodes:

1. pause normal ICBS-h at \(N\);
2. compute all state statistics;
3. serialize snapshot;
4. destroy original state;
5. reconstruct snapshot;
6. recompute statistics.

Verify:

\[
g_{\text{before}}=g_{\text{after}}
\]

\[
h_{\text{before}}=h_{\text{after}}
\]

\[
Conf_{\text{before}}=Conf_{\text{after}}
\]

and identical paths/constraints.

Create a deterministic hash over:

\[
\{
constraints,\ paths,\ g,\ depth
\}
\]

and verify reconstruction produces the same hash.

Until this passes 100%, do not start bulk data generation.

---

# 7. Phase 2 — implement the local subtree runner

Core API:

```cpp
RolloutResult evaluate_candidate(
    const OracleSnapshot& state,
    const Conflict& forced_conflict,
    const RolloutConfig& config
);
```

Inside:

```text
restore N

force conflict c

generate child L
generate child R

local_OPEN = {L,R}
expanded = 0
generated = number of valid children

while local_OPEN not empty:

    X = pop using EXACT same
        ICBS-h CT-node ordering

    expanded++

    if X is conflict-free:
        record solution
        stop normal rollout

    otherwise:
        choose subsequent conflict
        using BASELINE ICBS-h

        run normal splitting/bypass logic

        update h/pathmax/etc.

        insert resulting nodes

    if expanded == checkpoint:
        capture checkpoint statistics

    if expanded == max_budget:
        mark rollout CENSORED
        stop
```

The **first** conflict is forced.

Every later conflict during the rollout is chosen by your baseline ICBS-h policy.

Therefore the oracle estimates:

\[
Q^{\pi_{\text{ICBS-h}}}(N,c).
\]

---

# 8. Expansion counting convention

Freeze this convention now.

If the forced split directly produces a conflict-free child, it is not considered solved at expansion \(0\).

It must be popped/processed from OPEN.

Therefore:

```text
split:
expansion count = 0

pop conflict-free child:
expansion count = 1
```

This eliminates ambiguities later when comparing datasets generated by different machines.

---

# 9. Phase 3 — multi-horizon checkpoints

Every rollout should support:

\[
B=
\{0,8,16,32,64,128,256\}
\]

during the pilot.

Here:

\[
B=0
\]

is essentially your existing immediate-child oracle.

You therefore automatically obtain a clean comparison between Model B and deeper lookahead.

Crucially, do not run seven separate searches.

Run once to the maximum needed budget:

\[
256
\]

and capture snapshots along the way.

Thus:

```text
one candidate rollout

0 -------- 8 -------- 16 -------- 32 -------- 64 -------- 128 -------- 256
|           |           |           |           |            |             |
save        save        save        save        save         save          save
```

A run capped at 256 already contains the 8-, 16-, 32-, 64- and 128-expansion results.

---

# 10. Checkpoint statistics

At every checkpoint \(b\), save:

```text
candidate_id
checkpoint_b

solved
solution_expansion_if_found

expanded
generated
LL_expanded

OPEN_size
min_f
number_of_nodes_at_min_f

min_h_at_min_f
mean_h_at_min_f

min_conflicts_at_min_f
mean_conflicts_at_min_f

min_depth_at_min_f
mean_depth_at_min_f

min_MVC_at_min_f
mean_MVC_at_min_f
```

Where feasible also save:

```text
number conflicting pairs
number agents in conflicts

conflict graph:
    nodes
    edges
    average degree
    maximum degree
```

The checkpoint should represent the state **after exactly \(b\) expansions**.

---

# 11. Consistency test for checkpoints

This is mandatory.

Compare:

### Run A

Run directly with

\[
B_{\max}=16.
\]

### Run B

Run with

\[
B_{\max}=128
\]

and inspect the stored \(B=16\) checkpoint.

Everything at 16 must match.

If not, checkpointing is modifying solver behavior or nondeterminism exists.

---

# 12. Oracle 1 — multi-horizon signal \(O_{\text{MH}}\)

For candidate \(c\) at checkpoint \(b\):

\[
LB_b(c)
=
\min_{M\in OPEN_b(c)}f(M).
\]

This generalizes the one-step lower-bound idea.

Also define:

\[
W_b(c)=|OPEN_b(c)|
\]

and

\[
Conf_b(c)
=
\min_{\substack{M\in OPEN_b(c)\\f(M)=LB_b(c)}}
|\mathrm{Conf}(M)|.
\]

A basic checkpoint ranking tuple is:

\[
H_b(c)=
\left(
-LB_b(c),
Conf_b(c),
W_b(c)
\right)
\]

where lexicographically lower is better because a larger lower bound is represented as \(-LB_b\).

However, **do not immediately trust this as a teacher**.

Your Model B result has already shown that increasing immediate lower bounds can be globally misaligned.

Instead use multi-horizon **stability**.

For pair \((c_i,c_j)\), obtain preferences at:

\[
8,16,32,64,128.
\]

Example:

```text
8      -> ci
16     -> ci
32     -> ci
64     -> ci
128    -> ci
```

This is strong evidence.

But:

```text
8      -> ci
16     -> cj
32     -> ci
64     -> cj
128    -> ci
```

is unstable.

Define:

\[
Stability_{ij}
=
\frac{
\text{number of informative checkpoints agreeing with majority}
}{
\text{number of informative checkpoints}
}.
\]

A provisional strong threshold can be:

\[
Stability_{ij}\ge0.8
\]

plus agreement at the last two checkpoints.

The threshold must later be calibrated on the gold set.

---

# 13. Oracle 2 — exact/censored rollout \(O_{\text{roll}}\)

This is the strongest evidence.

If candidate \(c\) solves after:

\[
E(c)=47
\]

then:

```text
rollout_status = SOLVED
expansions_to_solution = 47
```

If the rollout reaches 128 without solving:

```text
rollout_status = CENSORED
expansion_lower_bound = 128
```

Never save:

```text
expansions_to_solution = 128
```

for a censored candidate.

Mathematically the information is:

\[
E(c)>128.
\]

This distinction is essential.

---

# 14. Common-budget comparison

Adaptive deepening means two candidates may eventually have different evaluated horizons.

Suppose:

\[
c_i
\]

solves at 170,

but

\[
c_j
\]

has only been evaluated to 128.

You cannot conclude:

\[
c_i\succ c_j.
\]

The relevant pairwise budget is:

\[
B_{ij}
=
\min(B_i,B_j).
\]

Only compare exact/censored rollout evidence at the **largest common budget**.

If:

\[
E_i=83
\]

and

\[
c_j
\]

is known unresolved through 128,

then:

\[
c_i\succ c_j.
\]

That is valid because both have evidence through expansion 83.

---

# 15. Practical equality margin

Do not train the model that:

\[
101\text{ expansions}
\]

is definitely superior to

\[
103.
\]

Define a small indifference margin.

Initial proposal:

\[
\epsilon(E_i,E_j)
=
\max
\left(
5,\;
0.05\min(E_i,E_j)
\right).
\]

If

\[
|E_i-E_j|\le\epsilon,
\]

label:

\[
c_i\sim c_j.
\]

Tune this on development data.

The purpose is to avoid turning search noise and tiny differences into strong supervision.

---

# 16. Phase 4 — train the auxiliary node oracle \(D_\phi\)

The AAMAS paper learns to rank CT nodes according to their distance to a desirable solution leaf, rather than relying only on conflict counts.

Its labels use coarse distance buckets:

\[
<10,\quad10\!-\!30,\quad30\!-\!60,\quad\ge60
\]

instead of requiring exact ordering between nearly equivalent nodes.

We should adapt this idea into an **offline-only CT-node proximity model**:

\[
D_\phi(M)
\approx
\text{distance from CT node }M
\text{ to an optimal solution leaf}.
\]

Lower is better.

---

# 17. Dedicated solution-harvesting mode

Do not try to train \(D_\phi\) solely from the censored conflict rollouts.

Create a separate data-generation mode.

Run ICBS-h on easy/medium training instances where solutions can be obtained reasonably cheaply.

When the first optimal solution is found with cost:

\[
C^*,
\]

record the solution leaf.

Optionally continue searching only while:

\[
f(M)\le C^*
\]

to discover additional optimal solution leaves.

Stop when either:

\[
T_{\text{opt}}=3
\]

solution leaves have been found,

or an extra harvesting budget is exhausted.

Do **not** change the normal production solver.

This is strictly an offline instrumentation mode.

---

# 18. Computing distance-to-solution labels

For each harvested CT tree \(T\):

Initialize:

\[
d(S)=0
\]

for every observed optimal solution leaf \(S\).

Process nodes in reverse depth:

\[
d(N)=
1+
\min_{C\in children(N)}d(C)
\]

if at least one child has finite distance.

Thus:

\[
d(N)
=
\min_{S\in Solutions(N)}
\operatorname{treeDistance}(N,S).
\]

This mirrors the core labeling idea in the AAMAS paper.

---

# 19. Do not incorrectly label truncated nodes as infinity

This is an important improvement over a naïve adaptation.

Suppose no solution was observed below \(N\), but its subtree was stopped because of the harvesting cap.

Then:

\[
d(N)
\]

is **unknown**, not necessarily infinite.

Use three states:

```text
FINITE
PROVEN_NO_OPTIMAL_SOLUTION
CENSORED_UNKNOWN
```

`CENSORED_UNKNOWN` nodes should not be treated as strong negative labels.

A node can be considered a strong negative only when its relevant subtree has been exhausted or bounded away from the optimal solution cost.

---

# 20. \(D_\phi\) target buckets

Start with the paper's thresholds:

\[
y_D(N)=
\begin{cases}
0,&d<10\\
1,&10\le d<30\\
2,&30\le d<60\\
3,&d\ge60
\end{cases}
\]

and unknown for censored nodes.

These thresholds are a **paper-faithful baseline**, not a final assumption.

The AAMAS authors use coarse classes specifically to avoid wasting learning capacity on nodes that are nearly equally good.

Later compare them against empirical/quantile buckets.

---

# 21. \(D_\phi\) base features

The paper uses nine atomic features based on:

- conflict count;
- conflicting agent pairs;
- agents participating in conflicts;
- CT-node cost and lower-bound relationships;
- CT depth;

and then adds pairwise products to obtain 54 features.

For optimal ICBS-h, create an analogous nine-feature baseline:

\[
f_1=|\mathrm{Conf}(N)|
\]

\[
f_2=\#\text{conflicting agent pairs}
\]

\[
f_3=\#\text{agents participating in conflicts}
\]

\[
f_4=g(N)
\]

\[
f_5=\frac{g(N)}{LB_{\text{local}}+\epsilon}
\]

\[
f_6=g(N)-LB_{\text{local}}
\]

\[
f_7=g(N)-S_0
\]

\[
f_8=\frac{g(N)}{S_0+\epsilon}
\]

\[
f_9=\text{local CT depth}.
\]

Here \(S_0\) is a fixed instance/root reference such as the sum of individually shortest-path costs.

Then add:

\[
f_if_j,\qquad i\le j.
\]

Nine atomic features plus their degree-two interactions give:

\[
9+\frac{9(10)}2=54
\]

features, directly paralleling the paper's feature construction.

---

# 22. Extended \(D_\phi\) features

Only after the base version works, test an ICBS-h-specific extension:

```text
h / MVC value

number cardinal conflicts
number semi-cardinal conflicts
number non-cardinal conflicts

conflict-graph edge count
mean degree
max degree

number constraints
number affected agents

minimum conflict timestep
mean conflict timestep

normalized CT depth
delta-f from forced root
```

Do an ablation:

\[
D_\phi^{54}
\]

versus

\[
D_\phi^{extended}.
\]

Do not silently mix feature changes with oracle changes.

---

# 23. \(D_\phi\) model

First implementation:

\[
\boxed{\text{linear pairwise ranker}}
\]

because:

- it closely follows the uploaded paper;
- inference inside C++ is trivial;
- it is very fast;
- it is easy to debug;
- its weights can be stored as a plain vector.

The paper itself learns a linear ranking function using pairwise losses.

Export:

```text
mean[feature]
std[feature]
weights[feature]
```

so C++ evaluation is simply:

\[
z_j=\frac{x_j-\mu_j}{\sigma_j}
\]

\[
D_\phi(N)=w^\top z.
\]

Lower score should mean closer to solution.

A nonlinear version can be tested later.

---

# 24. Training pairs for \(D_\phi\)

Never randomly split CT nodes into train/test.

Split by:

\[
\boxed{\text{MAPF instance}}
\]

otherwise nodes from the same tree leak heavily between partitions.

For a tree, create pairwise labels only where the target ordering is meaningful.

If:

\[
y_D(N_i)<y_D(N_j),
\]

then:

\[
N_i\succ N_j.
\]

Ignore equal-bucket pairs.

Prefer pairs that:

- co-occurred on a frontier; or
- at minimum are non-ancestor nodes.

The AAMAS paper also avoids ranking ancestor-descendant pairs because such nodes would not compete simultaneously in the focal list.

---

# 25. Evaluate \(D_\phi\) independently

Before using it inside conflict labels, report:

\[
\text{pairwise ranking accuracy}
\]

\[
\text{bucket accuracy}
\]

\[
\text{Spearman/Kendall correlation with }d
\]

and most importantly:

\[
\text{Top-K frontier success}.
\]

For a frontier containing an actual ancestor of a solution:

> Does \(D_\phi\) put a solution-leading node in its top 1, top 3 or top 5?

That metric is directly relevant to its eventual oracle use.

---

# 26. Oracle 3 — applying \(D_\phi\) to a censored candidate

Suppose conflict \(c\) remains unsolved after checkpoint \(B\).

Its local OPEN is:

\[
OPEN_B(c).
\]

Because ICBS-h is still an optimal best-first search, don't score arbitrary high-\(f\) frontier nodes equally.

First define the currently relevant frontier:

\[
F_B(c)
=
\left\{
M\in OPEN_B(c):
f(M)=\min_{X\in OPEN_B(c)}f(X)
\right\}.
\]

These are the nodes closest to becoming relevant under the high-level search ordering.

Compute:

\[
D_\phi(M)
\]

for each \(M\in F_B(c)\).

---

# 27. Robust frontier aggregation

Do not use only:

\[
\min_M D_\phi(M).
\]

One erroneous prediction could dominate.

Sort:

\[
D_{(1)}
\le
D_{(2)}
\le\cdots
\]

and use:

\[
K=\min(5,|F_B|).
\]

Then:

\[
D_B(c)
=
\frac1K
\sum_{r=1}^{K}D_{(r)}.
\]

Lower is better.

Also record:

\[
D_{best}
\]

\[
D_{top3}
\]

\[
D_{top5}
\]

\[
\operatorname{std}(D_{\text{topK}})
\]

for later ablations.

---

# 28. Phase 5 — hierarchical fusion of the three oracle signals

This is the most important part.

Do **not** initially define:

\[
Score
=
\alpha O_{\text{roll}}
+\beta O_{\text{dist}}
+\gamma O_{\text{MH}}.
\]

The three quantities have different semantics and reliability.

Use hierarchical evidence.

For candidate pair:

\[
(c_i,c_j)
\]

follow these rules.

---

## Level A — exact rollout

At their largest common evaluated budget:

### Both solved

If:

\[
E_i+\epsilon<E_j,
\]

label:

\[
c_i\succ c_j
\]

with:

```text
source = EXACT_BOTH_SOLVED
confidence = 1.00
```

### One solved, one censored

If:

\[
E_i\le B_{ij}
\]

and \(c_j\) remained unresolved through \(B_{ij}\):

\[
c_i\succ c_j.
\]

Use:

```text
source = EXACT_VS_CENSORED
confidence ~= 0.9-1.0
```

### Approximately equal

If within the equality margin:

\[
c_i\sim c_j.
\]

No directional pair is generated.

---

# 29. Level B — \(D_\phi\) plus multi-horizon agreement

If both remain censored:

calculate:

\[
\Delta_D
=
D_B(c_i)-D_B(c_j).
\]

Use a threshold:

\[
\delta_D
\]

estimated from validation data.

If:

\[
|\Delta_D|<\delta_D,
\]

\(D_\phi\) abstains.

Otherwise it prefers one candidate.

Now inspect multi-horizon evidence.

### Case 1: both agree

Example:

\[
D_\phi:c_i\succ c_j
\]

and

\[
O_{\text{MH}}:c_i\succ c_j.
\]

Generate:

```text
source = DPHI_PLUS_MULTIHORIZON
confidence = HIGH-MEDIUM
```

Provisional weight:

\[
0.7-0.8.
\]

### Case 2: \(D_\phi\) decisive, multi-horizon abstains

Generate:

```text
source = DPHI_ONLY
confidence = MEDIUM
```

Provisional weight:

\[
0.5-0.7.
\]

### Case 3: disagreement

\[
D_\phi:c_i\succ c_j
\]

but

\[
O_{\text{MH}}:c_j\succ c_i.
\]

Do **not** label.

Request deeper rollout.

### Case 4: \(D_\phi\) abstains, multi-horizon strongly stable

Initially:

```text
source = MULTIHORIZON_ONLY
confidence = LOW
```

Only allow these labels if the gold pilot proves that such predictions are useful.

Otherwise mark UNKNOWN.

---

# 30. Level C — abstain

If evidence remains contradictory after the maximum allowed budget:

\[
\boxed{UNKNOWN}
\]

Do not generate a pair.

This is a feature, not a failure.

A partially labeled dataset containing high-quality comparisons is preferable to a fully labeled dataset polluted by arbitrary rankings.

---

# 31. Phase 6 — adaptive deepening

Do not evaluate every candidate to 256.

Configure:

\[
B_1=32
\]

\[
B_2=64
\]

\[
B_3=128
\]

\[
B_4=256.
\]

Initially all oracle-queried candidates run to \(B_1\).

After each checkpoint:

1. run the fusion rules;
2. identify clearly dominated candidates;
3. freeze candidates for which useful comparisons are already obtained;
4. deepen only unresolved/high-value candidates.

Conceptually:

```text
all queried candidates
        |
        v
      B=32
        |
    resolve easy
   comparisons
        |
        v
unresolved survivors
        |
        v
      B=64
        |
        v
    unresolved
        |
        v
      B=128
        |
        v
top/ambiguous candidates
        |
        v
      B=256
```

The final values of 32/64/128 are not sacred.

The implementation must make these YAML-configurable.

---

# 32. Candidate explosion now matters

Because all conflicts are eligible:

\[
|C(N)|
\]

may be much larger than six.

Do not confuse:

\[
\boxed{\text{candidate eligibility}}
\]

with

\[
\boxed{\text{expensive oracle-query budget}}.
\]

At deployment all conflicts are eligible.

During offline training, it is perfectly reasonable to query only a carefully selected subset with the expensive oracle.

Define:

```text
all_conflicts = C(N)

oracle_query_set = Q(N) subset of C(N)
```

---

# 33. Oracle-query policy

If:

\[
|C(N)|\le 8,
\]

evaluate all conflicts.

If:

\[
|C(N)|>8,
\]

initially query approximately eight candidates.

Call this parameter:

```text
max_oracle_queries_per_state
```

not `max_candidates`, because it does not restrict inference.

Round-0 selection should include diversity:

```text
1 baseline ICBS-h choice

1-2 conflicts from traditional high-priority classes

1 early-timestep conflict

1 highly repeated agent-pair conflict

2 approximately uniform/random conflicts

remaining slot(s):
feature-diverse conflict(s)
```

Do not select only “good-looking” candidates.

Otherwise you introduce severe selection bias.

---

# 34. Later DAgger rounds should use active sampling

Once a student model exists, query:

- current model's top-ranked conflict;
- second/third-ranked competitors;
- high-uncertainty candidates;
- some uniformly random exploration candidates;
- baseline ICBS-h choice.

This gives both:

\[
\text{exploitation}
\]

and

\[
\text{exploration}.
\]

It is especially important now that all conflicts are eligible.

---

# 35. Store why a candidate was queried

Each candidate record needs:

```text
query_reason
```

possible values:

```text
ALL_CONFLICTS_SMALL_STATE
BASELINE_CHOICE
MODEL_TOP
MODEL_UNCERTAIN
RANDOM_EXPLORATION
DIVERSITY_SAMPLE
TRADITIONAL_PRIORITY_SAMPLE
```

This lets you later verify that the training set is not dominated by one sampling mechanism.

---

# 36. Phase 7 — pilot experiment for choosing the horizon

Do **not** begin massive production generation yet.

First run a controlled horizon study.

Use approximately 100–200 decision states distributed across:

- random maps;
- warehouse maps;
- game maps;
- easy/medium/hard instances.

The AAMAS work evaluates several structurally different MovingAI maps, including random, warehouse, maze, game and city maps.

Good initial maps are:

```text
random-32-32-20
warehouse-10-20-10-2-1
den520d
```

Then add maze/city for final generalization experiments.

---

# 37. Pilot budget

For pilot candidate rollouts collect:

\[
0,\;8,\;16,\;32,\;64,\;128,\;256.
\]

For a smaller gold subset:

- continue to 512;
- or continue to complete solution where computationally feasible.

Do not call a 512-censored result “gold.”

Gold ranking requires sufficiently resolved candidates.

---

# 38. Gold subset

Select perhaps 20–50 states where:

- candidate count is manageable;
- baseline search is nontrivial;
- complete counterfactual rollouts are feasible.

For each selected state evaluate all relevant candidates to completion if possible.

Then you obtain:

\[
Q_\infty(N,c).
\]

This is your reference teacher.

---

# 39. Horizon analysis

For:

\[
B\in\{1,8,16,32,64,128,256\}
\]

measure against \(Q_\infty\):

\[
\text{pairwise accuracy}
\]

\[
\text{Top-1 agreement}
\]

\[
\text{Kendall }\tau
\]

\[
\text{pair coverage}
\]

\[
\text{mean oracle computation}.
\]

Coverage is:

\[
Coverage(B)=
\frac{
\#\text{candidate pairs receiving confident labels}
}{
\#\text{candidate pairs considered}
}.
\]

This is extremely important because your current problem is fundamentally a **coverage problem**.

---

# 40. Select the production base horizon scientifically

Choose the smallest \(B\) near the quality plateau.

For example define a rule:

Choose \(B^*\) such that:

1. its Top-1 agreement is within approximately 2–3 percentage points of the best larger tested horizon;
2. pair coverage is acceptable;
3. moving to the next horizon produces small marginal accuracy gain relative to compute growth.

Then you can legitimately write:

> “The production oracle uses a base horizon of \(B^*\) because deeper expansion produced little additional ranking information.”

This is much stronger than selecting 128 or 256 by intuition.

---

# 41. Phase 8 — production data schema

Use separate normalized tables.

## `states.parquet`

```text
state_id
instance_id
map_id
scenario_id
agent_count

ct_depth
g
h
f

num_conflicts
num_conflicting_pairs
num_conflicting_agents

state_hash

policy_that_generated_state
dagger_round
```

---

## `candidates.parquet`

```text
candidate_id
state_id

agent_i
agent_j

conflict_type
vertex/edge data
time_step

cheap feature vector

query_selected
query_reason
```

---

## `rollout_checkpoints.parquet`

One row per:

\[
(candidate,checkpoint).
\]

```text
candidate_id
budget

solved
solution_expansion

CT_expanded
CT_generated
LL_expanded

OPEN_size
min_f
min_h
min_conflicts
mean_conflicts

dphi_best
dphi_top3
dphi_top5
```

---

## `pair_labels.parquet`

```text
state_id

candidate_i
candidate_j

label:
    +1 = i preferred
    -1 = j preferred
     0 = practical tie
     null = unknown

confidence_weight

label_source

common_budget

exact_E_i
exact_E_j

dphi_i
dphi_j
dphi_margin

mh_preference
mh_stability

max_budget_i
max_budget_j

dagger_round
```

Do not throw away raw oracle information after generating final labels.

You will need it for ablations.

---

# 42. Phase 9 — parallelization architecture

Your 12-core laptops are sufficient for meaningful data generation if candidate rollouts are parallelized correctly.

The natural unit of parallel work is:

\[
\boxed{(state\ snapshot,\ candidate\ conflict)}.
\]

These jobs are almost completely independent.

---

# 43. Two-stage generation

Do not repeatedly rerun the main solver for every candidate.

### Stage A — state extraction

Run baseline/current policy once.

Save selected `OracleSnapshot`s.

### Stage B — oracle evaluation

Create independent candidate jobs from those snapshots.

Example:

```text
snapshot N127

candidate c1 -> job 127_01
candidate c2 -> job 127_02
candidate c3 -> job 127_03
...
```

Now all candidate rollouts can run simultaneously.

---

# 44. Worker executable

Create a command-line worker:

```text
icbsh_oracle_worker
```

Input:

```text
snapshot_path
candidate_id
config_path
output_path
```

The worker:

1. loads map/scenario;
2. restores snapshot;
3. verifies hash;
4. forces candidate;
5. runs local rollout;
6. records checkpoints;
7. optionally evaluates \(D_\phi\);
8. writes one result;
9. exits cleanly.

This design works locally, on Kaggle and on cloud machines.

---

# 45. Laptop scheduling

Use process-level parallelism.

Do not use nested parallelism initially.

For a 12-core laptop:

```text
worker processes = configurable
```

Start experimentally around:

\[
8-10
\]

workers rather than automatically using all 12.

Set internal numerical libraries/solver threading to one thread:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Otherwise:

```text
10 processes × multiple internal threads
```

will oversubscribe the CPU badly.

Benchmark:

\[
4,\;6,\;8,\;10
\]

workers and choose the throughput optimum.

Memory can become the limiting factor before CPU.

---

# 46. Persistent workers

Starting a C++ process for every single candidate may introduce overhead.

After correctness is established, use persistent workers:

```text
Python scheduler
       |
       +-- worker process 1
       +-- worker process 2
       ...
       +-- worker process 10
```

Each worker repeatedly reads jobs from a queue.

Static map structures can remain loaded.

This substantially reduces repeated initialization cost.

---

# 47. Do not let workers write to one database

Avoid many processes writing simultaneously to one SQLite database or one JSON file.

Instead:

```text
worker_00/
    part_000001.jsonl
    part_000002.jsonl

worker_01/
    part_000001.jsonl
...
```

After completion:

```text
merge_oracle_results.py
```

merges, validates and converts to Parquet.

Use atomic output:

```text
job.tmp
```

then rename to:

```text
job.done.json
```

only after complete serialization.

That makes interrupted runs resumable.

---

# 48. Deterministic job IDs

Define:

\[
job\_id=
hash(
state\_hash,
candidate\_id,
oracle\_config\_hash
).
\]

The configuration hash should include:

```text
checkpoint schedule
max budget
solver version
D_phi version
tie-breaking mode
```

If the output for that job ID already exists and passes validation, skip it.

This gives you reliable resume behavior.

---

# 49. Cloud sharding

For multiple laptops/cloud sessions:

\[
shard(job)=hash(job\_id)\bmod N_{\text{shards}}.
\]

Example:

```text
machine A -> shard 0
machine B -> shard 1
Kaggle    -> shard 2
cloud     -> shard 3
```

Every machine receives the same manifest but processes only its shard.

Merging is safe because job IDs are unique.

Never rely on shared mutable global state.

---

# 50. Phase 10 — production dataset generation

After the horizon pilot selects \(B^*\), generate a much larger state dataset.

A reasonable initial structure is:

```text
3 map families

× 3 difficulty levels:
    easy
    medium
    hard

× roughly 20 training instances
```

which gives roughly:

\[
180
\]

base training instances.

Do not define difficulty only by agent count globally.

For each map choose agent counts based on baseline ICBS-h behavior.

For example:

```text
easy:
baseline solves comfortably

medium:
meaningful CT search but high success

hard:
near the regime where baseline begins struggling
```

The AAMAS paper similarly uses curriculum learning across increasing numbers of agents because learned policies did not automatically generalize to much larger agent counts.

---

# 51. Sample states across the whole search

Do not simply take the first 20 nodes from every instance.

That will heavily bias the data toward root-like states.

Use search-progress strata.

For instance:

```text
0-20% of run
20-40%
40-60%
60-80%
80-100%
```

or stratify by CT depth and \(f\)-layer.

Aim initially for approximately:

\[
10-20
\]

sampled states per useful run.

The exact number should be controlled to avoid enormous correlation within one instance.

---

# 52. Split by instance, not state

Never place:

```text
state 1 from instance A -> training
state 2 from instance A -> validation
```

Use:

```text
entire instance A -> one split only.
```

Ideally also test:

### Same-map generalization

new scenarios on known maps.

### Cross-map generalization

entire map family withheld.

These answer different research questions.

---

# 53. Phase 11 — train the final conflict ranker

The final model is different from \(D_\phi\).

Call it:

\[
R_\theta(N,c).
\]

\(D_\phi\) exists only to help generate labels.

The production model takes:

\[
\phi(N,c)
\]

and returns one scalar:

\[
s_\theta(N,c).
\]

Then:

\[
c^*
=
\arg\max_{c\in C(N)}
s_\theta(N,c).
\]

---

# 54. Production features must be cheap

Do not repeat Model B's possible overhead problem.

The final ranker should not require:

- constructing both candidate children;
- low-level replanning every candidate;
- computing rollout information;
- running \(D_\phi\).

Candidate features should come directly from the current node.

Useful groups:

### Conflict-local

```text
vertex vs edge
conflict timestep
agent path lengths
remaining goal distances
number of conflicts between this pair
corridor membership
local obstacle degree
local congestion
```

### State

```text
g
h
f
MVC

total conflicts
conflicting pairs
conflicting agents

conflict graph:
    edge count
    mean degree
    max degree

CT depth
```

### Relative

For feature \(x\):

\[
x_{\text{relative}}(c)
=
x(c)
-
\frac1{|C|}
\sum_{c'}x(c').
\]

Also useful:

\[
\frac{x(c)-\mu_C}{\sigma_C+\epsilon}.
\]

Conflict quality is inherently relative to the alternatives available at the same state.

---

# 55. First student model should remain simple

To isolate whether the **new labels** are responsible for improvement:

### Experiment A

Use essentially the same student architecture as Model B.

Change only:

\[
\text{teacher labels}.
\]

If performance improves, you have strong evidence that supervision was the main issue.

Only afterwards test better models.

---

# 56. Weighted pairwise RankNet formulation

Because your final dataset is partially ordered, I recommend a true pairwise scoring loss.

For labelled pair:

\[
c_i\succ c_j
\]

with confidence weight \(w_{ij}\):

\[
L_{ij}
=
w_{ij}
\log
\left(
1+
e^{-(s_i-s_j)}
\right).
\]

where:

\[
s_i=R_\theta(N,c_i).
\]

This naturally handles:

- incomplete labels;
- different confidence levels;
- variable number of candidates;
- ties by simply omitting nearly equal pairs.

A linear \(R_\theta\) gives a very interpretable baseline.

A small MLP can later use exactly the same loss.

---

# 57. Suggested confidence baseline

Initial values only:

\[
w=
\begin{cases}
1.00,&\text{both exact rollout}\\
0.90,&\text{exact versus censored}\\
0.75,&D_\phi+\text{MH agree}\\
0.60,&D_\phi\text{ decisive only}\\
0.35,&\text{MH only}\\
0,&\text{unknown}
\end{cases}
\]

Do not present these as scientifically established.

Tune/ablate them.

---

# 58. Phase 12 — DAgger

The uploaded AAMAS work repeatedly collects training states produced by earlier versions of its learned ranking function rather than training only on baseline-generated states.

That is highly relevant because your earlier Model B likely suffered from distribution shift.

Start with:

\[
\pi_0=\text{ICBS-h}.
\]

Generate:

\[
D_0.
\]

Train:

\[
\pi_1.
\]

Then run \(\pi_1\) on training instances and collect states it actually visits.

Query the expensive composite oracle there.

\[
D_1
=
D_0\cup D_{\pi_1}.
\]

Train:

\[
\pi_2.
\]

Repeat:

\[
\pi_2\rightarrow D_2\rightarrow\pi_3.
\]

I would initially use:

\[
\boxed{2-3\text{ DAgger rounds}}
\]

rather than immediately reproducing the paper's larger number of iterations, because your oracle is significantly more expensive. The AAMAS experiments used iterative DAgger/curriculum training and reported ten DAgger iterations in their setup.

---

# 59. Active sampling in DAgger becomes especially important

At a state with 40 conflicts you cannot oracle-evaluate all 40 repeatedly.

In DAgger rounds evaluate primarily:

```text
student rank #1
student rank #2
student rank #3

baseline ICBS choice

1-2 uncertainty candidates

1-2 random exploration candidates
```

This concentrates expensive labels exactly where the current policy can make consequential mistakes.

---

# 60. Phase 13 — curriculum

Train progressively:

\[
easy\rightarrow medium\rightarrow hard.
\]

Do not discard easier data.

Aggregate it.

For example:

\[
D^{medium}
=
D^{easy}\cup D_{new}^{medium}.
\]

Then:

\[
D^{hard}
=
D^{medium}\cup D_{new}^{hard}.
\]

This is useful for both:

\[
D_\phi
\]

and the final conflict ranker.

It also addresses the central problem:

> Hard instances are precisely where full labels are difficult to obtain.

Use easier solved instances to teach structural notions of solution proximity, then use that teacher to assist hard censored instances.

---

# 61. Phase 14 — preserve correctness while choosing across tiers

Because the learned ranker is no longer constrained to traditional cardinal/semi/non-cardinal priority tiers, implement this carefully.

At an expanded CT node:

```text
detect all conflicts

compute cheap ranker features

score every conflict

choose argmax score

if operations such as bypass require
cardinality information:
    classify the SELECTED conflict

apply the existing correct ICBS logic

split/bypass

continue normal best-first CT search
```

Do not modify:

\[
OPEN\text{ ordering}
\]

or admissible \(h\).

Conflict prioritization in ICBS was introduced as an efficiency mechanism because bad choices can create large same-cost subtrees; basic CBS correctness comes from sound conflict splitting and high-level best-first search.

Nevertheless, verify empirically that your exact implementation preserves correctness because modern ICBS-h code often contains interacting optimizations.

---

# 62. Correctness regression suite

Use small MAPF instances for which you can extensively test.

Compare:

```text
baseline ICBS-h
learned conflict selector
random conflict selector
```

All should return identical optimal SOC whenever solved.

Create hundreds/thousands of small randomized cases.

Check:

\[
SOC_{\text{baseline}}
=
SOC_{\text{learned}}
\]

and verify collision-free paths.

If any differ:

\[
\boxed{\text{stop the performance experiment immediately}}
\]

and investigate bypass/symmetry/splitting interactions.

---

# 63. Phase 15 — oracle validation before ML

This is the strongest stop/go gate.

Before training the final student, evaluate the **oracle itself**.

Compare:

\[
O_1
\]

your existing one-step teacher,

\[
O_{\text{MH}},
\]

\[
O_{\text{roll}},
\]

\[
O_{\text{dist}},
\]

and:

\[
O_{\text{combined}}.
\]

Against deep/full counterfactual rankings.

Report:

| Oracle | Pair accuracy | Top-1 | Kendall \(\tau\) | Coverage |
|---|---:|---:|---:|---:|
| one-step | | | | |
| multi-horizon | | | | |
| capped rollout | | | | |
| \(D_\phi\) | | | | |
| combined | | | | |

This table may become one of the most important results of the project.

---

# 64. Oracle policy test

Also use each expensive oracle directly for conflict selection on a small development set.

Ignore oracle-computation runtime.

Measure only:

\[
CT_{\text{expanded}}.
\]

The central question is:

\[
\boxed{
\text{Does choosing conflicts according to the new oracle generate a smaller CT?}
}
\]

If the composite teacher itself cannot improve search:

\[
\textbf{do not train a student yet}.
\]

Fix the teacher first.

This prevents repeating Model B.

---

# 65. Phase 16 — final solver evaluation

Final comparison should include:

```text
ICBS-h baseline
old residual/tier rule
Model B
new oracle-imitation linear model
new model + DAgger
optional nonlinear model
```

Main metrics:

\[
\text{instances solved}
\]

\[
\text{wall-clock runtime}
\]

\[
CT_{\text{expanded}}
\]

\[
CT_{\text{generated}}
\]

\[
LL_{\text{expanded}}
\]

\[
\text{PAR10 or capped-runtime score}
\]

\[
\text{ML feature/inference overhead}.
\]

Do not report runtime only on instances solved by both without clearly marking that subset.

Also report success rate under a fixed time limit.

---

# 66. Essential ablations

You should aim for the following experiments.

### Oracle depth

\[
1,\ 8,\ 16,\ 32,\ 64,\ 128,\ 256.
\]

### Oracle components

\[
O_{\text{roll only}}
\]

\[
O_{\text{roll}}+O_{\text{MH}}
\]

\[
O_{\text{roll}}+O_{\text{dist}}
\]

\[
O_{\text{roll}}+O_{\text{dist}}+O_{\text{MH}}.
\]

### \(D_\phi\)

```text
no D_phi
paper-style 54 features
extended ICBS-h features
```

### Label handling

```text
forced complete ranking
vs
abstention/partial labels
```

### Confidence

```text
equal pair weights
vs
confidence weighting
```

### Distribution shift

```text
one-shot training
vs
DAgger
```

### Candidate scope

Later, if useful:

```text
traditional ICBS tier only
vs
all conflicts
```

That last comparison is particularly important now that you've decided not to restrict the learned policy.

---

# 67. Research questions the final experiments should answer

### RQ1

Does long-horizon search information predict actual conflict quality better than the one-step child oracle?

### RQ2

At what rollout horizon does useful information saturate?

### RQ3

How much label coverage is lost under a fixed rollout cap?

### RQ4

Can \(D_\phi\) recover meaningful rankings for rollout-censored conflicts?

### RQ5

Does multi-horizon stability improve the reliability of \(D_\phi\)-based labels?

### RQ6

Is abstaining on ambiguous pairs better than forcing complete labels?

### RQ7

Can a cheap conflict ranker imitate the composite oracle sufficiently well?

### RQ8

Does that imitation translate into fewer CT expansions?

### RQ9

Does fewer CT expansions actually translate into lower runtime after model overhead?

### RQ10

Does DAgger reduce the gap between offline ranking accuracy and online solver performance?

---

# 68. Proposed implementation order

Do the project in this exact dependency order:

```text
1. Freeze deterministic ICBS-h baseline

2. Add CT/state metrics

3. Implement OracleSnapshot

4. Validate snapshot reconstruction

5. Implement forced-conflict local subtree runner

6. Verify local runner against direct solver behavior

7. Add checkpoint recording
   0/8/16/32/64/128/256

8. Add exact/censored rollout recording

9. Build solution-harvesting mode

10. Generate first D_phi dataset

11. Train paper-style D_phi

12. Integrate D_phi inference into oracle worker

13. Implement fusion + abstention rules

14. Implement adaptive deepening

15. Implement oracle job manifest

16. Implement parallel worker pool

17. Run horizon/gold pilot

18. Select production horizon B*

19. Validate composite oracle against gold

20. Test composite oracle directly as conflict policy

---------------- STOP/GO GATE ----------------

21. Generate production training data

22. Train student using same model family as Model B

23. Compare labels only

24. Add confidence-weighted RankNet if useful

25. Run DAgger round 1

26. Run DAgger round 2

27. Run curriculum hard-instance generation

28. Freeze final model

29. Benchmark on unseen instances

30. Run all ablations and confidence intervals
```

The stop/go gate after step 20 is critical.

---

# 69. Suggested configuration format

Use one experiment configuration file.

```yaml
solver:
  deterministic: true
  conflict_policy_after_forced_choice: baseline_icbsh

candidate_scope:
  inference: all_conflicts
  oracle_query_limit: 8

oracle:
  checkpoints: [0, 8, 16, 32, 64, 128, 256]

  adaptive:
    enabled: true
    initial_budget: 32
    next_budgets: [64, 128, 256]

  practical_tie:
    absolute_expansions: 5
    relative_fraction: 0.05

dphi:
  enabled: true
  model_path: models/dphi_linear.json
  top_k_frontier: 5

  distance_buckets:
    - 10
    - 30
    - 60

fusion:
  exact_both: 1.00
  exact_vs_censored: 0.90
  dphi_plus_mh: 0.75
  dphi_only: 0.60
  mh_only: 0.35

parallel:
  workers: 10
  internal_solver_threads: 1

random:
  seed: 20261102
```

Every generated result should save the hash of this configuration.

---

# 70. Logging requirements

Every run should generate enough information that you can reproduce failures later.

At minimum:

```text
git commit hash

compiler version
build type

machine identifier
OS

map/scenario
agent count

random seed

oracle config hash
D_phi model hash
student model hash

start timestamp
end timestamp

job ID
state hash
candidate ID

exit status
```

Do not rely on filenames alone.

---

# 71. Failure handling

Worker statuses should include:

```text
SUCCESS_SOLVED
SUCCESS_CENSORED
NO_VALID_CHILD
LOW_LEVEL_NO_PATH
INVALID_SNAPSHOT
TIMEOUT
OUT_OF_MEMORY
EXCEPTION
```

Do not silently turn errors into censored samples.

A solver crash is **not**:

\[
E(c)>256.
\]

It is missing data.

---

# 72. Performance instrumentation

For every oracle rollout separate:

\[
T_{\text{restore}}
\]

\[
T_{\text{split}}
\]

\[
T_{\text{LL}}
\]

\[
T_{\text{heuristic}}
\]

\[
T_{\text{Dphi}}
\]

\[
T_{\text{total}}.
\]

This helps determine whether the bottleneck is:

- low-level search;
- MVC computation;
- copying state;
- D_phi inference;
- disk I/O.

It will also tell you where parallelization actually helps.

---

# 73. What I expect the final production oracle to look like

The likely final system—not guaranteed, but what the pilot should test—is roughly:

\[
\boxed{
B_{\text{base}}\approx32\text{ or }64
}
\]

with checkpoints:

\[
8,16,32,64
\]

and selective deepening:

\[
128
\]

then:

\[
256.
\]

Candidates that solve receive exact labels.

Censored candidates are compared using:

\[
D_\phi
\]

plus multi-horizon stability.

Disagreement triggers deeper search.

Persistent disagreement becomes:

\[
UNKNOWN.
\]

That is much more compute-efficient than:

\[
6\times1500
\]

or full rollout for every conflict.

But the pilot—not intuition—must determine whether 32, 64 or 128 is the correct production base horizon.

---

# 74. What is genuinely novel/research-worthy here

The original learned-conflict work proposes an expensive oracle and learns a fast conflict-selection policy from it.

The uploaded AAMAS node-selection work instead learns to prefer CT nodes closer to solution leaves, uses coarse distance labels, and iteratively collects states with DAgger/curriculum learning. 

Your proposed method connects these ideas differently:

\[
\boxed{
\text{capped counterfactual conflict rollouts}
}
\]

+

\[
\boxed{
\text{multi-horizon search-progress observations}
}
\]

+

\[
\boxed{
\text{learned CT-node distance-to-solution model}
}
\]

+

\[
\boxed{
\text{confidence-aware partial conflict labels}
}
\]

+

\[
\boxed{
\text{adaptive computation only for ambiguous candidates}.
}
\]

The important contribution is therefore not:

> “We looked 128 nodes instead of one node.”

It is:

> **We design a scalable oracle for conflict ranking when exact counterfactual labels are censored by computational limits.**

That framing is significantly stronger.

---

# 75. The single most important rule for this implementation

At every stage maintain the distinction:

\[
\boxed{
\text{unknown}\neq\text{bad}.
}
\]

If a candidate does not solve within 64/128/256 expansions, that does not tell you its exact quality.

That is precisely why:

- \(D_\phi\);
- multi-horizon evidence;
- adaptive deepening;
- confidence;
- abstention;

exist.

If the implementation preserves this principle, your new dataset should be fundamentally better than one where every capped rollout is converted into an artificial fixed cost.

---

# 76. Minimum viable research milestone

Before generating thousands of training examples, the first milestone should produce exactly these four outputs:

### A. Horizon plot

\[
B
\quad\text{vs}\quad
\text{agreement with full/deep rollout}.
\]

### B. Coverage plot

\[
B
\quad\text{vs}\quad
\text{fraction of conflict pairs confidently labelled}.
\]

### C. Oracle-ablation table

```text
one-step
rollout only
D_phi only
multi-horizon only
combined
```

against deep/full labels.

### D. Direct-oracle solver comparison

```text
baseline ICBS-h
vs
combined oracle conflict selection
```

measured in:

\[
CT_{\text{expanded}}.
\]

If the combined oracle does **not** significantly reduce CT expansions at this stage, stop and redesign the oracle.

If it does, then you have finally established that there exists a better conflict policy worth distilling into ML.

Only then should large-scale training begin.
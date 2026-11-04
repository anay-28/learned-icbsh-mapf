# MS8-v1: full-search multi-work-signal study

## Result

The predefined primary runtime-improvement claim is **not supported**.
Confirmed+aux solved 36/48 runs, the same as residual, but its capped-runtime
ratio was 1.010 with a stratified scenario-family bootstrap 95% interval
[1.003, 1.018]. The complete interval is above 1, indicating a small slowdown
in this experiment. Confirmed-only was indistinguishable from residual:
ratio 1.000 [0.991, 1.007], with 36/48 solved.

| Policy | Solved | Runtime ratio vs residual | 95% interval |
|---|---:|---:|---|
| plain | 30/48 | 1.566 | [1.129, 2.367] |
| residual | 36/48 | 1.000 | [1.000, 1.000] |
| pool_random | 30/48 | 1.307 | [1.050, 1.673] |
| model_b | 32/48 | 1.251 | [1.051, 1.536] |
| confirmed | 36/48 | 1.000 | [0.991, 1.007] |
| confirmed_aux | 36/48 | 1.010 | [1.003, 1.018] |

The auxiliary ablation, confirmed+aux versus confirmed, was 1.010
[1.003, 1.018]. Auxiliary HL supervision therefore did not improve deployment
runtime on this protected test.

## Evidence and gates

The official empty-8-8 inventory was split before solving into 4 pilot, 26
training, 8 validation and 12 test scenario files, balanced between even and
random categories. Duplicate assignment fingerprints at 32 agents were found
before the pilot, so that count was excluded. The study used 16 and 24 agents.

Collection produced 341 decision queries and 1,196 residual/challenger pairs
from 1,537 complete candidate replay jobs. Outcomes were 683 solved, 287
time-capped and 567 expansion-capped. Strict LL-work evidence labeled 284
residual wins and 158 challenger wins; 754 comparisons remained unknown.
Only 38 unresolved LL pairs had a strict auxiliary HL preference.

Both pairwise models passed the training support gate. On validation, every
enabled threshold was slower than residual. The predefined rule consequently
selected tau=1.01 for both variants, disabling overrides. This conservative
fallback is the intended result of the validation gate. Small test-time
differences between these fallback policies and residual measure execution
noise/overhead, not learned decisions.

## Interpretation

Full-search replay fixed the old local-subtree mismatch and produced substantially
more confirmed preferences than the earlier BDRO study. That did not translate
into a useful enabled policy: validation rejected every threshold. The evidence
therefore does not support the claim that adding low-weight HL preferences to
LL-work labels improves total solver runtime.

Ratios aggregate two counts and two repeats within each of 12 scenario files.
Failed solves receive the full cap. Intervals bootstrap files within even/random strata.
There were 288/288 test outputs and no missing/error runs. This is evidence
about one Python implementation on unseen scenarios of one known empty map.
It is not an unseen-map result or a state-of-the-art comparison. D_phi,
multi-horizon fusion, local-subtree checkpoint ablations and DAgger from the
earlier MSCO plan were not implemented in this stage.

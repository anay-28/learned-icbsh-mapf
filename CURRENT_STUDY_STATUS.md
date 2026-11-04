# Completed MS8 study

The MS8-v1 campaign completed successfully. The complete three-oracle MSCO design has not been
implemented or evaluated. Read FINAL_IMPLEMENTATION_PLAN.md, including its
reconciliation with the preserved EARLIER_MSCO_PLAN.md.

Implemented: full-solver single-decision replay with prefix fingerprints;
actual LL expansion counters; 25 versioned branch-symmetric features; separate
strict LL and HL evidence; pairwise logistic student and auxiliary ablation;
official scenario hashes and disjoint splits; isolated resumable jobs;
validation threshold selection; serial frozen evaluation and family bootstrap.

Checks: 23 new tests passed across independent correctness, replay, features,
model and analysis. The supported historical suite had 70 passes and two
freeze failures: intentional source additions/instrumentation, and Windows
CRLF conversion of an otherwise unchanged historical file. Historical evidence
has not been rewritten to hide those failures.

Pilot: 40 runs on four held-out pilot files. At 16 agents all 8 runs solved;
median capped runtime was 0.0335 seconds. At 24 agents 1/8 solved under the
8-second cap, median capped runtime 8 seconds. Smaller levels were trivial.
The predeclared fallback chose 16 and 24. This weak easy/hard coverage is a
limitation; results must not imply a broad difficulty distribution.

Input audit found duplicate 32-agent assignment sets across official files;
32 was excluded before the pilot. All remaining instance fingerprints are unique.
Split: 4 pilot, 26 training, 8 validation, 12 test files.

The campaign completed 136 source collection runs, 1,537 full-search candidate
replays, 208 serial validation runs and 288 serial held-out runs. Every scheduled
job produced a result and all four stages exited successfully. Both trained
models failed the validation runtime gate, so their frozen thresholds disable
overrides and preserve residual selection.

Inspect these files for the measured evidence:

- results/multisignal_v1/campaign_stages.jsonl
- results/multisignal_v1/audit.json
- results/multisignal_v1/validation_selection.json
- results/multisignal_v1/summary.json and RESULTS.md

Raw jobs/results and monitors are in collection/, replays/, validation/ and
test/. Each source job records hashes and parameters. Frozen source archives
are stored with the results. Failed jobs remain visible; do not delete them
to turn an incomplete campaign into an apparent success.

The primary runtime-improvement claim was not supported. Confirmed+aux solved
36/48 runs, equal to residual, with runtime ratio 1.010 and family-bootstrap
95% interval [1.003, 1.018]. Confirmed-only was indistinguishable from residual:
ratio 1.000, interval [0.991, 1.007].

# BDRO reproduction

Run commands from the repository root using Python 3.12. This study adds separate
modules; it does not change the historical solver, Model B, or frozen results.
The internal `cro` name denotes bounded same-state forced-action continuation,
not a formal causal counterfactual.

## Verify the delivered artifacts

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-recovery.lock.txt -r requirements-report.txt
.venv/bin/python -m unittest -v test_counterfactual_override
.venv/bin/python -m verify_downstream --final
```

On PowerShell use `.\.venv\Scripts\python.exe` in place of `.venv/bin/python`.
No GPU, service, or secret is needed for offline verification.

The unmodified historical checksum test hashes *all* `mapf/*.py`. Adding the five
new modules necessarily changes that directory membership. To run its original
36-test suite without changing its scope or disabling a check, use:

```bash
.venv/bin/python -m verify_downstream --legacy-regressions
```

This command first checks every historical file's checksum, copies only those
byte-identical files to a temporary directory, and runs the original suite and
`verify_submission` there. It exclusively creates `legacy_regressions.txt` and
`legacy_regressions.json`; the delivered logs can be inspected instead of
overwritten. The new integration tests exercise the existing solver and new
selectors together in the actual checkout.

## Run a separately budgeted campaign

Existing outputs are immutable. Use a **new output root**, and keep the source
unchanged throughout that campaign. Do not remove the delivered evidence to
reuse its directory. Each task already has an isolated checkout; no worktree is
needed.

```bash
.venv/bin/python -m cro_protocol freeze --output-root results/bdro_rerun --seed 20261101 --workers 6
.venv/bin/python -m collect_downstream --output-root results/bdro_rerun --stage q32 --workers 6 --resume
.venv/bin/python -m verify_downstream --root results/bdro_rerun
.venv/bin/python -m collect_downstream --output-root results/bdro_rerun --stage q128 --workers 6 --resume
.venv/bin/python -m analyze_downstream --root results/bdro_rerun --headroom --stage q128
```

Proceed to Q512 only if `headroom_q128.json` has `gate_passed: true`:

```bash
.venv/bin/python -m collect_downstream --output-root results/bdro_rerun --stage q512 --workers 6 --resume
.venv/bin/python -m analyze_downstream --root results/bdro_rerun --headroom --stage q512
.venv/bin/python -m verify_downstream --root results/bdro_rerun
.venv/bin/python -m train_override --root results/bdro_rerun --primary logistic --tau .75 --seed 71
```

If Q128 fails, verify and finalize that stage; do not force Q512 or model fitting.
If the final data gate fails, `train_override` records the STOP decision and does
not create a model. If model validation fails, do not run the confirmatory test.
Ridge is a diagnostic, never a replacement primary selected after seeing results.

For a five-query smoke campaign, use a separate root:

```bash
.venv/bin/python -m collect_downstream --stage q32 --limit-queries 5 --workers 1 --output-root results/bdro_smoke_new
.venv/bin/python -m verify_downstream --root results/bdro_smoke_new
```

Only the parent process appends rollout evidence. Resume keys are
`(query_id,candidate_id,budget)`; completed evidence is never silently replaced.
A higher stage reuses exact completions and escalates only unknown pairs. After
escalation, inspect earlier immutable snapshots instead of recollecting them.
An OS-released campaign lock prevents concurrent collectors on the same root.
Malformed/truncated journal rows cause an explicit failure and are not skipped.

## Fresh final evaluation, only after validation passes

Provide official MovingAI maps and random/even `.scen` files under
`data/movingai/` (subdirectories are supported). Required maps are
`random-32-32-20`, `room-32-32-4`, and `warehouse-10-20-10-2-1`.
Use the official benchmark source, <https://movingai.com/benchmarks/mapf/>.
Network access to that site may need environment settings; the local parser
does not download or invent missing scenarios.

```bash
.venv/bin/python -m prepare_movingai_holdout --root results/bdro_rerun --benchmark-root data/movingai
.venv/bin/python -m override_study freeze --root results/bdro_rerun --timeout 10 --repetitions 2 --seed 71
.venv/bin/python -m override_study run --root results/bdro_rerun
.venv/bin/python -m override_study analyze --root results/bdro_rerun --bootstrap-replicates 10000 --bootstrap-seed 20261101
.venv/bin/python -m override_figures --root results/bdro_rerun
.venv/bin/python -m verify_downstream --root results/bdro_rerun --final
```

Run final timing serially, with no other compute jobs. Each of the 192 scheduled
runs starts a fresh process with numerical-library threads set to one. The
controller enforces the end-to-end cap and retains failures. Native process RSS
is measured on completed workers; killed workers have sampled lower bounds and
may lack selector telemetry. Missing telemetry is not zero cost. Timing faults
invalidate a speedup claim instead of authorizing selective reruns.

The holdout selector scans historical results/configuration in this repository
and the sibling `../learned-icbsh-mapf` when present. If other exploratory
checkouts exist, explicitly supply all roots with repeated `--exclude-root`
arguments; include `--exclude-root .` too. Archive roots and scenario basenames
are recorded. Selection uses frozen seeded SHA-256 ordering, never outcomes.

## Evidence interpretation

- Capped work is an observed lower bound, not the nominal expansion cap.
- Equal completed work is `TIE`; unresolved comparisons remain `UNKNOWN`.
- Only strict residual/challenger labels enter training, with query weight one.
- Headroom is conditional on informative sampled training queries. Where residual
  remains censored, the recorded relative saving is a conservative lower bound.
- The saved-node continuation recomputes its root heuristic and rebuilds the
  descendant problem; it does not restore the original global OPEN frontier.
- Model files carry a payload fingerprint; adjacent metadata carries the actual
  JSON file checksum. Scaling is fitted on training pair differences only.
- Validation override precision includes accepted ties/unknowns in its denominator
  without counting them as demonstrated wins; they never become training negatives.
- The final primary estimator averages nested counts/repeats within each family
  before computing ratios and performs a map-stratified family bootstrap. A
  speedup claim also requires no worse solved coverage and valid timing.
- Figures and reports for gated-off stages are omitted, not synthesized.

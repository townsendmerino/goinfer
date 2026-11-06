# N-26: the banner's fit line after a declined resident (nobara-sb01, 2026-11-06)

`Model.FitBudgetSummary` (`decoder/prefill_budget.go`) priced the banner's `fit:` line at `m.resCtxReq` whenever no resident reported a context cap (a GPU backend that declined, or the CPU). A pin past the model's own window
was not clamped, so the line stated a KV cost for a context the model cannot reach, while the load-time guard (`decoder/fitguard.go`) prices a pin past the window at the window.

- **Red first** (`red.log`): `TestBanner_fitLineAfterADeclineIsClampedToTheModelWindow` (`internal/app/serveapp/banner_test.go`) loads the committed `glm-tiny.gguf` on a backend that declines residency (the existing `decliningBackend`), once with no pin and once with `ResidentContext` 8x the window (1,024 against a window of 128), and reads `factsOf` and the `fit:` line of `modelBanner`, the path `serve` prints.
  Before: fit ctx 1024, KV 786,432 bytes, and the line `fit: 1024-token cap needs 0.0 GB KV ...`; want ctx 128, KV 98,304.
- **Change:** `FitBudgetSummary` clamps the context to `cfg.MaxPositions` when it is positive. The resident-capped branch is untouched.
- **Green** (`green.log`): the new test and every `TestBanner_*` / `TestFactsOf_*` in `serveapp` pass; `TestFitBudgetSummary*`, `TestParityManifest*` and `TestParityKey*` in `decoder` pass (no parity refresh: the manifest is not stale). `cmd/lint`: 7 of 7 (`lint.log`).
- **`touched.py` is RED on this box for one reason that is not this change** (`touched.log`, `touched-enospc.log`): `TestDecisionCensus` (selected because `prefill_budget.go` changed) fails with `no space left on device` writing its synthetic fixtures to `/tmp`, a 32 GiB tmpfs; with `TMPDIR` on the NVMe it needs 97 GiB of scratch and ran past the default 10-minute timeout before reaching D5 (`census.log`), so the census was **not** read here. Its D5 `FitBudgetSummary` rows load each model with default options (`ResidentContext` 0), which take the unchanged branch, so no golden row is expected to move; CI runs the census and is the check.
- **Left on disk:** `~/goinfer-bench/tmp-census/` (97 GiB of that census scratch); my cleanup of it was refused by the permission classifier, so it is Francis's to remove.

**Known environment red:** `TestDecisionCensus` in `touched.py` on nobara-pc is an environment red, like the Mac's `backend/cuda` build: do not run it on nobara. It needs more than the box's 32 GiB `/tmp` (97 GiB of scratch on NVMe) and over 10 minutes; its check is CI, where the amd64 golden runs (coordinator, 2026-11-06). The scratch directory named above has gone to Francis.

## Review round 1 (2026-11-06)

- **A direct decoder unit test for the clamp:** `TestFitBudgetSummary_clampsAPinToTheModelWindow` (`decoder/prefill_budget_test.go`, `gptoss_tiny.gguf`): the unpinned summary prices the window; a pin 8x past it, and a pin at it, price the same ctx and KV bytes; a pin of half the window prices that ctx and fewer bytes. Red without the clamp (`decoder-red.log`: `a pin of 32768 past the window 4096: ctx 32768, KV 8390656 bytes; want ctx 4096, KV 1050624`), green with it (`decoder-green.log`).
- **The decoder tests the PR text cites** are in `decoder-green.log`: `TestFitBudgetSummary*`, `TestParityManifest*`, `TestParityKey*` (the manifest is fresh after merging main, so no `deps_hash` refresh), and `TestMM1*` (none matched a test in this tree, so it names no test; the census is the environment red above). `green.log` holds the `serveapp` banner tests only.

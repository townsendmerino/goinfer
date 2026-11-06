# MEM4(b) on CUDA: do the towers' host blocks need dropping after the device tower attaches? (2026-11, nobara-sb01)

Asked for by the coordinator (2026-11-06 03:51): "MEM4(b) on CUDA: drop the towers' host copy after attach, with a lazy reload on a CPU fallback (D2, owner 2026-10-21). One PR, with a red-first test and its registered gate."
**Units:** every size is MiB (1 MiB = 1,048,576 bytes) or GiB (1,024 MiB) with its unit.

## What the code already does

MEM4(b) is built and gated on the Mac (`docs/measurements/mem4b-2026-10/README.md`, TWR-G1 to G3, aikit v1.72.0 `ReleaseBlocks`). Serve calls `releaseTowerBlocks` (`internal/app/serveapp/main.go`) once a device tower is built, for any backend that is not the
CPU: Qwen2.5-VL (when `qwenTowerPlacement` names a device, which it does for CUDA), Qwen3.5/Qwen3-VL and GLM-OCR (when their accelerator exists), Gemma 4 (when its tower exists and is not swapped); the CPU fallback reloads the blocks (TWR-G2 with a fake accelerator). So on CUDA the
release **is already wired**; what has no reading is the heap on a CUDA tower, which TWR-G3 read on Metal only. This is that reading, and a rule that says whether anything is left to build.

## A development run came first, and the registration is not blind

`dry-run-unregistered/`, 03:46 PST, run to see the log's wording and whether the driver works. On Qwen3-VL-2B, both arms (release, and a binary with `releaseTowerBlocks` made a no-op) read `RssAnon` about 440 MiB after the second image turn, a difference of −7.6 MiB, and about 1.65 GiB after the first turn in both.
The bar below is TWR-G3's, carried over unchanged; it was not set from that run.

## MEM4(b)-CUDA-G3, registered 2026-11-06 (time from `date` below) before the reading

- **Instrument:** `mem4b_cuda.py` here. Two `serve` binaries built from the same tree (`683ec25d2`, main): `serve-mem4b-release` (as it ships) and `serve-mem4b-norelease` (the same, with `releaseTowerBlocks` returning at once, edited in a scratch copy of `internal/app/serveapp/main.go` and not committed). Each at `--model <dir> --vision <dir> --backend cuda --ctx 4096` on `~/models/qwen3-vl-2b-instruct` (local NVMe), the exclusive timing lock, one server per arm, release first, one run each. `/proc/<pid>/status` after load, then for each of two image turns (two different 1920 × 1080 PNGs, so the feature cache cannot answer the second) at the turn's end and 10 s later; peak sampled every 250 ms.
- **Void, enforced and planted:** the one-minute load average above 8.0 at an arm's start; a server that does not start; a log that shows no vision tower on CUDA; an image turn not answered with 200.
- **The decision (TWR-G3's, on `RssAnon`):**

| | |
|---|---|
| **PASS** | `RssAnon` 10 s after the second image turn is lower on the release arm than on the no-release arm by at least **80% of the tower blocks' float32 bytes, 1,153.2 MiB (0.8 × = 922.6 MiB)** (the Mac's own figure for this tower, `twr-g3.log`), and the replies are equal |
| **NOT NEEDED** | the difference is under 100 MiB: the host blocks are not what the process holds on CUDA, so a release is not a lever there; reported with the figures |
| **PARKED** | the difference is 100 to 922.5 MiB |
| **FAIL** | the replies differ |

- **Reported, not graded:** the first-turn peak and what the process holds after the first image turn, in both arms.
- **Stopping rule:** one run per arm. **Cost:** about 3 minutes. **Void if:** a card run outside the lock, a bar moved after a reading.

## The first registered run, and why one run per arm is not enough (2026-11-06 03:48 to 03:50 PST)

`reading/` (`readings.json`, `serve-*.log`, `../reading.log`). Load average 1.49 and 1.10, no void arm. After the second image turn plus 10 s, `RssAnon`: **release 473.8 MiB (0.46 GiB), no-release 2,015.5 MiB (1.97 GiB)**; the difference **1,541.7 MiB**, above the 922.6 MiB bar; the replies equal. By the rule as registered that is a PASS.

**It is not a result I can stand on.** The development run before it, on the same two binaries, read the no-release arm at 438.9 MiB after the second turn (difference −7.6 MiB, NOT NEEDED by the same rule). The two runs' logs are the same in content (no fallback, no device error, the same two images encoded), and the release arm agreed with itself (446.5 and 473.8 MiB). The no-release arm ended turn 2 at 433 MiB in one and 2,014 MiB in the other, from the same binary. If the host blocks (1,153.2 MiB) were live in that arm they could not be absent in the first run; so something other than the release decides the process's anonymous memory at the read, most likely when the Go runtime collects and returns the tower build's garbage. One run per arm cannot tell the two apart.

## MEM4(b)-CUDA-G3, amended 2026-11-06 (time from `date` below), before the further runs

- **What is added:** five more pairs of the same two binaries, the same instrument and flags, the release arm first in each pair, each pair its own server start, each in its own `reading-N/`. The first registered run is `reading/` (pair 1) and counts; the development run is listed and **not** counted. Six pairs in all.
- **The rule is unchanged** (the 922.6 MiB bar, the 100 MiB NOT NEEDED line, the replies equal) **and is applied to the median of each arm's `RssAnon` 10 s after the second turn over the six pairs**, with every pair's figures in the table. If the arms' figures overlap (some no-release value under some release value, or a spread above 1,000 MiB in either arm), the verdict is **INCONCLUSIVE**, not a pass: the release's effect would then not be separable from the collector's timing.
- **Void:** as above, per pair. **Cost:** about 3 minutes a pair.

## MEM4(b)-CUDA-G3 read, 2026-11-06 03:50 to 03:57 PST: INCONCLUSIVE by the amended rule

`reading/` (pair 1), `reading-2/` to `reading-6/` (pairs 2 to 6) and their `.log` files. Load average 0.52 to 2.10 at every arm's start, no void arm, the replies equal ("Color.") in all six pairs. `RssAnon` 10 s after the second image turn, MiB:

| pair | release | no-release | no-release minus release |
|---|---|---|---|
| 1 (the first registered run) | 473.8 | 2,015.5 | +1,541.7 |
| 2 | 498.5 | 424.0 | −74.5 |
| 3 | 462.8 | 537.1 | +74.3 |
| 4 | 490.9 | 524.6 | +33.7 |
| 5 | 504.3 | 452.3 | −52.0 |
| 6 | 527.8 | 517.5 | −10.3 |
| median | 494.7 | 521.1 | +26.3 |
| range | 462.8 to 527.8 | 424.0 to 2,015.5 | |

**Verdict: INCONCLUSIVE, as the amended rule says.** The medians differ by 26.3 MiB, which alone would be NOT NEEDED (under 100 MiB), but the no-release arm's spread (1,591.5 MiB) is over the 1,000 MiB line and its range overlaps the release arm's, so the rule does not let the median stand. The development run's 438.9 MiB is not counted; it agrees with five of the six.

**What the six pairs show, and what they do not.**
- Five of six no-release readings sit within 75 MiB of the release arm's. One pair (pair 1) read 2,015.5 MiB in the no-release arm and nothing else differs in that run. Whatever put 1.5 GiB in `RssAnon` there is not the tower's blocks being present in every no-release run, because the other five no-release runs ended near 520 MiB, below the 1,153.2 MiB the blocks measured on the Mac.
- `RssFile` is the same in both arms in every pair (1,807.2 against 1,807.5 MiB after turn 2): the weights are file pages in both, so a release cannot show in `RssFile` either.
- After turn 1 both arms usually hold about 1.6 GiB of `RssAnon` (1,616 to 1,695 MiB in 9 of 12 arm-runs); it is back near 500 MiB after turn 2, or already after turn 1 in two runs (pair 2 no-release 390.8, pair 3 release-arm turn 1 455.9 against its 462.8 at the end). That is the same in both arms, so it is the tower build's transient garbage waiting for the runtime to return it (the MEM6 finding's shape), not the blocks. Whether a `debug.FreeOSMemory` after a build would remove it was not measured here and is a separate question from MEM4(b).
- Not shown: that the release does nothing for this model. The instrument reads `RssAnon` once, 10 s after a turn, and the runtime's return of freed pages decides that reading in a way one pair cannot separate from the release.

**State of the work.** The release is already in `main` for CUDA: `releaseTowerBlocks` is called after the device tower is built for Qwen2.5-VL, Qwen3.5/Qwen3-VL, GLM-OCR and Gemma 4, and `internal/app/serveapp/tower_release_test.go` already pins, with a fake accelerator that is not Metal-specific, that the call is made once per attach and that a fallback onto reloaded blocks gives the same tower (`TestTowerRelease_gridTowersFallBackOnReloadedBlocks`, `TestTowerRelease_startupTowers`). So there is no code to add for CUDA, and a red-first test for a call that is already made and pinned would be red against nothing. The ledger's MEM4(b) row stays done (Mac gates TWR-G1 to G3); this adds the CUDA record. No claim that the CUDA host copy is saved is made, and none that it is not.

**If this is to be settled,** it needs a reading that does not depend on collector timing: a `debug.FreeOSMemory` (or `runtime.ReadMemStats` heap-in-use) taken after the second turn in each arm, through a test hook, in more than one pair. That is a change to `serve` or a test hook and is the coordinator's call; it was not started.

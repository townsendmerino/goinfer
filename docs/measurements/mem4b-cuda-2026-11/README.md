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

# G46 re-read on 2026-11-24: Gemma 3's tensor-core GEMM prefill, AMBIGUOUS by the registered rule

<!-- citations-at: 7d4b58916 -->

A dated re-read of `docs/measurements/g46-gemma3-gemm-fidelity-2026-10-21.md` (registered and first read 2026-10-21, FAIL), on Francis's
word of 2026-11-24 to re-read it on today's main before deciding a remedy. Nothing about the registration changed: the same rule, the same
twelve cells and controls, the same bar.

## Did the port run as registered

Yes. `backend/cuda/kv3_tiled_test.go:TestG46_gemma3GEMMFidelityLadder` (ported by #1312 and never run until now) ran from a fresh
worktree of `origin/main` at `7d4b58916` through the record's own `run-g46.sh`, unmodified. The design (two prompts, seeds 4242 and 9001, x six
depths 4,096 to 12,160; one per-token reference per prompt; the levers-off controls at the shortest and longest depth) and the per-cell rule
(**PASS** cosine >= 0.992 and equal argmax; **FAIL** under 0.99 or a different argmax; **AMBIGUOUS** between) are the 2026-10-21
registration's. The script writes into the 10-21 record's `run/` directory; today's files were moved to `g46-gemma3-gemm-fidelity-2026-11-24/run/`
and the 10-21 files restored, unchanged.

## Run

2026-11-24, 16:42:59 to 16:52:30 PST (570 s, against 550 s on 10-21), under nobara's exclusive timing lock, launched only once no other
session held the lock and the 1-minute load was under 4.0. Commit `7d4b5891`, Go 1.27.2, driver **595.112.07** (10-21: 595.91.07). At the
start: load 3.20, `MemAvailable` 62.7 GB, 7,467 MiB of VRAM free (valid: under 8.0 and far above the need). Test process: peak `VmHWM` 6.47 GB
(6,469,048 kB), `VmSwap` 0. Checkpoint `~/models/gemma-3-4b-it-Q4_K_M.gguf` (local NVMe). Records: `g46-gemma3-gemm-fidelity-2026-11-24/run/`.

## Cells, today beside 2026-10-21

| depth | seed 4242, 10-21 | seed 4242, today | seed 9001, 10-21 | seed 9001, today |
|---|---|---|---|---|
| 4,096 | 0.994150 PASS | 0.994949 PASS | 0.995975 PASS | 0.994757 PASS |
| 6,144 | 0.996105 PASS | 0.994929 PASS | 0.996998 PASS | 0.995513 PASS |
| 8,192 | 0.995950 PASS | 0.996313 PASS | 0.991072 AMBIGUOUS | 0.994981 PASS |
| 10,240 | **0.987131 FAIL** | 0.996370 PASS | 0.995030, **argmax differs: FAIL** | 0.994360 PASS |
| 12,000 | 0.996317 PASS | 0.994378 PASS | 0.996074 PASS | 0.993700 PASS |
| 12,160 | 0.995662 PASS | 0.994849 PASS | 0.990061 AMBIGUOUS | **0.990694 AMBIGUOUS** |

Every argmax equal today. Every default cell launched the tensor-core GEMM (476 launches at 4,096 to 1,428 at 12,000 and 12,160; 10-21: 1,904
to 5,712) and no row-kernel group; every logit differs from the per-token path in every default cell (worst 1.68 to 3.18). **All four controls
held:** levers off, 0 of 262,144 logits differing and 0 tensor-core launches at 4,096 and 12,160 on both prompts. Per-token reference 136.1 s and
136.8 s for 12,160 rows (10-21: 135.9 and 136.2).

**Verdict, by the registered rule: AMBIGUOUS, parked.** No cell is under 0.99 and no argmax differs; one cell, seed 9001 at 12,160 tokens
(0.990694), is in the parking band, the same cell that was parked on 10-21 (0.990061). As registered, a parked cell is reported with no
conclusion and the owner decides what it means.

## What changed since 2026-10-21, as far as can be told

- **The prompts.** The test encodes its prompts from the tree's own sources (`kv0TextTokens`), so every cell today reads different text from
  the 10-21 cell beside it; the 10-21 record already showed that the GEMM's distance from the per-token path depends on the content at least as
  much as on the length. The two FAIL cells of 10-21 (10,240 on both prompts) read 0.996370 and 0.994360 today on today's text.
- **The driver** (595.91.07 to 595.112.07), whose JIT compiles the PTX. Not separable from the prompts with these readings.
- **The launch structure.** The tensor-core launch count fell by exactly 4x at every depth. `gemm_w4a8_mma.cu` is unchanged since 10-21 apart
  from its move under `backend/` (47c717c84); a 4x fall is consistent with the CUDA prefill pass's restructuring that the prefill launch golden
  guards (31318e197), which was held bit-identical, but that attribution is not checked here.

None of these is a change to the GEMM's arithmetic that this record can name, so today's PASS cells are not evidence that the 10-21 FAIL was
fixed: they read other text on another driver.

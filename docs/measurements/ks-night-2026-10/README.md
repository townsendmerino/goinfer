# Kernel survey first measurements: KS-D5, KS-D6, KS-D7, KS-H4, 2026-10-22

Items from `docs/tasks/task-kernel-survey-2026-10.md`. Each is its ledger row's first measurement, and none changes product code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. Run at night (22:58 to 23:07) on the owner's word, 2026-10-22. None is
timed, so each held the timing lock shared. The 1-minute load was 3.5 to 5.4 at the starts, left over from earlier builds.

| id | instrument | reading | verdict |
|---|---|---|---|
| KS-D5 | `TestKSD5_guFShare`: Qwen2.5-Coder-1.5B (the Metal `.giw`), one 3,900-row batched pass (the block set to the whole prompt), the scratch counter (`prefillScratchSeen`) against the planned bytes | counted 261.9 MB, planned 261.9 MB; `guF` (Mpad × 2 × 8,960 halves) 139.9 MB, **53.4%**; 121.9 MB without it. The survey's 255 → 116 MB and 52–55% hold. The 7B figure (563 → 268 MB) was not re-read: a Metal `.gguf` load of the 7B writes a same-size `.giw` and swaps on this Mac | confirmed; the fix (SwiGLU fused into the gate\|up GEMM, M) is next, behind a byte compare |
| KS-D6 | `TestKSD6H4_gridTowerScratch`: the Qwen3-VL-2B grid tower, f32, a 4K grid (32,160 patches); each scratch buffer's bytes; a second forward with the buffers called unread filled with NaN | scratch 2,622.5 MiB; called unread 918.7 MiB, **35.0%** (`qh`, `kh`, `vh` 125.6 each, `sc` 31.4, `vt` 7.9, `ob` 0.1, `f2` 502.5); `pt` 188.4 MiB; the NaN-filled forward: **0 of 32,931,840 outputs differ** | confirmed unread; the fix (allocate them only for the towers and attention forms that read them, `f2` only for a gated MLP) is next, gated on MEM2-G1's seven tower fixtures |
| KS-D7 | `TestKSD7_half2DequantExhaustive`: every f16 scale × every nibble (1,048,576 cases), the prefill GEMM's staging dequant (`half(float(n-8) * sc)`) against the half2 form, compiled as the prefill library is (fast math) | **0 cases differ**, NaN and infinite scales included; both half2 lanes agree in every case. Plant (1031 for 1032): 964,206 differ, so the check can go red | bit-identical; the speed read (time to first token at 3,900, kill under 2%) is next, by day |
| KS-H4 | the same test: Metal's allocated size net of the tower's build, right after the 4K forward and 10 s later | **2,622.5 MiB, and 2,622.5 MiB 10 s later**: nothing is released until `Close` | confirmed; the fix (release above a ceiling after the forward) is next, S |

Logs: `~/goinfer-logs/ks-d5-2026-10-22.log`, `~/goinfer-logs/ks-d6h4-2026-10-22.log`.

## KS-D7's speed read (D7-G), 2026-10-23

Gates registered before the change, on a branch that was not merged (the survey doc's "KS-D7 gates" carry them here). The build
under test: both staging sites of the prefill GEMM dequantize two nibbles per half2 operation, with `KSD7_OLD_DEQUANT` compiling
the per-weight form back. Run 05:03 on the owner's word, 2026-10-23 (started 05:09, timing lock held exclusively, the 1-minute load
1.35 at the start, 6.3 GB available).

| gate | reading | result |
|---|---|---|
| D7-G1 model identity | llama-tiny at int4, a 100-token prefill: 0 of 256 logits differ; Qwen2.5-Coder-1.5B, a 3,900-token prefill: 0 of 151,936 differ. A plant (1031 for 1032) turned the tiny cell red, and the startup self-test declined the backend | **PASS** |
| D7-G2 speed | Qwen2.5-Coder-1.5B (the Metal `.giw`), a 3,900-token `PrefillLast`, two residents in one process, A B B A over 10 pairs: per-weight 4,094–4,165 ms, half2 4,094–4,164 ms; **median new/old 0.9983, 5 of 10 pairs below 1**. Log: `~/goinfer-logs/ksd7-g2-2026-10-23.log` | **KILL** (bar: ≤ 0.98 and 9 of 10) |

The dequant is bit-identical but buys nothing measurable at this shape: the staging is not where the 3,900-token prefill spends
its time. The change was not merged, and the row closes as measured.

## KS-D6, the fix (D6-G), 2026-10-23

Gates registered before the code (survey doc, "KS-D6 gates"). The build in `gridVAccel.grow` has three parts. The unfused
attention's `qh`, `kh`, `vh`, `sc`, `vt` and `ob` are allocated only when that attention can run. `f2` is allocated only for a
gated MLP. `pt` is a view of `f1` when `f1` is wide enough. `gridScratchAllForTest` restores main's allocation.

| gate | reading | result |
|---|---|---|
| D6-G1 identity | `TestKSD6_towerOutputsUnchanged`: MEM2-G1's seven towers, three inputs each, new allocation against main's: 0 differ on every tower (gemma4-vision 4,032, gemma4-audio 4,352, embeddinggemma2 16,560, qwen3.5 7,936, glm-ocr 15,872, siglip 1,536, qwen2.5-vl 16,384 outputs). The qwen3.5 and glm-ocr grids under `oldAttn` run the unfused attention on its lazily allocated scratch, with no NaN. `TestGridVisionMetal_matchesCPU` and `TestMEM2_towerScratchFreed` pass | **PASS** |
| D6-G2 allocated size | `TestKSD6H4_gridTowerScratch`, Qwen3-VL-2B at 4K, 05:19, timing lock held shared: Metal's allocated size net of the build **2,622.5 → 1,515.4 MiB** (bar ≤ 1,520); the NaN fill still changes 0 of 32,931,840 outputs (there is nothing left to fill). The test's buffer sum (2,017.9 MiB) counts `pt` and `f1` twice, since they are now one buffer. Log: `~/goinfer-logs/ks-d6g2-2026-10-23.log` | **PASS** |

**Plant** (`pt` aliased to `x`): red. On the tiny grid fixtures `x` is narrower than the patch rows, so the staging copy panics
(slice bounds) before any output.

## KS-H4, the fix (H4-G), 2026-10-23

Gates registered before the code (survey doc, "KS-H4 gates"). The build: after a forward whose scratch is over 256 MiB, the grid
tower releases it once idle for 5 s. A forward in the meantime (in `grow`) cancels the release, and the next forward after a
release allocates again. A generation counter makes a release that fires late a no-op, and `Close` cancels it.
`gridReleaseOffForTest` keeps the scratch until `Close`.

| gate | reading | result |
|---|---|---|
| H4-G1 identity | `TestKSH4_releaseKeepsOutputs`, ceiling 0 and a 1 ms wait, under `-race`: qwen3.5 and glm-ocr grids release after every forward (scratch 0 bytes) and give 0 of 8,960 and 0 of 17,920 outputs differing from a tower that never releases, over grids small, large, middle and small again; qwen2.5-vl 0 differ | **PASS** |
| H4-G2 allocated after 10 s | `TestKSD6H4_gridTowerScratch`, Qwen3-VL-2B at 4K, 05:27, timing lock held shared: right after the forward 1,515.4 MiB, **10 s later 0.0 MiB** (bar ≤ 16). Log: `~/goinfer-logs/ks-h4g2-2026-10-23.log` | **PASS** |
| H4-G3 repeat call | `TestKSH4_repeatCallTiming`, Qwen3-VL-2B at 4K, 05:29, timing lock held exclusively (the 1-minute load 0.95 at the start, 9.2 GB available): a repeat call 49.9–50.1 s with the release on and off, **median on/off 0.9996** (bar ≤ 1.02). Read only: the first call after a release 50,092.8 ms against a warm call's 50,025.8 ms, **+67 ms (0.13%)**. Log: `~/goinfer-logs/ks-h4g3-2026-10-23.log` | **SHIP** |

**Plant** (the release keeps `cap`, so the next forward skips the allocation): H4-G1 red, a slice-bounds panic on the released
buffer.

A 4K forward of this f32 tower is about 50 s on the M1 Pro, so H4-G3 took 12.5 minutes, not the minute registered.

## KS-D5, the fused SwiGLU (D5-G), 2026-10-23: PARK

Gates registered before the code (survey doc, "KS-D5 gates"). The build is `gemm_w4f16_swiglu`: the 64 × 64 int4 GEMM with its
feature tile split into 32 gate rows and the matching 32 up rows. It rounds the tile to half in threadgroup memory as `guF` held it,
and writes `glu_act_f16(g) × u`. A dense layer whose pass takes the 64 × 64 tile uses it, and `guF` is allocated only for a pass that
keeps the two-step form. `PrefillBlockRows` now prices a row on a long pass and steps to the most rows that fit, since a short pass
may keep `guF`.

| gate | reading | result |
|---|---|---|
| D5-G1 identity | `TestKSD5_fusedSwigluMatchesTwoStep`: 12 cells (I = 96 and 8,960, M = 64, 100, 200, SiLU and GELU-tanh), 0 outputs differ; `TestKSD5_prefillFusedIdentity`: llama-tiny 0 of 256, Qwen2.5-Coder-1.5B at 3,900 rows 0 of 151,936 logits differ. Plant (the activation on the unrounded accumulators): 2,532 of 6,144 differ in the first cell | **PASS** |
| D5-G2 scratch | `TestKSD5_guFShare`'s counter, one 3,900-row pass on the 1.5B: **121.9 MB** counted (planned 121.9), from 261.9 | **PASS** (bar ≤ 123) |
| D5-G3 speed | `TestKSD5_prefillSpeed`, the 1.5B, a 3,900-token `PrefillLast`, fused against two-step on one resident, 08:52, timing lock held exclusively (the 1-minute load 3.37 at the start, 9.3 GB available): two-step 4,177–4,239 ms, fused 4,619–4,660 ms, **median 1.1040, 0 of 10 pairs below 1**. Log: `~/goinfer-logs/ksd5-g3-2026-10-23.log` | **PARK** (bar ≤ 1.01): 10.4% slower |

The fused kernel saves `guF`'s 139.9 MB on that pass and is bit-identical, but makes the prefill 10.4% slower. The likely cause, not
measured: its added 8 KB threadgroup tile, which takes the kernel from about 9.6 KB to 17.6 KB of threadgroup memory and so
cuts occupancy. The staging arrays (`sa`, `sb`) are dead once the K loop ends, so a build that places the tile in their memory would
test that. The branch is pushed and not merged; the next step is the owner's decision.

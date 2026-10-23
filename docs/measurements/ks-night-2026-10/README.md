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

Gates registered before the change (survey doc, "KS-D7 gates"). The build under test: both staging sites of the prefill GEMM
dequantize two nibbles per half2 operation, and `KSD7_OLD_DEQUANT` compiles the per-weight form back. The change was not merged.

**The first read (05:09) was an A/A, and is withdrawn.** The A/B's switch prepended `KSD7_OLD_DEQUANT` to the resident's main
library (`allKernels`). But the prefill GEMM's staging is in `prefillKernels`, which `ensurePrefill` compiles into its own library,
lazily, at the first prefill. So both arms ran the half2 staging. Its 0.9983 median, 5 of 10 pairs, measured nothing, and so did
the D7-G1 byte compare (the plant then turned red only through the startup self-test). The KILL recorded from that read stood on
no measurement.

**The re-run (08:31, on the owner's word, 2026-10-23).** The switch now goes on the prefill library's source, captured by the
resident at build time (the library compiles after the test's switch is reset). Before the run, a defect planted only in the old
branch (`#ifdef KSD7_OLD_DEQUANT`) made the startup self-test decline the old arm, so the define reaches the prefill library, and
only that arm. The timing lock was held exclusively; the 1-minute load was 2.77 at the start, with 7.5 GB available.

| gate | reading | result |
|---|---|---|
| D7-G1 model identity | llama-tiny at int4, a 100-token prefill: 0 of 256 logits differ; Qwen2.5-Coder-1.5B, a 3,900-token prefill: 0 of 151,936 differ, per-weight against half2 | **PASS** |
| D7-G2 speed | Qwen2.5-Coder-1.5B (the Metal `.giw`), a 3,900-token `PrefillLast`, two residents in one process, A B B A over 10 pairs: per-weight 4,194–4,230 ms, half2 4,150–4,187 ms; **median new/old 0.9888, 10 of 10 pairs below 1** (1.1% faster). Log: `~/goinfer-logs/ksd7-g2-rerun-2026-10-23.log` | **KILL** by the registered bar (≤ 0.98 and 9 of 10; "kill under 2%"): the gain is real and consistent but under 2% |

The half2 dequant is bit-identical and makes the 3,900-token prefill 1.1% faster on the 1.5B, every pair agreeing. That is under the
survey's 2% kill line, so the row closes as measured. A 1.1% bit-identical gain is a smaller bar than this item registered; whether
to take it is the owner's call, not the gate's.

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

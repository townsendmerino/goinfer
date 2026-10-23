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

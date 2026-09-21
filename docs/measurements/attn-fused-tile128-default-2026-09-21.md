# The 128-row `attn_fused` tile is now the default for hd128 layers without a sliding window (R5 / P24)

Pre-registration: `attn-fused-tile128-default-PREREGISTERED.md` (written before these measurements). Kernel/mechanism record: `attn-fused-tile-2026-09-21.md`. RTX 2070 SUPER, driver 595.91.07, idle box, `TestPrefillDecomp`, arms interleaved as separate processes, 3 rounds. Raw logs beside this file.

## Gates
1. **Kernel bit-identity:** 128x64 == 64x64 (and 32x64) bit for bit on **456 shapes** (window=0; hd 64/128; M 1..517 incl. 16,17,48,96,127,128,129,255,256,257; startPos 0,5,384,1024; MHA/GQA/sinks). PASS.
2. **Whole model:** full chunked `PrefillLast`, default selector vs 64x64 forced: **0 of 151,936 last-row logits differ** on the 1.5B at 1300 tokens (3 chunks, 84 launches of the 128-row kernel = 28 layers x 3) and 3900 (8 chunks, 224 launches), and on the 0.5B at 1300 (0 launches: hd64 does not use it). Mutation-checked: pointing the default at the non-identical 32x32 arm turns it red (151,936/151,936 differ).
   gemma3-1b and phi3 were NOT tested: hd 256 / 96 are not served by `attn_fused` at all (it declines to `attn_batched`), so the gate is vacuous for them.
3. `go test -short ./cuda/...`, vet, staticcheck, gofmt, `TestPrefillChunked_*`, `TestAttnFused_*` (all arms), `TestAttnBatched_bitIdentical`, `TestPrefillLastNArgmax`, `TestHiddenLastResidentParityCUDA`: PASS.
4. No new §3 fidelity gate: outputs are bit-identical to the shipped kernel (gates 1-2), whose gate already passed.

## Threshold T (registered rule)
Kernel micro-sweep (attention only, M 16..512 x startPos 0/1024/3400): **hd128 t128/t64 <= 0.99 at every cell** (0.32-0.99) -> T = 16 (already `attnFusedMinRows`). **hd64 fails the rule** (1.15x slower at M=512/startPos 3400, 1.03x at M=384/1024, 1.15x at M=16/startPos 0), so **no joint T exists and the registered clause "hd64 fails, hd128 passes -> hd128 only" applies.**
The served 0.5B (hd64) measurement agrees: attention at K=3900 **145.8 ms (64x64) -> 152.8 ms (128x64), 1.05x SLOWER**, all 3 rounds; K=512-2048 within +-6% either way. hd64 therefore stays on 64x64. Windowed layers stay on 64x64 by construction (untested for identity, not measured).

## Performance (1.5B, dense int4, hd128; default vs 64x64 forced, median of 3 rounds; spread < 0.5%)

| K | attention 64x64 -> default | whole prefill (GEMM+attn+glue) | gemv control |
|---|---|---|---|
| 512 | 12.8 -> 6.5 ms (1.98x) | 91 -> 84 ms (1.09x) | 67.5-68.3 vs 67.7-68.2 |
| 1024 | 46.1 -> 23.5 ms (1.96x) | 204 -> 181 ms (1.13x) | unchanged |
| 2048 | 209.7 -> 96.2 ms (2.18x) | 528 -> 414 ms (1.28x) | unchanged |
| 3900 | 804.1 -> 336.8 ms (**2.39x**) | 1421 -> **950 ms (1.50x)** | 527-528 vs 526-530 |

**Decision rule: SHIPS** for hd128 — 1.5B K=3900 attention 2.39x (>= 2.0x), K=512 faster, whole prefill 1.50x (>= 1.3x), gemv control within +-0.6% (<= 3%). The 0.5B (hd64) condition FAILED (1.05x slower), so the default is hd128-only, as registered.

## Not established
Peer (Ollama) TTFT was not re-measured (`scripts/bench_peer_prefill.py`); `docs/benchmarks.md` still carries the pre-regression 1.9-3.2x-behind prefill row. Other hd128 models (7B D7, Llama-class) have the same kernel and geometry but were not timed. Registers/thread of the 128 kernels and ncu occupancy were not read.

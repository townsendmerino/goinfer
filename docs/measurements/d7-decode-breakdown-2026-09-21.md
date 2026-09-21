# D7 decode breakdown, and a bit-identical +4-5% from one launch-config change

The reminder: with the flash-decode lane on, goinfer was 0.95-0.97x of Ollama on qwen2.5-7b at depth 2048-3900 (0.98x at 128). The lane had already taken
attention off the critical path, so this asks where the rest goes. Instrument: `ncu --metrics gpu__time_duration.sum` over a window of a served plain-decode
request (`d7-decode-breakdown-2026-09-21/ncu_decode.py`, `summ.py`), lane on (`GOINFER_CUDA_FLASH_DECODE=16`), greedy, RTX 2070 SUPER, driver `595.91.07`, idle box.
Per-token figures divide by the count of LM-head launches in the window (one per token; 16-18 tokens per window).

## Where a D7 token goes (GPU time per token, before the change)

| kernel | ~130-token prompt | ~4000-token prompt | note |
|---|---:|---:|---|
| `fused_rms_gu` (rmsnorm + gate/up GEMV) | 5.90 ms (44.1%) | 5.93 ms | 216 us/layer; ~316 GB/s, ~70% of the card's peak |
| `gemv_w4a8_fwd` (o-proj and down-proj) | 3.15 ms | 3.15 ms | 57.5 us avg over two launches per layer |
| attention | 0.48 ms (`attn_batched`) | 1.73 ms (`fa_partial` 1.48 + `fa_combine` 0.25) | the lane |
| `fused_rms_qkv` | 1.39 ms | 1.39 ms | 50.7 us/layer for ~8 MB of weights: ~163 GB/s, launch-latency-bound |
| `gemv_w8a8_fwd` (LM head) | 1.30 ms | 1.29 ms | ~420 GB/s, at the roof |
| **`glu_quant`** | **0.98 ms (7.3%)** | **0.98 ms** | **35.7 us/layer for a vector operation** |
| total GPU | **13.39 ms** | **14.66 ms** | wall per token: 13.7 ms (72.9 tok/s) / 15.1 ms (66.3 tok/s) |

Two facts: **decode is GPU-bound at both depths** (GPU-busy is 97-98% of wall, so launch overhead is not the story), and the **non-attention work is a fixed ~12.9 ms
per token**, identical at both depths. The gap to Ollama at depth is therefore that fixed cost plus the lane's 1.73 ms, not attention alone.

## The change

`glu_quant` is ONE block (its int8 scale needs the max over the whole intermediate vector), launched at 256 threads, so its time is the per-thread serial loop over I
elements (I = 18944 on D7, each with an `expf` and a divide). Its only reduction is a MAX, which is exact and order-independent, and every value and packed byte is
per-element, so the block size changes only how the work is divided. It is now 1024 threads (`glueQuantThreads`, three launch sites in `cuda/resident.go`); the kernel
source and PTX are untouched.

- **Bit-identity, on real logits:** SHA-256 over the full logits of a 300-token prefill plus 24 decode steps, before and after, on the 1.5B, the 0.5B and gemma3-1b
  (which exercises the GELU-tanh branch): **all three identical.** A scratch test, not committed (a golden hash would break on any legitimate numerics change).
- **Permanent pin:** `TestGluQuantBlockSizeInvariant` launches the kernel at 256 and at 1024 threads on random rows, both activations, seven widths (including ones not a
  multiple of the block size): 24,196 packed words and every scale equal. Not mutation-checked (a kernel mutant needs `glue.ptx`, the audited frozen module, to be rebuilt).
- **Kernel time (ncu):** `glu_quant` 35.7 -> **13.6 us/layer**; D7 GPU time per token at ~130 tokens **13.39 -> 12.78 ms (-4.6%)**.

## Served, same session, three engines (`scripts/bench_peer.py`, goinfer new / goinfer previous build / Ollama v0.32.5; both goinfer arms with the lane on)

| model | depth | new | previous | new / previous | Ollama | **new / Ollama** |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B | 128 | 344.1 | 341.7 | 1.007 | 268.9 | 1.28 |
| 0.5B | 2048 | 305.0 | 304.0 | 1.003 | 272.5 | 1.12 |
| 0.5B | 3900 | 310.5 | 309.0 | 1.005 | 259.3 | 1.20 |
| 1.5B | 128 | 240.7 | 227.4 | **1.058** | 195.1 | 1.23 |
| 1.5B | 2048 | 217.5 | 206.2 | **1.055** | 180.0 | 1.21 |
| 1.5B | 3900 | 204.6 | 194.3 | **1.053** | 175.0 | 1.17 |
| D7 | 128 | 76.2 | 72.9 | **1.045** | 74.1 | 1.03 |
| D7 | 2048 | 71.7 | 68.8 | **1.042** | 71.0 | 1.01 |
| D7 | 3900 | 68.9 | 66.2 | **1.041** | 69.7 | **0.99** |

D7 goes from 0.95-0.98x of Ollama to **0.99-1.03x**: parity at depth, ahead at shallow. The gain scales with the intermediate width (I = 4864 on the 0.5B: +0.5%; 8960 on
the 1.5B: +5.5%; 18944 on D7: +4.2-4.5%). The change is not lane-specific: it applies to every CUDA decode. Cells started under the harness's 1.0 load cap
(recorded per-cell loadavg 0.66-0.98), one sweep, no repeat, so cross-session drift (~3.5% on this box) is not bounded; the paired new/previous ratios are the
sturdier reading.

## Remaining levers on the same profile (not touched)

`fused_rms_qkv` at ~163 GB/s (1.39 ms/token; ~0.8 ms of it recoverable if it reached ~400 GB/s), the gate/up and down GEMVs at ~70% of peak, `glu_quant` still 13.6 us/layer
(a multi-block two-pass version, or fusing it into the down GEMV's prologue, would remove most of the rest), the lane's `fa_combine` (0.25 ms/token), and the ~3% wall-vs-GPU gap.
None is measured beyond the profile above; each is a numerics-sensitive or design change, unlike this one.

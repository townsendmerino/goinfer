# Metal prefill decomposition — where the fast path's time goes (S0, 2026-09-25)

**Question.** The Metal prefill GEMM is the item "biggest gap on the Mac": TTFT 0.377× Ollama's at K=512 and 0.239×
at K=3900 (`peer-claim-2026-09-25.md` cell h, 1.5B, Ollama v0.32.5 at its defaults). R4 killed the tile-tuning
approach and the audit priced parity at "≈3.5× on the GEMM" (`audit-metal-2026-09-12.md` §6) — arithmetic from
before M-03/M-04 that assumed the GEMM was all of TTFT. Nobody had split TTFT by kernel since. This measures the
split, so the scope of any GEMM work follows from a number: a GEMM-only redesign moves TTFT by at most
1/((1−X) + X/s), X being the GEMM's share.

**Answer.** At K=512 the GEMM is **92.7% of GPU time** (96.0% of `PrefillLast` wall), and **gate/up alone is
62.6%**. Parity at K=512 needs the GEMM category **≈2.85× faster** — not 3.5×. At K=3900 the GEMM is 76.0%, and the
non-GEMM remainder (0.239 of wall) already equals the parity target (TTFT ratio 0.239), so **no GEMM speedup alone
reaches parity at K=3900**; attention (25.0% there) has to move as well. The kernel's throughput is strongly
shape-dependent: **1.26–1.42 TFLOPS on qkv and o, but 0.74–0.76 on gate/up and 0.68–1.02 on down** — the same kernel
running the widest-N and longest-K shapes at roughly half the rate of the others.

## Provenance

- **Machine:** Apple M1 Pro, 16 GB, macOS 26.6.2 (kernel 25.6.0). Idle at start (load1 1.89 after the runner's
  wait for ≤ 2.0).
- **Build:** goinfer `558c6cad`, plus the uncommitted test file this record adds (`metal/prefill_decomp_test.go`,
  the only dirty path; no production file differs). Metal backend, `-tags goinfer_testhooks`.
- **Model:** `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (internal SSD), loaded `Quant: "int4"` through its
  v14 `.metal.giw` sidecar. H=1536, I=8960, 28 layers, 12 heads, V=151936. Fused attention on (the default).
- **Run:** 2026-09-25, 13:18:52–13:24:20 local (20:18:52–20:24:20 UTC), 5 reps per measurement after one warm-up.
  `GOINFER_METAL_DECOMP=1 go test -tags goinfer_testhooks -run '^TestMetalPrefillDecomp$' -v -count=1 ./metal/`.
  Raw log: [`metal-prefill-decomp-2026-09-25/run.log`](metal-prefill-decomp-2026-09-25/run.log).

## Method

`PrefillLast` is one command buffer, so its GPU timestamps give only a total. The test gives each kernel category
its own command buffer holding that category's dispatch **for all 28 layers, each with its own weights**, using
`PrefillLast`'s exact grids and buffers, so the cache footprint matches production (repeating one layer's dispatch
would keep its weights resident in cache and flatter the kernel). Reps interleave the full replay and every category,
so drift reaches all of them alike. GPU time is `GPUEndTime − GPUStartTime` of each command buffer. Three checks:

1. **A full in-order replay of every category reproduces `PrefillLast`'s logits bit for bit** — all 151,936 values,
   at both K. So the replica issues production's dispatches, and its split describes production.
2. **The categories sum to 99.9% (K=512) and 101.4% (K=3900) of the full replay** — splitting into per-category
   command buffers neither hides nor adds meaningful time.
3. **`PrefillLast`'s wall time matches the full replay's GPU time** (K=3900: 17,618 vs 17,653 ms) — host-side work
   (f16 conversion, buffer allocation, encoding) is indistinguishable from zero. At K=512 the wall median sits 58 ms
   *below* the GPU median, which is noise, not negative overhead: `PrefillLast`'s five wall readings spread 17%
   (1414–1688 ms) while the replica's GPU readings spread 0.5%.

## Result (medians of 5)

**K=512** (Mpad 512)

| category | GPU ms | share of full | spread | achieved |
|---|---:|---:|---:|---|
| GEMM gate/up (N=17920, K=1536) | 1042.38 | **62.6%** | 0.7% | 0.76 TFLOPS |
| GEMM down (+residual) (N=1536, K=8960) | 386.00 | 23.2% | 1.6% | 1.02 TFLOPS |
| attention (fused) | 109.86 | 6.6% | 1.7% | |
| GEMM qkv (N=2048, K=1536) | 66.75 | 4.0% | 5.0% | 1.35 TFLOPS |
| GEMM o (+residual) (N=1536, K=1536) | 49.20 | 3.0% | 1.1% | 1.38 TFLOPS |
| swiglu | 5.81 | 0.3% | 0.2% | |
| LM head (last row) | 1.55 | 0.1% | 5.4% | |
| rope q+k, rmsnorm ×2, kv store | 3.19 | 0.2% | | |
| **sum / full replay (GPU) / `PrefillLast` wall** | **1664.72 / 1665.73 / 1607.97** | | 0.5% / 17.0% | |

GEMM 1544.3 ms: **92.7% of GPU, 96.0% of wall.**

**K=3900** (Mpad 3904)

| category | GPU ms | share of full | spread | achieved |
|---|---:|---:|---:|---|
| GEMM gate/up | 8109.02 | **45.9%** | **51.9%** | 0.74 TFLOPS |
| attention (fused) | 4419.04 | 25.0% | 0.5% | |
| GEMM down (+residual) | 4397.04 | 24.9% | **54.7%** | 0.68 TFLOPS |
| GEMM qkv | 544.32 | 3.1% | 3.0% | 1.26 TFLOPS |
| GEMM o (+residual) | 362.61 | 2.1% | 0.8% | 1.42 TFLOPS |
| swiglu | 43.17 | 0.2% | 4.5% | |
| rope, rmsnorm ×2, kv store, LM head | 18.69 | 0.1% | | |
| **sum / full replay (GPU) / `PrefillLast` wall** | **17893.89 / 17652.60 / 17618.09** | | 2.5% / 0.6% | |

GEMM 13413.0 ms: **76.0% of GPU, 76.1% of wall.**

**The two >50% spreads at K=3900 are recorded, not explained.** gate/up's and down's category runs varied by half
their median across the five reps while the full replay held to 2.5% and the categories still summed to 101.4% of
it, so the medians are consistent with production — but the test prints only medians, so which reps were slow (and
whether it was thermal, the ~8 s single-category command buffers, or something else) is not recoverable from this
run. S1 prints per-rep values.

## What the split implies

The TTFT target is Ollama's: r = 0.377 at K=512, 0.239 at K=3900 (TTFT ≈ `PrefillLast` wall on this path —
1580 vs 1608 ms median at K=512). A GEMM speedup of s gives (1−X) + X/s of today's TTFT, X = GEMM share of wall:

| K | X (GEMM share of wall) | non-GEMM remainder | parity target | GEMM speedup needed |
|---|---:|---:|---:|---|
| 512 | 0.960 | 0.040 | 0.377 | **≈2.85×** |
| 3900 | 0.761 | 0.239 | 0.239 | **unreachable by the GEMM alone** — the remainder is already the target |

- **At K=512 the item is the GEMM, and mostly one GEMM.** gate/up is 62.6% on its own; qkv and o, together 7%, run
  the same kernel at nearly twice gate/up's rate. So the shape-dependence is itself a finding: whatever limits gate/up
  (N=17,920 is 560 column tiles, each re-reading the A panel from device memory — the current kernel stages nothing
  between simdgroups) does not limit the small-N shapes as much. A 2.85× category speedup puts gate/up near 2.2 TFLOPS.
- **At K=3900 the item is the GEMM plus attention.** Removing the GEMM entirely would leave 0.239 of today's TTFT —
  exactly parity, not better. Any claim of K=3900 parity from GEMM work alone would be wrong by construction.
- **The 3.5× in the audit is superseded by ≈2.85× at K=512**, because the GEMM's share is 96%, not 100%, and because
  the parity target moved to Ollama's default configuration (`peer-claim-2026-09-25.md`).
- **What this does not say:** what the M1 Pro can actually reach at these shapes. That is S1 — a plain f16
  simdgroup-matrix GEMM (no dequant) as the attainable ceiling, llama.cpp's own `mul_mm` at the same shapes, and the
  per-rep values the >50% spreads need.

## Scope of this record

One checkpoint (1.5B), one machine, two prompt lengths, the plain dense layer shape (no MoE, sandwich norm or
QK-norm). The 7B's shapes (H=3584, I=18944) are not measured here; the 7B GEMM share is not implied by the 1.5B's.

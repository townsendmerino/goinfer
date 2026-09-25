# Metal prefill decomposition — where the fast path's time goes (S0, 2026-09-25)

> **Amended the same day (see *Addendum: isolated time is not in-sequence cost*, below).** Two follow-up runs
> showed that a kernel category timed in its own command buffer does not give its cost inside production: the MLP
> GEMMs cost **~1.9× more in the sequence than alone**, at both K. The shares below still describe production (the
> first run's medians happened to sit in the slow mode, which is why they summed to ~100%), but the TFLOPS column is
> the sustained rate, and run 4 shows that is the right baseline: gate/up's ~1.45 TFLOPS alone is a burst right after
> the GPU has been idle, and any sustained GPU work — production included — runs it at ~0.75.

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

## Addendum: isolated time is not in-sequence cost (2026-09-25, runs 2 and 3)

**Run 2** (K=3900 only, every rep printed, page-in / swap-in deltas and `pmset -g therm` around each command
buffer; 13:29–13:34 local; raw: [`run2-k3900-perrep.log`](metal-prefill-decomp-2026-09-25/run2-k3900-perrep.log)).
Every category was stable (spreads ≤ 1.2%), but the isolated MLP GEMMs ran **~2× faster than in run 1**: gate/up
4151 ms (run 1 median 8109), down 2131 (4397); attention 4169 (4419). The categories then summed to only **64.6%** of
the full replay, which held at 17.5 s, like `PrefillLast`. So each isolated GEMM runs in one of two modes about 2×
apart, and run 1's 52–55% spreads were a mix of both. **Neither hypothesis for the spread is supported**: no swap-ins,
page-ins small and steady (~850 per 4 s system-wide, i.e. not the GPU re-reading evicted weights), no thermal warning
recorded, 80–82% memory free.

**Run 3** (both K, leave-one-out; 13:39–13:52 local; raw: [`run3-leave-one-out.log`](metal-prefill-decomp-2026-09-25/run3-leave-one-out.log)).
A category's in-sequence cost = the full replay minus the same replay with that category removed, 5 paired reps.
Kernel timing does not depend on the values read, so removing a category changes only the cache and memory state the
rest see. The in-sequence costs are additive — **92.4% (K=512) and 100.2% (K=3900) of the full replay** — where the
isolated ones summed to 65%.

| K | category | alone (ms) | in sequence (ms) | share of full | in-seq ÷ alone | TFLOPS alone → in sequence |
|---|---|---:|---:|---:|---:|---|
| 512 | GEMM gate/up | 551.9 | **1050.5** | 62.9% | **1.90×** | 1.43 → 0.75 |
| 512 | GEMM down | 292.3 | 378.6 | 22.7% | 1.30× | 1.35 → 1.04 |
| 512 | attention | 108.2 | 114.8 | 6.9% | 1.06× | |
| 3900 | GEMM gate/up | 4134.9 | **8032.5** | 46.1% | **1.94×** | 1.46 → 0.75 |
| 3900 | GEMM down | 2126.4 | **4037.3** | 23.2% | **1.90×** | 1.41 → 0.75 |
| 3900 | attention | 4181.2 | 4428.1 | 25.4% | 1.06× | |
| 3900 | GEMM qkv | 443.7 | 578.2 | 3.3% | 1.30× | 1.55 → 1.19 |
| 3900 | GEMM o | 333.7 | 394.2 | 2.3% | 1.18× | 1.55 → 1.31 |

Paired deltas are tight (gate/up at K=3900: 7987–8197 ms). What this changes:

- **The MLP GEMMs run at about half the rate inside production that they reach alone** — 0.75 TFLOPS against
  ~1.45, at both K. The ratio sits near 2× and the isolated runs flip between the same two levels, which reads more
  like a discrete state than graded cache pressure; attention, the memory-bound kernel, barely moves (1.06×). The
  mechanism is not identified here.
- **The GEMM-share conclusions above stand** (they are in-sequence shares). Run 4 below shows the in-sequence rate is
  the GPU's steady state, not an interaction to recover.
- **Any kernel benchmark timed alone (S1 as scoped) would read the fast mode** and overstate what the same kernel
  delivers in production: a microbenchmark that reproduces the shape but not the conditions around it exonerates the
  kernel by leaving out the thing that slows it. S1 has to measure in sequence.

## Run 4: the 2× follows the GPU's recent workload (2026-09-25, stopped after round 1 of 3)

K=512, two processes (`GOINFER_METAL_ALIAS` unset, then `=0`; separate processes because knobs are snapshotted per
model at `decoder.Load`), each with leave-one-out and a **prior-state** phase: GEMM gate/up timed alone, interleaved
per rep, after four different things. 13:58:54–14:02:14 local; the remaining two rounds waited on a busy box
(load1 2.3–3.4 from UI processes, not the benchmark) and were stopped by owner decision at 14:10. Raw:
[`run4-alias-ab-prior-state-partial.log`](metal-prefill-decomp-2026-09-25/run4-alias-ab-prior-state-partial.log).

| what ran just before gate/up | default arm (ms, median of 5) | `ALIAS=0` arm | spreads |
|---|---:|---:|---|
| 2 s of idle | **552.5** | **558.7** | 1.7–1.8% |
| a full replay | 1042.6 | 1039.3 | ≤ 1.0% |
| an attention-only command buffer | 1040.8 | 1038.8 | ≤ 2.1% |
| itself, back-to-back (second of two) | 1039.0 | 1040.2 | ≤ 0.7% |

- **The fast mode is a burst after idle; the slow mode is the steady state.** gate/up run back-to-back on identical
  data with warm caches is already slow, so the 2× is not cache state or anything a preceding kernel leaves behind.
  It follows whether the GPU has just been working: after idle this kernel reaches ~1.45 TFLOPS, under any sustained
  GPU work ~0.75. Production prefill is sustained work, so **~0.75 TFLOPS is the kernel's real rate there, and the
  in-sequence costs above are the right baseline** — there is no interaction to recover. *Corrected by S1a the same day* ([`metal-gemm-ceiling-2026-09-25.md`](metal-gemm-ceiling-2026-09-25.md)):
  this is **not a GPU-wide clock or power state** — Apple's MPS GEMM runs the same shape at the same rate after idle
  and under sustained load (242 vs 232 ms). The burst is specific to how goinfer's kernel uses the GPU; its mechanism
  is not identified. No thermal warning was recorded.
- **Aliasing is not the cause, on one round each and with one caveat.** gate/up in sequence 1046.8 (default) vs
  1043.4 ms (`ALIAS=0`); `PrefillLast` wall 1682 vs 1653 ms. The test did not print whether the aliaser was active in
  each process (it should have been off in the second — the knob is read at `decoder.Load` from the environment the
  runner exported), so the A/B is unverified; the test now prints it.
- **For S1:** time any kernel under sustained load, with no idle gap before the timed run, or it may read a burst.
  S1a found MPS has no burst, so the rule protects against goinfer-kernel behaviour, and a replacement kernel should be
  shown both sustained and after idle.

## Scope of this record

One checkpoint (1.5B), one machine, two prompt lengths, the plain dense layer shape (no MoE, sandwich norm or
QK-norm). The 7B's shapes (H=3584, I=18944) are not measured here; the 7B GEMM share is not implied by the 1.5B's.

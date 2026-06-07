# goinfer performance campaign (decode tok/s)

> Opportunistic, profile-driven. The goal is to recover the tok/s goinfer is
> leaving on the table **without** changing what it competes on (portability /
> embeddability, not raw throughput) and **without** touching numerics.

## Premise — there's headroom, by the roofline

Single-token decode (batch=1, the chat case) is **memory-bandwidth bound**: each
token streams the whole int8 weight set once. The ceiling is roughly
`memory_bandwidth ÷ model_bytes`.

Measured (M-series CPU, prequant int8int8):

| model | int8 bytes/token | tok/s | effective bandwidth |
|---|---|---|---|
| Qwen2.5-Coder-0.5B | ~0.5 GB | ~44 | **~22 GB/s** |
| Qwen2.5-Coder-1.5B | ~1.5 GB | ~20 | **~30 GB/s** |

An M-series chip delivers ~68 GB/s (base) to ~200–400 GB/s (Pro/Max). goinfer is
using a fraction of that, so it's **compute/overhead-bound, not bandwidth-bound —
headroom exists.** A hand-tuned kernel (llama.cpp-class) gets several× closer to
the roofline.

## Strategic frame (gate the effort)

- Speed is **not** the pitch. Do the cheap wins; only invest in the hard kernel
  work if Phase 0 says the payoff is there. ~44/~20 tok/s is already interactive —
  **do not block the launch on this.**
- **Non-negotiable: HF logit parity stays green on every change.** Perf must not
  move numerics — that's goinfer's contract. Any change that fails the parity
  gate is reverted, full stop.
- Pure Go, no cgo, in the default build. SIMD lives in `aikit/linalg` asm; that's
  fair game. No cgo BLAS.

## Where the work lives (cross-repo)

- **Forward-loop / allocations:** `goinfer/decoder`.
- **Matmul kernels (SDOT, etc.):** `aikit/linalg` — kernel changes are an aikit
  PR + version bump, then goinfer's `go.mod` follows.

## Phases (each gated on Phase 0's findings)

### Phase 0 — profile (the easy experiment; do FIRST, no optimization)
Benchmark a decode loop, capture CPU + alloc profiles, report the breakdown:
how much time is in the matmul kernel vs `mallocgc`/GC vs goroutine
dispatch/scheduling vs attention vs activation quantization — plus `allocs/op`
and the effective GB/s. This turns "there's probably headroom" into "here's the
30% sitting in X." Decide the next
phase from the numbers.

### Phase 1 — kill per-token allocations (`goinfer/decoder`)
`runLayers` allocates several `[]float32` per layer per token
(`make([]float32, hidden)`, `append([]float32(nil), h...)`, etc.) — GC pressure
in the inner loop. Reuse scratch buffers (an arena/pool tied to the cache or
model). Cheap, low-risk; expect lower GC + smoother latency + a modest tok/s
bump. *Do only if Phase 0 shows meaningful alloc/GC cost.*

### Phase 2 — matmul kernel tuning (`aikit/linalg`)
The W8A8 SDOT kernel (`MatmulBTW8A8`): multiple accumulators, cache
blocking/tiling, prefetch, wider NEON; confirm the amd64 AVX2/FMA path keeps
pace. Biggest lever, hardest, cross-repo. *Do only if Phase 0 shows the kernel
dominates* (it should, if everything else is small).

### Phase 3 — parallelism granularity + fusion
Per-matmul `parallelCols` recurs ~7 matmuls × N layers per token — goroutine
dispatch overhead. Try a persistent worker pool / coarser (whole-layer)
parallelism, and **fuse QKV** into one matmul for cache locality. *Do only if
Phase 0 shows dispatch/scheduling overhead.*

## Success criteria

- Fixed model + prompt + seed; report tok/s before/after each phase.
- A committed decode benchmark so regressions are caught.
- Parity tests green after every change.
- Aspirational (not a promise): ~2–3× from the cheap+medium phases; the roofline
  is the hard ceiling. Record actual results here per phase.

## Findings log

### Phase 0 — profile (2026-06-05, Apple M1 Pro, 6 P + 2 E cores)

Setup: `BenchmarkDecode` (`decoder/decode_bench_test.go`) — Qwen2.5-Coder-0.5B at
`int8int8`, steady-state single-token `forward`+greedy-sample, model loaded once.

**Decode throughput (this benchmark, short context):** ~56 tok/s at the default
GOMAXPROCS=8, **`4660 allocs/op`, `4.3 MB/op`**.

**The headline: dispatch overhead dominates, not the kernel.** GOMAXPROCS sweep:

| GOMAXPROCS | tok/s |
|---|---|
| 1 | 51.3 |
| 2 | **60.3** |
| 4 | 60.3 |
| 8 | 56.0 |

Parallelism past 2 cores buys **nothing**, and 8 cores is **slower than 2** —
single-thread is within 15% of the best. The CPU profile agrees: ~**70%** of
samples are `runtime.pthread_cond_wait` (38.8%) + `pthread_cond_signal` (31.4%) —
goroutine park/wake from `aikit/linalg.parallelCols`, which forks ~7 matmuls ×
24 layers per token. At 0.5B batch=1 each matmul slice is microseconds of work,
so the per-matmul goroutine wakeups cost more than they save.

CPU buckets (10 s profile; one-time model-load dequant excluded):

| bucket | ~CPU | notes |
|---|---|---|
| goroutine dispatch (`pthread_cond_*`) | **~70%** | `parallelCols` fork/join per matmul |
| matmul kernel (`dotI8SDOT`, `MatmulBTW8A8`, `dotI8`) | ~12–15% | the actual SDOT work |
| activation quant (`QuantizeRowInt8`) | ~5–7% | W8A8 quantizes each activation |
| attention (`causalAttention`, RoPE) | ~3–4% | small at short context |

**Allocations (`-alloc_space`):** `MatmulBTW8A8.func1` 1.55 GB (39%, aikit kernel
allocs the int8 activation + result per matmul) and `gatedMLP` 0.86 GB (22%,
goinfer MLP scratch) dominate — confirming per-token, per-layer churn.

**Roofline:** ~604 MB int8 streamed/token × ~56 tok/s ≈ **~34 GB/s effective** vs
the M1 Pro's ~200 GB/s rated → **~18% of the roofline.** Compute/overhead-bound
with large headroom, as premised. We can't reach the roofline while dispatch
caps useful parallelism at ~2 cores.

### Phase 1 (goinfer) — decoder scratch reuse (2026-06-05, done)

Reuse per-stream forward buffers off the KV cache (`decodeScratch`): the residual
`h`, the norm input, the attn/MLP output, q/k/v/ctx/scores, gate/up, and the
vocab-sized logits — no per-layer allocation in the hot path. Parity green
(`TestDecodeParity` unchanged).

| metric | before | after |
|---|---|---|
| bytes/op | 4.3 MB | **2.07 MB** (−52%) |
| allocs/op | 4660 | 4395 (−6%) |
| tok/s | ~48 | ~48 (flat) |

Read: this halved the *bytes* churned (the big float buffers were the decoder's),
but the alloc *count* and tok/s are unchanged — because the remaining ~4400
allocs/op are the **aikit kernel's per-call allocations** (`MatmulBTW8A8.func1`),
and tok/s is gated by **dispatch**, not GC. So goinfer Phase 1 is a real
memory-churn / latency-jitter win that **pairs with** aikit's Phase 1 (kernel
scratch) — it is not a standalone tok/s mover. Confirms the bottleneck is in
aikit (dispatch + kernel allocs), per the spec.

### Phases 1+3 combined — aikit v0.5.0 wiring (2026-06-05, M1 Pro)

aikit v0.5.0 added `Workspace` + `MatmulBTW8A8Into` (zero-alloc activation quant)
and `MatmulBTW8A8Batch` (q/k/v or gate/up in one quantize + one parallel region,
weights read **in place** — no concat, prequant aliasing intact) + a tunable
`SetParallelThreshold`. goinfer wired: a `Workspace` per stream (on the cache),
batched q/k/v and gate/up, `Into` for o_proj/down/LM-head.

**Allocations: 4395 → 19 allocs/op (1.5 KB/op).** Decode is now zero-alloc.
**Parity: bit-identical** (`TestDecodeParity` unchanged — the batched/Into kernels
are numerically identical by construction).

Threshold × GOMAXPROCS sweep (the end-to-end arbiter the aikit caveat asked for):

| config | P=1 | P=2 | P=4 | P=8 |
|---|---|---|---|---|
| serial decode (aikit default 16.78M) | 51 | 51 | 51 | 52 |
| **parallel decode (threshold ≈0.3M)** | 51 | 65 | **68** | 66 |

**~68 tok/s vs the ~48 tok/s baseline — +42%, zero-alloc, parity-identical.**

Two numbers, kept distinct (be honest at launch):
- **Runtime decode (`BenchmarkDecode`, pure forward+sample): ~48 → ~68 tok/s.**
- **Demo end-to-end (`demo/chat`, longer gen): ~44 → ~58 tok/s.** The ~10 tok/s
  gap is streaming overhead — the channel handoff in `Generate` + the per-token
  decode/stdout — which is a *larger relative* share now that decode is faster
  (constant ~2–3 ms/token overhead vs a 14.7 ms vs old 21 ms decode step). It's
  real streaming cost; quote the demo number for the README, the benchmark for
  ARCHITECTURE/this log. (A buffered `Generate` channel — let the model compute
  the next token while the UI prints the current — should narrow it; optional.)

goinfer owns the tuning: `decoder.SetDecodeParallelThreshold` (the demo calls it
with `DefaultDecodeParallelThreshold` = 0.3 M MACs). aikit's library default
stays conservative — the crossover is hardware/model-specific (x86, core count,
1.5B), so the consumer that knows the workload sets it.

Read: Phase-0's "70% `pthread_cond`" was **idle worker threads parking**, not
critical-path cost — serial decode (≈51) only matches old GOMAXPROCS=1, while
*parallel* batched decode hits ~66–68. So the win was zero-alloc + **keeping the
matmuls parallel** (the batched dispatch parallelizes better than the old per-op:
66–68 vs the old 60). **Recommendation to aikit: lower the default
`parallelThreshold`** (decode parallelizes) — the end-to-end arbiter says serial
was the wrong default. ~0.3 M MACs is a good cut (parallelizes the batched
projections + LM head; leaves only trivially-small ops serial).

### Phase 0 RE-RUN — profile the *optimized* path (2026-06-05)

Re-profiled `BenchmarkDecode` at the shipping config (threshold 0.3 M, parallel,
zero-alloc-serial / ~1.7 k-alloc-parallel). New breakdown:

| bucket | ~CPU |
|---|---|
| parallel fork/join (`pthread_cond_signal`+`_wait`, `parallelSpawnCols`) | **~71%** |
| matmul kernel (`dotI8SDOT`/`dotI8`/`w8a8Span`) | ~15% |
| activation quant (`QuantizeRowInt8`) | ~9% |
| attention (`causalAttention`) | ~4% |

**The scaling is the problem, not the kernel.** Serial = 51 tok/s; 8-core
parallel = 68 — only **~1.3× off 6–8 cores.** Each batch=1 matmul's parallel
work is microseconds, so the per-dispatch **fork/join latency** (futex park/wake
of the workers) dominates — even after batching cut it to ~4 fork/joins/layer.
Effective ~36 GB/s = **~18% of the M1 Pro's ~200 GB/s roofline**: the headroom
the user sees is real, and it's gated by parallel-dispatch latency.

**Next lever (aikit, Phase 3b): cut the per-dispatch fork/join latency.** A
**persistent worker pool that spins briefly before parking** keeps workers hot
across the ~4 dispatches/layer × 24 layers, so back-to-back decode matmuls don't
pay a full futex wake each time. `parallelSpawnCols` currently spawns/parks per
call (the `Spawn` in the name). This is the path from 68 toward the bandwidth
roofline. Caveat: batch=1 decode is *fundamentally* fork/join-bound (~4 serial
dependencies/layer can't be merged), so the practical ceiling is below the 330
tok/s pure-bandwidth number — but ~1.3× scaling says there's a lot left.

goinfer is near its floor: ~4 matmuls/layer is near-minimal (o_proj/down/LM-head
have distinct activations — can't batch further), scratch is reused, quant is
batched. The remaining wins are in `aikit/linalg`'s pool.

### Phase 3b ARBITER — persistent pool does NOT win (2026-06-05)

aikit added an opt-in persistent worker pool (`Workspace.SetWorkers`) that spins
before parking. goinfer is the arbiter; end-to-end decode sweep (threshold 0.3 M):

| | P=4 | P=6 | P=8 |
|---|---|---|---|
| pool **off** (spawn) | **67.6** | 61.9 | 65.0 |
| pool on, workers=4/6 | 64.1 | ~62 | ~62 |

**Pool is neutral-to-slightly-slower.** Parity bit-identical, but no tok/s win —
matching the aikit microbench. The deeper finding: spawn ≈ pool, so the
Phase-0 "71% `pthread_cond`" is **not pool-fixable** — it's the *inherent* cost
of synchronizing workers for a batch=1 matmul whose parallel work is
microseconds, plus idle-worker park samples that overcount. The fork/join is a
**floor**, not an overhead a hotter pool removes.

**Decision: don't enable the pool in goinfer; recommend aikit ship v0.5.0 as the
proven Phases 1/3 and pull the experimental pool** (the aikit Claude offered
this). ~68 tok/s is the practical ceiling for pure-Go batch=1 CPU decode with
this approach — the ~330 tok/s pure-bandwidth roofline is unreachable at batch=1
(can't parallelize the tiny per-token matmuls efficiently). Net campaign result:
**~48 → ~68 tok/s runtime (+42%), ~44 → ~58 demo, zero-alloc, parity-identical.**
Further gains would need a different approach (bigger batch, speculative decode,
or a fundamentally faster single-thread kernel), i.e. diminishing returns — and
speed was never the pitch.

### Fan-out width cap (aikit v0.5.1 `SetParallelWidth`) — NO default change

Hypothesis: capping the matmul fan-out below GOMAXPROCS tightens the fork/join
join by avoiding an E-core straggler (M1 Pro 6 P + 2 E). aikit added
`SetParallelWidth` (parity bit-identical at every width — output columns are
partitioned, not the K-reduction; confirmed `TestDecodeParity` green at W∈{1,4,8}).
goinfer swept it (`BenchmarkDecode`, GOMAXPROCS default, threshold 0.3 M):

| tier | W=4 | W=8 (GOMAXPROCS) | Δ |
|---|---|---|---|
| 0.5B (alternating A/B, 4–5 rounds) | ~72.8 | ~72.7 | **~0% (wash)** |
| 1.5B (alternating A/B, 4 rounds) | ~36.6 | ~34.7 | **+5% (real, confirmed)** |

(Re-confirmed clean post-v0.5.2; decode/width is M=1 so the re-block didn't move
it. The 1.5B +5% is genuine, not noise — but see the verdict below.)

**Decision: default stays GOMAXPROCS (no win on the gate).** The gate is anchored
on the 0.5B (shipping/headline tier), and there the cap is within noise (~1%, far
under the ≳3% bar — the earlier apparent win was thermal drift). The 1.5B *does*
gain ~4.4% at W=4 (bigger matmuls + straggler), but: (a) it doesn't meet the
0.5B-anchored gate, and (b) a *fixed* `DefaultDecodeParallelWidth` would cap
high-core machines too (a 16-core x86 server forced to 4-way) — the width is
hardware-specific, as the task doc itself notes, and pure Go can't pin P vs E
cores (statistical at best). The knob stays exported in aikit for consumers that
know their hardware / run the 1.5B on M-series; goinfer ships no default width.
Allocs unchanged. (`GINFER_PAR_WIDTH` left on `BenchmarkDecode` for future sweeps.)

### Speculative decoding — PARKED on CPU (the win is bandwidth-bound, this isn't)

goinfer built greedy speculative decoding (0.5B drafts K tokens, 1.5B verifies in
one `forwardN(M=K+1)`; token-identical to plain greedy — gates green) and aikit
re-blocked the W8A8 kernel column-outer (v0.5.2) so M>1 reuses each weight across
the M rows. Both correct; both bit-identical (`TestDecodeParity`,
`TestForwardN_matchesSequential`, `TestSpeculativeGreedyParity`). Decode (M=1)
unchanged at ~73 tok/s.

**But end-to-end speculative is ~0.65–0.70× — a loss — even with the re-block,
and even at 85% acceptance.** Isolated `forwardN(M=K)` vs K plain decode steps
(1.5B, 120-token context, v0.5.2):

| M | forwardN cost vs M decodes |
|---|---|
| 4 | 0.78× |
| 8 | 0.67× |
| 16–32 | **0.51×** (asymptote) |

The re-block *works* (without it this was ~1.0× / flat), but it asymptotes at
**~0.5×, not ~1×.** The reason is the campaign's recurring lesson: **pure-Go CPU
int8 decode is ~half compute** (the SDOT), and compute is K× at M=K — only the
dispatch + weight-bandwidth (~half) amortize. Speculative's verify runs at small
M (K=4–5 → 0.78×) and adds draft cost on top, so the round costs **more** than the
(accepted+1) plain decodes it replaces. Speculative is a *bandwidth-bound /
GPU* win; on this CPU it can't clear the ≥1.3× gate (best case ~K=2 ≈ break-even).

**Decision: PARK the speculative speed claim; keep the code.** It's correct,
exact, gated, and `--draft` is wired — it will win on a bandwidth-bound backend
(a future GPU path) where compute is ~free. Don't advertise it as a CPU speedup.

**The re-block's real payoff is batched prefill — SHIPPED.** Prefill is now one
batched M=len(prompt) pass (`prefillLogits` → `forwardLayersN`, LM head on the
last position only) instead of sequential M=1. Seed token bit-identical
(`TestDecodeParity` green). Measured TTFT on the 1.5B:

| prompt len | sequential | batched | speedup |
|---|---|---|---|
| 32 | 1324 ms | 459 ms | **2.9×** |
| 80 | 2752 ms | 1276 ms | **2.2×** |
| 160 | 5622 ms | 2961 ms | **1.9×** |

So a typical ~80-token chat prompt's first-token wait on the 1.5B drops ~2.7s →
~1.3s. Decode is untouched (M=1) — still ~73/~36 tok/s. Net: speculative parked,
but the kernel re-block delivered a real, felt 1.5B latency win via prefill.

### (superseded) Phase-0 recommendation → Phase 3 first, then Phase 1; Phase 2 not yet

1. **Phase 3 (parallelism granularity) — biggest, clearest lever.** Per-matmul
   `parallelCols` over-dispatches; the GOMAXPROCS sweep shows negative scaling
   past 2 cores and the profile is 70% park/wake. A **persistent worker pool**
   (no goroutine spawn/park per matmul), **coarser (whole-layer) parallelism**,
   and **QKV fusion** should let decode actually use the cores and climb toward
   the bandwidth roofline. Cross-repo (`aikit/linalg`).
2. **Phase 1 (kill per-token allocations) — cheap, do alongside.** 4660 allocs/op
   feeds GC, which compounds with the scheduler churn. Reuse scratch: the W8A8
   kernel's activation/result buffers (`aikit/linalg`) and `gatedMLP`'s
   intermediates (`goinfer/decoder`). Low-risk; mostly latency/jitter + a modest
   tok/s bump.
3. **Phase 2 (kernel tuning) — NOT the bottleneck here.** The dot kernels are only
   ~12–15% of CPU at 0.5B batch=1; tuning them won't move the needle while
   dispatch dominates. Revisit after Phase 3, once the kernel is exposed as the
   next ceiling (and it'll matter more on the 1.5B / larger shapes).

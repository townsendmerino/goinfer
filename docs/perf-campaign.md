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
30% sitting in X." **See `docs/task-perf-phase0-profile.md`.** Decide the next
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

### Recommendation → Phase 3 first, then Phase 1; Phase 2 not yet

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

# Task (aikit): perf campaign — the matmul dispatch + allocation levers

> **For:** the aikit Claude, in `~/tmcode/aikit` (`aikit/linalg`).
> **Why:** goinfer's Phase-0 decode profile (`goinfer/docs/perf-campaign.md`)
> found the two biggest decode costs live in `aikit/linalg`, not goinfer. This is
> the high-leverage half of the campaign. **Non-negotiable: HF logit parity must
> stay green** — goinfer has a `TestDecodeParity` guard and the existing
> `*_forward_golden` tests; any numeric change is reverted.

## The data (goinfer decode, Qwen2.5-Coder-0.5B int8int8, Apple M1 Pro 6P+2E)

- **~70% of decode CPU is `runtime.pthread_cond_wait`(38.8%) + `pthread_cond_signal`(31.4%)** — goroutine park/wake from `linalg.parallelCols`, which forks ~7 matmuls × 24 layers per token. At batch=1 each matmul slice is microseconds, so the fork/join wakeup costs more than the work.
- **GOMAXPROCS sweep proves over-dispatch:** 1→51, **2→60**, 4→60, **8→56 tok/s.** Parallelism past 2 cores buys nothing; 8 is *slower* than 2; single-thread is within 15% of best.
- The actual SDOT kernel (`dotI8SDOT`/`MatmulBTW8A8`/`dotI8`) is only **~12–15%** of CPU.
- **Allocations:** `MatmulBTW8A8.func1` allocates **1.55 GB** over the run (39% of alloc_space) — the kernel allocates the quantized-activation + result buffers **per call** (×168 matmuls/token × N tokens).

## Phase 3 — fix the dispatch (the ~70% lever)

`parallelCols` forks goroutines per matmul call and they park/wake every call.
Options (measure; pick what keeps parity and helps both 0.5B and 1.5B):

1. **Threshold:** below some FLOP/row count, run single-threaded — no goroutines
   at all. The sweep says small matmuls shouldn't be parallelized; this alone may
   remove most of the 70%.
2. **Persistent worker pool** instead of `go func()` per call: N long-lived
   workers fed by a lightweight mechanism, so there's no goroutine *creation* per
   matmul and ideally less park/wake (consider a small spin before parking, or
   batching).
3. **Coarser granularity / batched dispatch:** a primitive that runs several
   matmuls (or a whole layer's projections) in **one** parallel region, amortizing
   one fork/join over more work. This also enables **QKV / gate-up fusion** on the
   goinfer side cheaply — see the note below.

Target: positive scaling past 2 cores, and decode CPU no longer dominated by
`pthread_cond_*`. Re-run goinfer's `BenchmarkDecode` + GOMAXPROCS sweep to confirm.

## Phase 1 — kill the per-call kernel allocations

`MatmulBTW8A8` (and the int8 dot path) allocates the int8-quantized activation
and/or the result buffer on every call. At ~168 matmuls/token that's the 1.55 GB.
Provide a caller-supplied scratch / reuse path so a steady-state decode does zero
kernel allocs: e.g. an `*Workspace` arg, or quantize the activation into a
caller buffer. goinfer will thread the scratch through (it owns the per-stream
KV cache). Keep the allocating signature too (or a wrapper) so existing callers
don't break.

## Coordination note (QKV / gate-up fusion)

goinfer can fuse q/k/v (and gate/up) into single matmuls to cut dispatch count
43% (7→4 matmuls/layer). But the prequant `.giw` path **aliases** the int8
weights zero-copy from the binary image, so a goinfer-side *concatenate* would
force a heap copy and lose the RAM win. **If aikit exposes a "run these K weight
matrices against one activation in one parallel dispatch" primitive** (option 3
above), goinfer gets the dispatch reduction with **no weight copy** — the
preferred path. Tell goinfer which way you go so it fuses (concat) or batches
(multi-matmul) accordingly.

## RESULT — goinfer end-to-end arbiter (2026-06-05, M1 Pro, 0.5B int8int8)

Your v0.5.0 is wired in goinfer: per-stream `Workspace`, batched q/k/v + gate/up,
`Into` for o_proj/down/LM-head. **Allocations 4395 → 19/op (zero-alloc decode);
parity bit-identical.** Now the threshold call, from the real decode loop (the
arbiter your microbench couldn't reproduce — back-to-back steps keep the pool
warm, so the cold park/wake never dominates):

| config | P=1 | P=2 | P=4 | P=8 |
|---|---|---|---|---|
| serial decode (your default 16.78 M) | 51 | 51 | 51 | 52 |
| **parallel decode (threshold ≈0.3 M)** | 51 | 65 | **68** | 66 |

**Parallel decode wins: ~68 vs ~51 tok/s (and ~48 pre-campaign) — +40%.** The
Phase-0 "70% `pthread_cond`" was idle workers *parking*, not critical-path cost.

### → No aikit-default change needed: goinfer owns its threshold

Earlier draft said "lower aikit's default to 0.3 M." Revised — **goinfer will call
`SetParallelThreshold` itself** for its batch=1 decode (~0.3 M MACs). That's
better than baking goinfer's number into aikit's shipped default, because:

- **0.3 M is not universal** — it's the *M1-Pro / 0.5B / batch=1* optimum. The
  crossover moves on x86/AVX2, on fewer-core machines, and for the 1.5B (bigger
  matmuls parallelize sooner). aikit/linalg also serves the **encoder (reranker)
  and ken**, whose shapes differ; a global default tuned to goinfer decode could
  mis-tune those.
- **Decoupling:** goinfer's speed shouldn't depend on aikit shipping a specific
  default or silently change if aikit retunes later.

So: **keep aikit's default whatever your own benches justify (conservative is
fine), just keep `SetParallelThreshold` exported and overridable.** goinfer sets
~0.3 M for decode (and will re-sweep for the 1.5B). Treat 0.3 M as "a reasonable,
overridable, hardware-specific value," not a universal constant.

## Phase 3b — cut the per-dispatch fork/join latency (the next lever)

After v0.5.0's threshold + batching + zero-alloc landed and goinfer wired it,
**re-profiling the optimized 68-tok/s path shows ~71% of CPU is *still*
`pthread_cond_signal/_wait` from `parallelSpawnCols`** — but now it's the
*parallel* path (productive: 68 vs 51 serial). The tell: serial = 51 tok/s,
8-core = 68 → only **~1.3× scaling off 6–8 cores**, and effective bandwidth is
~18% of the M1 Pro roofline. Each batch=1 matmul's parallel work is microseconds,
so the **per-dispatch fork/join latency** (futex park/wake) dominates even after
batching reduced it to ~4 dispatches/layer.

**`parallelSpawnCols` spawns + parks per call.** A **persistent worker pool that
spins briefly before parking** (option 2 from the original spec, the part not yet
done) should keep workers hot across the ~4 dispatches/layer × 24 layers, so
back-to-back decode matmuls skip the full futex wake. Measure the scaling: target
materially better than 1.3× on 6 P-cores. (Pin to P-cores or let the pool size be
set — the M1 Pro's 2 E-cores may be causing load-imbalance stalls; goinfer's
sweep had P=4 ≈ P=6 > P=8.)

This is the path from 68 toward the bandwidth roofline. Honest ceiling: batch=1
has ~4 serial matmul dependencies/layer that can't merge, so it stays
fork/join-bound below the 330 tok/s pure-bandwidth figure — but 1.3× says a lot
is recoverable. goinfer re-runs `BenchmarkDecode` + the GOMAXPROCS sweep as the
arbiter (warm-pool microbenches will *under*-show the win, like last time).

### ARBITER VERDICT (goinfer end-to-end, 2026-06-05): pool does NOT win → pull it

You built it correctly (parity bit-identical, opt-in, -race clean). goinfer swept
it end-to-end — **neutral-to-slightly-slower**: spawn P=4 = 67.6 tok/s, pool
workers=4 P=4 = 64.1 (and ~62 elsewhere). The hypothesis was wrong: spawn ≈ pool,
so the "71% pthread_cond" was **not** recoverable latency — it's the inherent
batch=1 sync floor + idle-worker park samples that overcount. A hotter pool
doesn't remove a floor.

**Recommendation: ship v0.5.0 as the proven Phases 1/3 (threshold + batched +
zero-alloc, the validated +42%) and PULL the pool** (`pool.go` + `SetWorkers`/
`Close`) — exactly the option you offered. No consumer enables it, it doesn't
address the real limit, and unproven concurrency isn't worth the maintenance
surface. ~68 tok/s is the practical pure-Go batch=1 CPU ceiling; the rest is a
different-approach problem (and speed was never the pitch). Thanks for the clean,
honest build — the opt-in design is exactly why pulling it is trivial.

## Done

- [x] Dispatch reworked + batched primitive; goinfer decode is zero-alloc and the
      `pthread_cond_*` share is gone in serial / amortized in batched-parallel.
- [x] Kernel decode allocs ~zero with the `Workspace` path (verified: 19/op).
- [x] Parity bit-identical (goinfer `TestDecodeParity`).
- [x] `SetParallelThreshold` exported + tunable (goinfer sets it, not aikit's
      default). Keep aikit's default to whatever its own benches justify.
- [ ] Push the aikit v0.5.0 tag. goinfer then bumps `go.mod` v0.5.0, sets its
      decode threshold, commits its held wiring, re-sweeps the 1.5B, reconciles
      the doc tok/s numbers, and rebuilds the demos.

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

## Done

- [ ] Dispatch reworked; goinfer `BenchmarkDecode` shows the `pthread_cond_*`
      share gone and positive multi-core scaling.
- [ ] Kernel decode allocs ~zero with the scratch path.
- [ ] Parity green (goinfer `TestDecodeParity` + `*_forward_golden`).
- [ ] Cut an aikit release; tell goinfer the version + the fusion/batched-matmul
      decision so it bumps `go.mod` and wires its side.

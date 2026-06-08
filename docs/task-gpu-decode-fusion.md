# Task (goinfer/gpu): make the full-token decode forward actually fast

> **For:** Claude Code in `~/tmcode/goinfer` (`-tags gpu`, RTX 2070 SUPER /
> 3700X box). Read `docs/gpu-assessment.md` §0 + §0.5 first. The full-token
> GPU forward is built and bit-exact (`decoderunner.go`,
> `7cb7840`); the conclusion attached to it — *"staged hybrid is optimal,
> GPU residency loses"* — is **premature and almost certainly wrong**. This
> task is to falsify it properly.

## The mistaken conclusion and why it's wrong

`DecodeRunner` issues **~18 dispatches/layer × 28 = ~500 compute passes per
token**, each wrapped in its own `BeginComputePass … End`. At 17.5 tok/s
(57 ms/token) the actual weight-bandwidth work is ~3.5 ms (≈1.2 GB of int8
weights ÷ ~350 GB/s). **~93% of the token is per-pass overhead** (~100 µs ×
500), not compute. We are at ~7% of the memory roofline.

So "full-GPU residency loses to the staged hybrid" is not a fact about GPUs
— it's a fact about *this* implementation having the worst possible
dispatch structure. The staged hybrid wins only because it has ~113 syncs
vs ~500 passes; both are dominated by per-dispatch overhead, and the hybrid
just has less of it. **Neither variant has measured a GPU decode that
minimizes dispatches.** That is the missing experiment.

## The mental model to adopt

Decode (M=1) is **bandwidth-bound on the weights**. The activation vectors
are tiny (hidden=1536 → 6 KB); every elementwise op (rmsnorm, quantize,
RoPE, residual, swiglu) touches only those few KB and is essentially free
**as arithmetic**. Their entire cost is that each is a separate dispatch in
a separate pass. Therefore:

> **A fast GPU decode does the fewest passes possible, and an elementwise
> glue op is NEVER its own dispatch — it is fused into the matmul kernel it
> borders, or batched with its neighbors.**

Target: **~6–8 dispatches/layer in ONE compute pass**, reaching **>50% of
weight-bandwidth roofline** (≥100 tok/s on the 1.5B int8 on this card).
Don't compare to CPU until you're there; compare to the roofline.

## Experiments, in order of leverage (measure after each)

The first metric every step reports: **effective GB/s = resident_weight_bytes
× tok/s**, vs the card's ~448 GB/s peak / ~350 streaming. tok/s-vs-CPU is a
distraction until roofline utilization is healthy.

### 1. Collapse 500 passes → one pass per token (biggest, safest win)

Today every op is `BeginComputePass/SetPipeline/Dispatch/End/Release`.
Instead: **one `BeginComputePass` for the whole token**, all
`DispatchWorkgroups` recorded in order inside it, one `End`. WebGPU
guarantees in-order execution within a pass and the backend (wgpu) inserts
the minimal storage-buffer barriers between data-dependent dispatches —
this is correct, not a hazard. Bit-exactness must hold (it will).

- Blocker: the KV-append is a `CopyBufferToBuffer`, which **cannot** live
  inside a compute pass and currently forces a pass break (2/layer). Fix by
  removing the copy entirely: have the **RoPE kernel write K directly into
  `KCache` at `pos*kvDim`**, and a one-line store-kernel (or the V-projection
  epilogue) write V into `VCache`. Then there is no copy and the token is a
  single pass. (Alternatively accept 2 pass-breaks/layer — still ~57× fewer
  than now — and measure that first since it's a 10-minute change.)
- Expected: this alone should be most of the win if the ~100 µs/pass
  hypothesis is right. **If collapsing passes does NOT move the number,
  stop and instrument** — the cost is elsewhere (see §5) and the rest of
  the plan is moot until you know where.

### 2. Fuse the glue into the matmul kernels

Cut dispatch count structurally:

- **rmsnorm + activation-quantize → one kernel.** Both read the same
  hidden vector; emit the quantized int8 activation + scale directly.
- **residual-add as the matmul epilogue.** The o_proj and down_proj GEMVs
  already write `dst[n]`; pass the residual buffer and write
  `dst[n] = residual[n] + result` — deletes the separate residual pass.
- **swiglu stays one kernel** but feed it straight into the down-proj's
  activation-quantize (fuse swiglu+quantize like rmsnorm+quantize).

Per-layer dispatch count: ~18 → ~7 (qkv, attn, o, gate/up, down + the two
fused norm/quant kernels).

### 3. Concatenate QKV and gate+up weights at load

One GEMV over a `[q|k|v]`-stacked weight instead of three; same for
`[gate|up]`. Already proven for prefill (`BatchTiled`, `f9cab37`) — apply to
the decode GEMV. Fewer, larger, more efficient dispatches; slice the output.
Per-layer: ~7 → ~5 dispatches.

### 4. Coalesce the per-token uniform writes

`Run` does a `WriteBuffer` per `posUni` every token. Pack all pos-dependent
uniforms into ONE buffer and write once. Minor vs §1–2, but free.

### 5. If §1 underdelivers: instrument before theorizing

- Enable timestamp queries if the binding exposes them (`device.go`
  features); otherwise bisect with wall-clock: time an N-dispatch
  no-op-kernel pass vs the real pass to separate per-pass cost from kernel
  cost.
- Confirm the big GEMVs (gate/up/down, ~13 MB each) actually hit bandwidth
  in isolation — bench one resident GEMV standalone, compute its GB/s. If a
  single GEMV is already far below roofline, the kernel (not the dispatch
  count) is the problem and §1–3 won't save it. (The kernel in `gemv.go`
  looks bandwidth-correct — coalesced vec4 loads, workgroup-per-row,
  tree-reduce — so this is a check, not the expected culprit.)

## Gate (this is what falsifies-or-confirms the §0 conclusion)

- **Primary:** full-token GPU decode on the 1.5B int8 `.giw`
  (`cmd/prequant`) reaches **≥50% weight-bandwidth roofline** (≥~100 tok/s,
  ≈4–5× the staged hybrid's 25.6). Hit → `docs/gpu-assessment.md` §0/§0.5
  are rewritten: residency wins once dispatches are minimized; the staged
  hybrid was a local optimum of an unfused implementation.
- **Miss after §1–3, with §5 instrumentation showing where the time goes:**
  *then* the "staged hybrid optimal on this HW" conclusion is earned, not
  assumed — record the per-pass cost number that makes it true (e.g.
  "wgpu charges X µs/pass on NVIDIA Vulkan, irreducible without a native
  backend"), which is the real finding.

## Rules

- Bit-exactness (cosine 1.0 vs CPU) after every step — the existing
  `decoderunner_test.go` / `e2e_test.go` parity gates stay green.
- Pure-Go core untouched; all of this is `-tags gpu`.
- `dot4I8Packed` is still upstream-blocked — out of scope here; this task is
  about dispatch structure, which is the dominant cost regardless.
- Report effective GB/s at each step, not just tok/s.

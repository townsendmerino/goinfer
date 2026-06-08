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

### RESULT LOG

- **§1 done (commit it):** KV copies eliminated (RoPE writes K into cache,
  kv-store writes V; offsets ride posUni), whole token in one
  `BeginComputePass`. Bit-exact (cosine 1.0). **17.6→22.4 tok/s, 27.3→34.7
  GB/s = 1.27×**, ~8%→~10% of roofline. **The §1 "93% pass overhead"
  hypothesis was wrong** — pass wrapper is only ~24 µs each; 44.7 ms (~10%
  roofline) remains, ~80 µs/dispatch unexplained. ⇒ **§5 instrumentation
  BEFORE §2** — that's the second hypothesis about this code to miss by ~4×
  (after the §0.5 1.48× probe); stop predicting, decompose the 44.7 ms.

- **§5 done — the decode is NOT bandwidth-bound; it is glue-serialization-
  bound.** Measured (`decode_instrument_test.go` + the throughput bench's
  phase-split / step-attribution; 2070 SUPER):
  - **Phase split of `Run`:** write 0.4 ms · encode 1.4 ms · **submit+poll
    42.3 ms**. The whole token IS GPU-execution time; host-side (the
    WriteBuffer storm §4 worried about) is negligible. §4 is moot.
  - **(a) gate GEMV [8960×1536] standalone: ~37 µs/dispatch = ~380 GB/s
    (≈108% of the 350 estimate, near the 448 peak).** The matmul kernel is
    bandwidth-perfect. Not the problem. Whole-token matmul stream = **4.1 ms**.
  - **(b) per-dispatch floor: 1.2–1.5 µs** (same-bg AND distinct-buffer
    550-deep RAW chain both). Dispatch/launch/barrier overhead is NOT the
    cost. The "~80 µs/dispatch" guess was wrong too (3rd miss).
  - **Step attribution (real built plan, GPU exec): all 41.9 ms = gemv 4.1 +
    glue 37.2.** Glue (rmsnorm/quant/rope/attn/swiglu/residual) is **89% of
    the token** while touching only KB each.
  - **The tell:** timing each glue pipeline *type in isolation* sums to only
    ~7–11 ms, but the glue dispatches *run together in the real dependency
    chain* cost 37 ms. The ~30 ms delta is **dependency-chain serialization**:
    isolated same-type dispatches are independent and the GPU overlaps them;
    in the real token each link (rmsnorm→quant→gemv→…→attn→…) forces a
    barrier the GPU can't hide, and the per-link drain/cache-flush latency
    scales with the data the bordering kernel touched (why (b2)'s 256-byte
    chain was free but the real 13-MB-gemv chain is not). `attn` alone
    (`@workgroup_size(1)`, one thread/head) is ~5.8 ms even overlapped.
  - **Verdict (refines the decision rule):** (a) fast + (b) low floor + (c)
    glue-dominated — but NOT via dispatch overhead. The matmul roofline floor
    is **4.1 ms ⇒ ~240 tok/s** if glue were free. So **§2/§3 fusion IS the
    lever**, but for the *measured* reason — it shortens the serialized
    dependency chain (~18→~5 dispatches/layer ⇒ fewer barrier links), not
    because "elementwise is free and dispatch is the cost" (the doc's premise,
    now falsified: glue arithmetic is cheap, glue *serialization* is the tax).
    **Plus a second, independent lever the doc underweighted: the `attn`
    kernel must be parallelized** (warp-per-head, not 1 thread) — it is the
    single largest kernel and fusion won't touch it. Do §2 (fuse
    rmsnorm+quant, residual epilogue, swiglu+quant) AND the attn rewrite; both
    are now justified by numbers. The §0 "staged hybrid optimal" claim stays
    falsified: the GPU floor is 4.1 ms, the implementation just serializes 535
    dependent dispatches on top of it.

### 5. Instrument before §2 — decompose the remaining 44.7 ms

Three measurements that fully split the cost; the decision rule chooses
§2 vs kernel-fix vs earned-conclusion. **Do not start §2 until all three
are in.**

- **(a) one resident gate/up GEMV (~13.8 MB) standalone in a loop → GB/s.**
  ~40–80 µs ⇒ kernel is bandwidth-correct, problem is dispatch count →
  §2/§3 pay. 200 µs+ ⇒ kernel is the problem, fusion won't save it. (The
  `gemv.go` kernel looks bandwidth-correct — coalesced vec4 loads,
  workgroup-per-row, tree-reduce — so this is a check, not the expected
  culprit, but the §1 miss says verify, don't assume.)
- **(b) a pass of K no-op / 1-workgroup dispatches → per-dispatch floor.**
  Isolates launch+barrier from kernel work. Floor that scales with K ⇒ cut
  dispatch count (§2/§3). High floor independent of work ⇒ wgpu/Vulkan
  structural cost → native-backend territory.
- **(c) full forward, glue dispatches removed (matmul-only, wrong output,
  timing only).** Splits 44.7 ms into matmul-time vs glue-dispatch-time.

Decision rule: (a) slow → fix kernel; (b) high floor + (c) glue-dominated
→ §2/§3 fusion lands the remaining ~10×; (b) high irreducible floor →
write the "staged hybrid optimal, X µs/dispatch on this stack" conclusion
with the number behind it.

### §5 RESULT (committed 626b3eb) — the actual finding

Decode is **glue-serialization-bound, not bandwidth- or dispatch-bound.**
- phase split: write 0.4 / encode 1.4 / submit+poll 42.3 ms ⇒ all GPU exec.
- (a) gate GEMV standalone ~380 GB/s ⇒ kernel bandwidth-perfect.
- (b) per-dispatch floor 1.2–1.5 µs even at 550-deep distinct-buffer chain
  ⇒ dispatch overhead is NOT it (kills the ~80 µs/dispatch guess).
- attribution: 41.9 ms = gemv 4.1 + glue 37.2. Matmul roofline floor is
  4.1 ms (~240 tok/s if glue were free); glue is 89% of the token.
- the tell: each glue pipeline isolated sums ~7–11 ms, but the SAME
  dispatches in the real RAW dependency chain cost 37 ms. The ~30 ms gap
  is per-link barrier serialization — isolated dispatches overlap; chained
  ones force a drain whose latency scales with the bordering kernel's data.

### §2/§3 — REVISED by the §5 finding (the lever is critical-PATH length)

Because the cost is the serialized RAW spine, optimize **critical-path
links, not dispatch count:**

- **Fold every `quantize` into its producer** (rms+quant, swiglu+quant,
  attn-context-quant into the attn epilogue) — 4 links/layer gone.
- **Fold both residual-adds into the o-proj / down-proj epilogues** — 2
  links gone.
- **Fold RoPE into the qkv epilogue** — 1 link gone. Path ~15 → ~8/layer.
- **§3 QKV / gate+up concatenation removes DISPATCHES but not LINKS**
  (distinct-buffer writes already overlap per (b)); it helps
  occupancy/bandwidth, not serialization — deprioritize vs the fusions.
- **If per-link drain scales with data:** the 8960-wide MLP intermediates
  (swiglu out, quant-mid) are the costly borders — fusing swiglu+quant so
  that ~36 KB vector never materializes/re-reads is a double win; keep it
  in workgroup memory.
- **Parallelize the attn kernel** (`@workgroup_size(1)` → warp-per-head):
  ~5.8 ms alone, on the critical path, untouched by fusion. Independent
  work — land in parallel.

Honest expectation: fusion (~15→~8 links) + attn fix ≈ 2–2.5× → ~45–60
tok/s. That decisively beats the staged hybrid (25.6) and overturns §0.
**The ≥100 tok/s / 50%-roofline gate may be unreachable in WebGPU** — the
only way past a per-link serialization floor is a single megakernel
(persistent-thread, whole layer in one dispatch), which WGSL can't express.
If fusion plateaus short of the gate, that ceiling IS the finding:
"WebGPU's dispatch model floors decode at ~N tok/s; native CUDA/Metal
megakernel is the only way past." Measure; if the plateau appears, write
it rather than grind.

### §2 RESULT (committed fbdb71f + decfccd) — 27→85 tok/s; §0 falsified

All bit-exact (DecodeRunner parity cosine=1.000000 maxAbs=0 after every
step; attn cosine=1.0 maxAbs=1.2e-7 from f32 reduction order). Steps, each
measured (1.55 GB resident, 2070 SUPER):

| step | tok/s | GB/s | all-GPU ms |
|---|---|---|---|
| §1 single-pass (baseline) | 22.4 | 34.7 | 41.9 |
| + rms+quant fused | 23.6 | 36.5 | 39.7 |
| + swiglu+quant fused (36 KB double-win) | 27.0 | 41.8 | 34.7 |
| + residual→gemv epilogue | 27.1 | 42.0 | 34.7 |
| **+ attn warp-per-head** | **84.5** | **130.8** | **9.7** |

- **The fusions landed ~1.2× — modest, as the serialization model predicts:**
  the rms/residual links border small (6 KB) buffers so their per-link drain
  is cheap; swiglu+quant was the exception (36 KB intermediate kept off the
  spine, +0.8× alone).
- **The attn rewrite landed 3.1× by itself — far more than its isolated
  5.8 ms.** `@workgroup_size(1)` (12 single-thread workgroups) wasn't just
  slow, it was the chain's serialization bottleneck: un-overlappable, 28× on
  the spine, every layer stalled on it. Warp-per-head (one workgroup/head,
  128 lanes, tree-reduced scores) dropped it to 0.3 ms AND collapsed the
  whole-token GPU time 34.7→9.7 ms. The serialization tax was concentrated in
  this one kernel far more than the link-count model suggested.
- **Result: 84.5 tok/s = 3.3× the staged hybrid (25.6), 9.95× CPU, 49% of the
  CUDA ceiling (171), 37% of the streaming roofline.** Step attribution:
  9.7 ms = gemv 4.3 (the matmul floor, already ~360 GB/s = at roofline) +
  glue 4.0. **§0 "GPU residency loses" is decisively falsified** —
  `docs/gpu-assessment.md` §0/§0.5 should be rewritten: residency wins
  handily once the dependency chain is unblocked; the staged hybrid was a
  local optimum of an unfused, single-threaded-attn implementation.

**On the ≥100 / 50%-roofline gate (currently 84.5 / 37%):** within ~16% but
not reached. The matmul floor is 4.3 ms (≈230 tok/s at 100% roofline); the
token is 11.8 ms = 9.7 GPU + 2.1 host. Closing the remaining gap means
shaving the 4.0 ms glue and ~2 ms host, i.e. more link-folds (vStore→V-gemv
epilogue, rope-q→attn-on-read) and uniform coalescing (§4) — diminishing
returns on small-buffer links, plus §3 QKV/gate-up concat (occupancy, not
serialization — deprioritized). Whether 50% roofline is reachable in WebGPU
or is the per-link-serialization wall the megakernel-only argument predicts
is the open question; the result already overturns §0 regardless. **Stopped
here for a decision rather than grinding the last 16%.**

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

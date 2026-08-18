# Prompt: prototype a batched small-M verify path on Metal (run on the M1 Pro)

> **STATUS: DONE (2026-08-17, `mac`).** This prompt and the direct instruction that landed the
> work crossed in flight — both converged on the same task independently. Full results, all five
> phases below, are in `docs/task-metal-batched-verify-kernel.md`: Phase 0's numeric contract
> confirmed (exact per-32-block int dot, decode's real `SA_BODY`/`W4A8_BODY` reduction, matching
> this prompt's own recon); Phase 1's threadgroup-budget design (`bvkMaxM`, capped by `2*K*M`
> against the 32 KiB limit — model-shape-dependent, not universal); Phase 2's prototype kernels
> (`metal/batched_verify_kernels.go`, `metal/batched_verify_test.go`, additive-only, no decode
> changes); Phase 3's bit-identity parity — **bit-exact `==`, not cosine, exactly as this prompt
> required** — green across M∈{2,4,8,16} on qwen2.5-coder-0.5b and M∈{2,4} on gemma-3-4b (M=8/16
> correctly declined there, not silently skipped — over that model's real threadgroup budget);
> Phase 4's ceiling re-derivation — **NO-GO**, every k below break-even, worse than the prior
> ~1.13× MMA-route estimate despite fixing the bit-identity blocker, because only 4 of ~12
> per-layer dispatches got batched (the marginal cost per verified token ends up ≈ real decode
> cost). Do not re-run this prompt; read the results doc instead.

Dispatched 2026-08-16, superseding the timing-only task in `metal-verify-curve.md`. That one
measured the ceiling (~1.13×) and found the real blocker: `metal.PrefillLast` is declined in
production because it is not bit-identical to decode. **This task attacks that blocker directly**
— an M-row hoist of the resident int8 decode matmuls that IS bit-identical, which is the
prerequisite for P10 on Metal existing at all.

Relationship to the CUDA leg: the same lever at a different layer. CUDA's verify already batches
its layer stack and was found to leave the **LM head** unbatched (weights re-read per row). Metal
does not batch the layer matmuls at all. Both are weight-read amortization.

**One bar difference worth carrying across.** This task's bar is bit-exact `==`, correctly — the
layer matmuls feed the residual forward, so any divergence compounds. The CUDA head work uses a
deliberately *weaker* bar: only the **argmax** must match, because the head's output feeds only
the accept comparison and nothing downstream. Do not weaken this task to the argmax bar; the
distinction is about what consumes the output, not about convenience.

---

**Task:** Prototype a batched small-M verification path on Metal — an M-row hoist of the resident
int8 decode matmuls — with bit-identity to sequential decode, and re-derive the P10-on-Metal
ceiling. One consolidated doc under `docs/` for all phases.

**Repo:** `~/tmcode/goinfer`. Prior context: `docs/ollama-chase.md` §A2-Metal,
`docs/task-int4-int8-exact-mma.md` (MMA route killed: decode's `simd_sum` combine order
unpinnable; this hoist is the surviving bit-identical path). **Recon already done — build on it,
don't redo it:** the dense per-layer sequence is 12 dispatches (`metal/model.go:1171-1315`);
Stage-A kernels (`pSA`/`pSABias`/`pSAResid`, K≤1536) stage via `DispatchTG` with `tgBytes=H*2`;
`pGemvResid` (down-proj, K=8960) doesn't stage; threadgroup limit is
`d.MaxThreadgroupMemoryLength()` (~32,768, asserted in `metal/tgbudget_test.go`), and
over-budget dispatches **silently no-op** (see the comment around line 649 of aikit's own
`gpu/metal.go` — that file is in the aikit module, not this repo) so the check is on you;
primary fixture `qwen2.5-coder-1.5b` (H=1536, I=8960) via `GOINFER_METAL_MODEL`, second family
`gemma-3-4b`, heavy tests gated by `GOINFER_HEAVY_TESTS=1`, test hooks need
`//go:build darwin && goinfer_testhooks`.

**Phase 0 — resolve the staging contradiction, then write the numeric contract.** The tgbudget
comments say Stage-A stages 2 bytes/element (f16), but the prior investigation recorded decode's
matmul contract as exact-int32 dot per 32-block over int8 activations with per-block f32 scale.
Read the Stage-A and gemv kernel sources and determine exactly what the staged bytes hold and
where dequantization (if any) happens. Then write the per-matmul numeric contract the batched
kernel must replicate: staged representation, per-lane block iteration order, per-block dot
precision, scale application point, cross-block accumulation, `simd_sum` combine, final scale.
**Scope:** only the four matmuls (`pSABias`, `pSAResid`, `pSA`, `pGemvResid`) get batched;
RMSNorm, RoPE, KV-store, attention, context-quant and SwiGLU stay per-token dispatches (attention
must, for per-token `nKeys` and reduction order). If the hoist can't preserve per-token order for
any matmul, stop and report.

**Phase 1 — design within the threadgroup budget.** For the Stage-A batched variants at
M×K×(bytes/elem) against the 32 KiB limit, evaluate the three order-preserving options: (a) cap M
at the largest full-staging value (M=8 at H=1536 f16), (b) K-tiled staging in ascending block
order, (c) non-staged batched variant reusing the `pGemvResid` structure. Also estimate per-lane
register/accumulator growth with M and the occupancy knee on the M1 Pro. Pick a design, justify
it, size `tgBytes` the way `BuildResident` does (`maxThreadgroupStageBytes` pattern,
`metal/model.go:291-`, checked against `MaxThreadgroupMemoryLength`).

**Phase 2 — prototype.** New kernels + a batched encode path alongside the existing ones, behind
a flag or build tag; **do not modify the existing decode kernels or dispatch path.** The resident
scratch buffers (`aq`/`aSc`, `cq`/`cSc`, `mq`/`mSc`, `dq`/`dSc`, `x`, `gu`, …) are sized for M=1 —
add M-row batched scratch (allocation pattern per `BuildResident`) rather than resizing the
existing ones. Support M ∈ {2..8} minimum; up to 16 only if Phase 1's design covers it. Correct
causal structure: all M draft tokens known upfront — batched QKV, per-token RoPE/KV-store in
position order, per-token attention over keys ≤ pos_i, batched O-proj and FFN matmuls, per-token
residual rows. Use single-commit encoding for the whole verify step (the `Run1DBatch` reps
pattern amortizes commit tax; here it is one command buffer containing the full per-layer
sequence across M).

**Phase 3 — bit-identity parity, Metal vs Metal.** New `*_parity_test.go` (tags
`darwin && goinfer_testhooks`, `GOINFER_HEAVY_TESTS` gate, `GOINFER_METAL_MODEL` fixture): load
one resident, run M tokens sequentially via `rf.Forward` recording logits and hidden states per
position, reset/replay KV state, run the batched path on the same tokens, assert **byte-for-byte
equality** at every position. M ∈ {2, 4, 8} (+16 if supported), on both fixtures. **Do NOT use
the cosine `assertParity` style** — that bar exists for GPU-int4-vs-CPU-int8 comparisons
(the cosine bar near line 64 of the metal gemma parity test); this is same-resident
Metal-vs-Metal and the bar is `==`. Any
mismatch means Phase 0 missed something; find it, don't loosen the test.

**Phase 4 — performance and ceiling.** On the M1 Pro 16 GB: (a) batched verify step vs M×
sequential decode steps for each supported M — the amortization curve; (b) dispatch/commit
overhead per verify step vs M sequential steps, since the prior analysis flagged Metal's
dispatch-cost shape as the second blocker; (c) both fixtures. Then recompute the P10-on-Metal
speculative speedup bound from these measured numbers plus the CUDA-side acceptance data, show
the arithmetic, and state go/no-go.

**Constraints:** investigation + prototype, not a ship — no public API changes, no changes to
decode numerics or the release path, everything behind the flag/tag. **Deliverable:** one
consolidated doc under `docs/` (all phases, including the Phase 0 contract and Phase 1 design
rationale), the prototype kernels + batched encode path, the parity test. **A measured negative
at any gate, written up, is a fully successful outcome.**

Phase 0 alone is worth a report-back — the f16-staging-vs-int32-dot question is the one thing in
the recon that could still change the design.

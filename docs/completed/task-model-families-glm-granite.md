# Plan: two next model families — GLM (now), Granite 4.0 Hybrid (the v1.0 trigger)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> **Audience:** internal planning, in the style of `roadmap.md` Track A. Two
> families, sequenced by the "popular *and* achievable" filter: **GLM first**
> (rides the shipped MoE path, no new primitive — momentum), **Granite 4.0
> Hybrid second** (the genuine *second hybrid family*, one new sequence-mixer on
> already-built hybrid scaffolding — the milestone). The recurring muscle holds:
> each family = **arch adapter + tensor schema + loader + parity golden**, and
> the whole v0.2–v0.5 surface (chat templates, tools, constrained decoding,
> `cmd/serve`, sampler) inherits automatically.

## Correction to the roadmap line that motivated this

The roadmap said *"Granite/GLM on the deltanet shapes = the v1.0 trigger."* Two
facts (verified 2026-06) make that imprecise, and the imprecision is load-bearing:

- **GLM-4.6 is not a hybrid.** It's a standard **GQA MoE** (355B / ~32B active,
  partial RoPE, QK-Norm). It lands on the *already-shipped* MoE path and touches
  the deltanet/hybrid-cache shapes **not at all**.
- **Granite 4.0 is Mamba-2, not DeltaNet.** It's a hybrid **Mamba-2 + Transformer
  (≈9:1) + MoE**. It exercises the *hybrid-cache abstraction* — but with a
  **different linear mixer** than our Gated DeltaNet.

So the real v1.0 evidence is *not* "GLM/Granite reuse deltanet." It's: **does the
hybrid-cache + per-layer-kind abstraction survive a second, architecturally
different linear mixer (Granite's Mamba-2)?** GLM is momentum; Granite is the
milestone. This plan is built on that distinction.

---

## Part 1 — GLM (target GLM-4.5-Air, then GLM-4.6)

### Architecture (what we're matching)

GQA attention + **partial RoPE** (`partial_rotary_factor`) + **QK-Norm**; a
DeepSeek-style **MoE** FFN with a **shared (always-on) expert** and top-k routed
experts; **MTP** (multi-token-prediction) heads we ignore, exactly as we ignore
Qwen3.6's. No linear attention, no SSM — every mechanism is one goinfer already
ships.

### What's already in place (map each piece to its seam)

| GLM needs | goinfer already has |
|---|---|
| MoE router + top-k + renorm | `MoEConfig{NumExperts, TopK, NormTopKProb}`, `moeMLP` (`mlp.go`) |
| Shared always-on expert | `MoEConfig.SharedIntermediateDim` + `LayerWeights.SharedExpert`/`SharedGate` (Qwen2-MoE) |
| Partial RoPE | `Architecture.RotaryDim` (rotates a prefix of the head dim) |
| QK-Norm | `Architecture.QKNorm` (Gemma 3 / Qwen3) |
| MTP heads | ignored at load (precedent: qwen3.6 VL — load the text decoder, drop MTP/vision) |
| big-MoE footprint | **expert demand-paging (#2, shipped)** — the 355B flagship runs under a RAM budget |

### The work

1. **`glmArchitecture(cfg) (*Architecture, *tensorSchema, error)`** in
   `registry.go` + a `glmTensorSchema` (GLM tensor names: shared-expert,
   QK-norm, partial-rotary). Register it in `knownModelTypes`.
2. **Loaders:** safetensors first (the parity oracle path), then the **GGUF
   loader** — reuse the `qwen3_5_moe` *transform-reverser* pattern if GLM's GGUF
   converter bakes layout transforms (fused/stacked experts, q‖gate splits) the
   safetensors path doesn't see; prove each transform-bearing tensor with a
   `weightDiff` test against the bit-exact safetensors loader (no oracle needed).
3. **MTP / output head:** confirm tied vs untied head; drop the MTP head tensors.
4. **QK-norm × partial-RoPE ordering (assembly subtlety).** Each ships alone, but
   the *combination* has an order: QK-norm is over the full `head_dim`, partial-RoPE
   rotates only the `RotaryDim` prefix. goinfer's gemma3/qwen3 do QK-norm → *full*
   RoPE; GLM is QK-norm → *partial* RoPE. Confirm the order matches HF
   (`q_norm` then `apply_rotary(prefix)`) — a 1-line detail, silent-wrong-logits if
   reversed. The layer-slice oracle (above) catches it.

### Target + gates

- **Parity-testable target: GLM-4.5-Air** (~106B / ~12B active). ⚠️ **Correction
  (2026-06-15): a *full* bf16 forward oracle does NOT fit the box.** A forward oracle
  needs every weight resident — 106B × bf16 ≈ **212 GB** on a **65 GB** box, the same
  wall qwen3.6-35B hit (≈70 GB → OOM-skip). So Air gets the **qwen35 pattern, not a
  full oracle**: (a) **`weightDiff`** (bit-exact goinfer loader vs the Q8_0 GGUF,
  no oracle needed), and (b) a **layer-slice numeric oracle** — load embed + layers
  0–3 at bf16 and run goinfer's forward on that slice vs HF (the `qwen35_realckpt`
  Gate-1 pattern: catches a wrong name/prefix/expert-stacking cheaply, no full model).
  **GLM-4.6** (355B) rides #2 expert-paging; `weightDiff`-proven, same call.
- **Gate:** `weightDiff` (loader bit-exactness) + the layer-slice argmax/cosine vs
  HF bf16 + GGUF `weightDiff` to Q8_0 tolerance + coherent generation on canned
  prompts. (No full-model bf16 logit oracle — infeasible at this scale on the box.)
- **Inherited for free:** chat template (GLM's), tool calling, constrained
  decoding, `cmd/serve` routing — the descriptor-close process delivers all of it.

### Effort / risk

**Low–medium. No new forward-pass primitive** — it's the recurring
descriptor+loader+golden muscle, with GLM-specific tensor names and the
shared-expert/partial-RoPE/QK-norm combination as the only assembly work.
**Does not advance v1.0** (standard attention — proves nothing new about the
loader contract); its value is popularity + a real demo of #2 paging.

---

## Part 2 — Granite 4.0 Hybrid (the v1.0 trigger)

### Architecture

Hybrid **Mamba-2 (state-space) + Transformer**, roughly **9 Mamba blocks : 1
attention block**, with **MoE** in the H variants (e.g. H-Tiny ≈ 1B active / 7B
total). Mamba-2 is a *selective state-space scan* — a different linear-attention
primitive from our Gated DeltaNet, with its own (A, B, C, Δ) recurrence.

### What's already de-risked (the qwen3_5_moe groundwork)

The hybrid *plumbing* exists and is the expensive part normally:

- **Per-layer-kind dispatch** — `arch.isLinearLayer(i)` + the branch in
  `forward_qwen35.go` already split a model into linear vs softmax layers.
- **Hybrid cache** — `kvcache.go`'s `delta []*deltaState` already carries
  recurrent state *alongside* the KV cache. A Mamba-2 SSM state is a second kind
  of recurrent state slotting into the same structure.
- **Opt-outs already handled** — recurrent layers opt out of prefix-reuse and
  speculative decode (the deltaState isn't position-truncatable); Granite's Mamba
  layers inherit that handling unchanged.

### The one genuinely new thing

A **Mamba-2 selective-scan sequence mixer** (`mamba2.go`), alongside `deltanet.go`.
The recurrence math differs (selective SSM: input-dependent A/B/C/Δ, the
chunked/segsum parallel scan), so this is real forward-pass work — but it's *one
primitive on built scaffolding*, not a from-scratch hybrid runtime.

### The work

1. **`mamba2.go`** — the selective-scan primitive: sequential recurrence
   (the parity oracle) + a chunked/segsum parallel scan for prefill, proven
   **algebraically equivalent** over random inputs/chunk sizes (the exact pattern
   `deltanet_chunked.go` / `TestGatedDeltaNet_chunkedMatchesSequential` already
   set — self-contained, no asset/torch).
2. **A Mamba-2 cache-state kind** in `kvcache.go` next to `deltaState`. Note Mamba-2
   carries **two** recurrent states per layer, not one: a short causal **conv1d
   window** *before* the scan + the **SSM state** itself. So the state-kind is
   `{conv window, ssm state}` — a small extension of `deltaState`'s single slot;
   size it for both from the start. Wired through the hybrid cache.
3. **`graniteArchitecture(cfg) (...)`** + `graniteTensorSchema`: per-layer-kind =
   `{mamba | attention}` (generalize `isLinearLayer` to a layer-kind enum if the
   binary split gets strained), MoE config for the H variants (reuse `moeMLP`).
4. **Loaders** safetensors → GGUF (transform-reverser as needed), per usual.

### Why this is the v1.0 trigger

It's the **second hybrid family on the hybrid-cache/per-layer-kind shapes** — and
crucially a *different mixer* (Mamba-2 vs DeltaNet). If those shapes absorb it
without breaking, the hybrid abstraction has passed the same "second family"
bar the transformer descriptor already passed (Mellum2 after Qwen). That's the
written v1.0 criterion. Granite is also enterprise-popular and explicitly
small-RAM / long-context (≈70% less memory) — squarely goinfer's positioning.

### Target + gates

- **Target: Granite-4.0-H-Tiny** (≈7B total, ≈1B active) — small enough for a full
  bf16 forward oracle on the box, so no OOM-skip; the larger H variants follow.
- **Gates:** the scan-equivalence test (model-free); full argmax + cosine vs the
  HF bf16 reference; GGUF `weightDiff`; coherent generation. Same bar every family
  clears.

### Effort / risk

**Medium–high.** The Mamba-2 scan is the only research-flavored piece, and it's
de-risked by (a) the deltanet chunked-scan precedent for *how* to validate, and
(b) the hybrid cache + per-layer-kind plumbing already shipping. Bounded to one
new primitive + its parity, on existing scaffolding.

---

## Sequencing, non-goals, and the v1.0 note

**Order:** GLM now (low-risk popular win + a real #2-paging showcase), Granite
after (the milestone). They share nothing on the hot path — GLM is pure MoE,
Granite adds the Mamba mixer — so GLM can't accidentally de-risk Granite; do them
in series. **Caveat (2026-06-15):** GLM explicitly does *not* move v1.0, so the only
risk in going GLM-first is letting its easiness make it the destination. Granite is
the milestone — bag GLM as a quick win, keep Granite as the actual target.

**Explicitly deferred (unchanged):**
- **DeepSeek V4** — **MLA** (multi-head latent attention) is a new attention
  primitive we don't have, *and* the scale makes a parity oracle infeasible on
  current hardware (we already OOM-skip the 35B full forward). Prestige, not
  achievable now.
- **Gemma 3n** — adjacency is tempting (its **PLE** stack already ships from the
  Gemma 4 E-models), but it piles on **MatFormer** nesting, **AltUp**, and
  **LAuReL** — several new mechanisms for no v1.0 credit (not a hybrid). Revisit
  if on-device demand pulls.

**v1.0:** GLM does not move the trigger; **Granite landing clean on the
hybrid-cache shapes is the evidence that freezes the hybrid abstraction for
v1.0** — the same two-consecutive-families bar the transformer descriptor met.

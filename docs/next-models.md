# Next model families — the breadth backlog, and what became of the last list

> **Audience:** internal planning. Renamed from `post-v1.0-models.md` on 2026-09-12: v1.0 has not
> been cut (the latest tag is v0.17.2), the "post-v1.0" backlog got built anyway, and the old name
> answered a question nobody was asking. What has *landed* is never read from here —
> [`docs/capability-matrix.md`](capability-matrix.md) (families, generated from the registry) and
> [`docs/hardware-matrix.md`](hardware-matrix.md) (which backend runs each resident, generated from
> the admission predicate) are the truth. This doc is for *which family next, and why*.

## Where the 2026-08-17 list ended up (status 2026-09-12)

| # | item | status | record |
|---|---|---|---|
| 1 | Nemotron 3 Nano (`G4`) | **landed**, T3'd on a real checkpoint; Nemotron 3.5 Lightning confirmed config-identical, no registry change | [`task-families-2026-09.md`](task-families-2026-09.md) F2 |
| 2 | Qwen3-Next (`G5`) | **landed**, resident on WebGPU, CUDA and Metal | matrix |
| 3 | Laguna (`G6`) | **landed**, CPU path (softplus attention gating and per-layer RoPE scales were the new primitives, as scoped) | matrix |
| 4 | gpt-oss residency (`G7`) | **landed**, resident on all three backends. The one real forward its gate demanded found three silent wiring defects that a 100%-green kernel suite could not see — the lesson `G8` now rests on | [`completed/queue-correctness.md`](completed/queue-correctness.md) |
| 5 | DeepSeek V4-Flash (`G8`) | **PARKED — unvalidatable here.** 0.16 TB against 62 GB of host RAM; no configuration runs one whole forward on hardware this project has. The fp8 prerequisite is half lifted: e4m3 with blockwise f32 scales reads (`decoder/fp8.go`), V4's `ue8m0` scale format does not | [`queue-correctness.md`](queue-correctness.md) |

From the watch list: **Ling 3.0 landed** as `bailing_hybrid` (v0.17.0) — KDA became the fifth
sequence-mixing family, the exact thing the old text said would be "natural" and told us to
verify the weights for first. **Qwen3.8** (the dense 27B) landed. From Tier B, **InternLM2** and
**`qwen3_moe`** landed. Of the seven families in v0.17.0 (`qwen3_moe`, dense Granite, Ministral 3,
SmolLM3, Olmo 3, Olmo Hybrid, Ling 3.0), none came from this doc's tiers — they came from
`task-families-2026-09.md` and the G-series. That is the actual finding of the period: this doc
stopped being where families came from once the axes were covered, which is what it predicted.

## Watching — current list

- **Kimi K3** — reportedly ~3–4T params; *not* a K2.x point release (K2 through K2.7-Code are
  DeepSeek-V3-shaped and already run as `kimi_k2`). Phase-0 scoping
  ([`completed/task-model-family-deepseek-v4-kimi-k3.md`](completed/task-model-family-deepseek-v4-kimi-k3.md))
  found MLA on only 24 of 93 layers, the other 69 a KDA mixer — which `bailing_hybrid` now
  implements — plus a latent-MoE wrapper, a new activation, and MLA-with-NoPE + output gate. Weights
  ship `mxfp4-pack-quantized` in a `compressed-tensors` layout not yet confirmed to match
  `decoder/mxfp4.go`'s packing. Same hardware wall as V4-Flash (1.56 TB). Re-check when weights drop.
- **GLM-5.x** — the ~753B flagship is out of scope; wait for an Air-class size. Shares the
  native-sparse-attention direction with V4-Flash; if the DSA path is ever built, evaluate against it.
- **MiniMax M3** — 428B MoE, ~22B active, natively multimodal. Too big for the niche, and the
  multimodal training makes text-only support less compelling than the size alone suggests.
- **Qwen3.8-Flash-Next (Qwen4 architecture preview)** — 180B on disk / 6B active: Gated DeltaNet +
  a new sparse-attention mixer (QSA) + a 4-branch gated residual + a 20M-entry n-gram embedding +
  native MTP. Doesn't fit either rig. Scoped in
  [`scoping-qwen38-flash-next.md`](scoping-qwen38-flash-next.md); revisit when a real Qwen4
  checkpoint ships at a size that fits.
- **Mistral Medium 3.5 (128B dense)** — the five-minute config check this doc asked for on
  2026-08-17 has still not been done. Do that before writing anything else about it.
- **Named in the 2026-08-28 field survey, unscoped:** Muse Glimmer 30B, Qwen3.8-Max. Verify the
  weights are actually posted (not announced) before spending scoping time — the trap this doc
  has now recorded twice.
- **Mamba-1 hybrids (Jamba, Falcon Mamba, Hunyuan-TurboS)** — unchanged: confirm Mamba-1 vs
  Mamba-2 before estimating; Mamba-1 is a new scan kernel + parity, not a config alias. Only on
  concrete demand.
- **Gated Residual — a primitive bet, not a family bet.** The 4-branch gated-residual mixer
  scoped for Qwen3.8-Flash-Next (above) is now shared by **three** families: this one, DeepSeek
  V4-Flash, and Tencent Hy4-preview — `docs/audit-2026-09-10.md`'s **L-02**. Fully specified
  (`docs/scoping-qwen38-flash-next.md` §"What's genuinely new" item 2): ~3 days on CPU with a
  tiny golden, no kernel novelty. The risk is blast radius, not the math —
  `decoder/forwardn.go` is in the core hashed-file set, so the edit re-stales every family's
  `deps_hash`, the same cost `docs/completed/scoping-lfm2.md` §G paid for a much smaller change.
  Watching, not queued: no single family here is worth building for yet, but the primitive
  itself is a plumbing change one of the three could unlock.

## Explicitly out of scope (not "families")

- **Diffusion LMs** (LLaDA / Mercury / Gemini-Diffusion class) — a different decode loop, a
  second engine, and a compute-bound play against goinfer's lane. Revisit only as a separate
  initiative if ever.
- **Full multimodal early-fusion / vision-out / audio** — text-decoder plus vision-in stays the
  policy ([`multimodal.md`](multimodal.md)); the rest is its own track.
- **A sixth recurrent mixer (RWKV / RetNet / lightning-attention)** — gated-linear, state-space,
  latent-KV and KDA are covered; another is diminishing returns on the axis. Only on strong,
  specific demand.

## Selection rule

Popularity is the tiebreaker, not the lead, and "rides a shipped path" beats "prestigious but
new primitive." Two rules earned since the 2026-08-17 list, both from this doc's own items:

1. **Weights posted, not announced** — scope from a config file, never from a press release
   (DeepSeek V4, Kimi K3 and Ling 3.0 each moved on this).
2. **A whole forward must be runnable on hardware this project has** — kernel-level gates all
   green is not evidence for a family (`G7`), and a family that cannot get one end-to-end run here
   is parked, not queued (`G8`).

Then: MLA-shaped and DeltaNet/KDA-shaped MoEs on shipped substrate > dense fillers > Mamba-1
hybrids (gated on the variant cost) > out of scope. Every one goes through
[`parity-coverage-policy.md`](parity-coverage-policy.md)'s definition of done before it earns a
matrix row — especially the cheap ones.

<details>
<summary>Historical — the 2026-08-17 list and framing, as written (citations de-lined 2026-09-12 so archived reasoning stops drifting the lint)</summary>

Archived docs cite this block by section number ("Next up §1–§5"); the numbering is preserved.
Read it for the reasoning, not the status — the table at the top of this doc is the status.

### Next up — 2026-08-17 priority list (build these first)

Superseding the Tier A/B/C ordering below for near-term work. Read through
`docs/capability-matrix.md` first before picking one up — it's the source of truth for what's
actually landed, not this doc. Filtered for goinfer's niche: single-user, batch-1, consumer
hardware, text-only, parity-gateable against HF, weights in safetensors/GGUF. Queued as
`G7`/`G8` in `docs/queue-correctness.md`; `G4`-`G6` closed and archived to
`docs/completed/queue-correctness.md` on 2026-08-31.

1. **Nemotron 3 Nano (30B-A3B) — best payoff per unit of work (`G4`).** Hybrid Mamba-Transformer
   (Nemotron-H shape) with the FFN layers replaced by sparse MoE: sigmoid-gated router, shared
   experts, squared-ReLU activation, no positional embeddings, RMSNorm, untied embeddings. Every
   ingredient already exists: the Mamba-2 single-op blocks (`decoder/forward_nemotron.go`,
   `nemotronhArchitecture` in `decoder/registry.go`), squared-ReLU + sigmoid/shared-expert
   routing (from `deepseekArchitecture`/`glm4moeArchitecture`, `decoder/registry.go`), and
   NoPE (`cohere2Architecture`, `decoder/registry.go`). Mostly adapter composition, not new
   kernels. 3.2B active fits the 8 GB expert-streaming path and the 16 GB Mac rig; 1M context lines
   up with the long-context work already measured to 32k. NVIDIA is shipping weights, training
   software, recipes, and most of the training data — real community traction, friendly license.
   (Nemotron Super 120B-A12B adds LatentMoE + MTP — skip until Nano lands and only revisit if
   demand shows up.)

2. **Qwen3-Next / Qwen3-Coder-Next (80B-A3B) — closes the gap in goinfer's strongest family (`G5`).**
   `qwen35Architecture` (`decoder/registry.go`, forward pass `decoder/forward_qwen35.go`) already
   implements Gated DeltaNet + softmax + MoE; Qwen3-Next is that architecture's direct ancestor —
   likely a config-mapping delta, not a new forward path (verify before estimating further, same
   discipline the DeepSeek-V4 scoping below learned the hard way). Qwen3-Coder-Next is the
   recommended agentic coding model for 64GB-class systems and shows up on most "best local coder
   2026" lists — coding + structured output is goinfer's positioning (same reason Mellum2 was
   added). Apache 2.0, GGUF everywhere. **Caveat:** 80B total won't fit the M1 Pro 16 GB rig even at
   int4 — this is a WebGPU/CUDA-streaming showcase, not a laptop one; scope the residency story
   accordingly.

3. **Laguna XS 2.1 (33B-A3B) and Laguna S 2.1 (118B-A8.5B) — new family, strategic tie-in with P10
   (`G6`).** Poolside's MoE, open weights, landed hard in the agentic-coding niche (Terminal-Bench
   2.1 thinking-mode 70.2%, ahead of open models several times its size). Architecturally
   softmax-GQA territory: mixed SWA/global attention in a 3:1 ratio (the interleave pattern already
   handled for Gemma/Mellum2/gpt-oss) — softplus attention gating and per-layer RoPE scales are the
   genuinely new primitives. Two reasons it ranks this high: (a) official GGUF conversions ship
   alongside **official DFlash draft models** — P10 (now owned by `docs/spec/08-dspark-dflash.md`; the performance queue closed 2026-08-31) just queued
   DSpark/DFlash block drafters, and Laguna would be the flagship demo with vendor-blessed
   drafters instead of self-trained ones; (b) XS 2.1 (33B/3B active) is the consumer-hardware entry
   point, S 2.1 (118B/8.5B active) targets the 96–128 GB Mac crowd, so one family covers two
   hardware tiers.

4. **gpt-oss residency upgrade — not a new family, higher user impact than #3 above might suggest
   (`G7`).** Already in (`docs/capability-matrix.md`: `gpt_oss`, real-oracle parity), but
   **GGUF-only, CPU-only** (`decoder/registry.go`'s own comment: "MXFP4 experts; CPU-only").
   gpt-oss-20b remains one of the most-run local models in 2026 guides. The MXFP4 *reader* already
   exists (`decoder/mxfp4.go`, GGML quant type 39, verified bit-for-bit against the reference
   `gguf` Python library) — it was built for the GGUF path. The upgrade is two pieces: (a) a
   safetensors loader for gpt-oss's native MXFP4-packed weights (HF ships these directly, not just
   GGUF), and (b) Metal/CUDA GPU residency, which nothing in `gpt_oss`'s current path has. Worth
   weighing against #1-3 above precisely because it's an upgrade to an already-popular family, not
   a new one — likely to move more real users than a new exotic family would.

5. **DeepSeek V4-Flash (repo `DeepSeek-V4-Flash`, 0.16 TB) — strategic, not urgent (`G8`).**
   **Scoping already done and refuted the easy path** — see
   `docs/completed/task-model-family-deepseek-v4-kimi-k3.md`'s Phase 0 verdict: V4 is
   **not** a `deepseekArchitecture` alias. Eight new primitives (DSA sparse attention over a
   learned Indexer, strided KV compression, sliding-window + attention sink, grouped low-rank
   output projection, hash routing on early layers, `sqrtsoftplus` router scoring,
   hyper-connections, clamped SwiGLU) — a new architecture that happens to share q-LoRA and a
   shared expert with V3, not a config delta. **Concrete blocker in front of the primitive work:**
   V4-Flash ships **fp8 e4m3 blockwise-quantized weights** (`weight_block_size: [128,128]`,
   `scale_fmt: "ue8m0"`) and **there is no fp8 support anywhere in the tree today** — reading the
   weights at all is a prerequisite to streaming them, and that wasn't in the original estimate
   either. MIT license, DeepSeek's brand pulls the whole local community, and native sparse
   attention is where the field is converging (V3.2, GLM-5.1, and V4 have all moved to it) —
   building the DSA/compressor path once plausibly buys the next several Chinese frontier releases.
   File as **post-1.0 architecture work**, not a near-term ship: it needs fp8 support built first,
   which is its own prerequisite task, not a subtask of this one.

#### Watching (don't build yet)

- **GLM-5.x** — the ~753B flagship is out of scope; wait for an Air-class size. Shares the
  native-sparse-attention direction with DeepSeek V4-Flash above (#5) — if V4-Flash's DSA path ever
  gets built, re-evaluate GLM-5.x against it rather than from scratch.
- **MiniMax M3** — 428B MoE, ~22B active, GQA backbone with MiniMax Sparse Attention layered on
  top, natively multimodal. Too big for the niche, and the multimodal training makes text-only
  support less compelling than the size alone would suggest.
- **Kimi Linear / Ling (KDA)** — KDA would be a natural fifth sequence-mixing family next to
  DeltaNet, but Ling 3.0's weights were announced Apache 2.0 without actually being posted as of
  late July 2026 — **verify weight availability before scoping**, same lesson the DeepSeek-V4/K3
  doc already learned about assuming a shape from an announcement rather than a config.
- **Mistral Medium 3.5 (128B dense)** — if it's a plain Mistral-shaped dense stack it may already
  load or need only a config alias. A five-minute config check earns this a real queue entry;
  don't scope further until that check is done.
- **Watch — Kimi K3** (reportedly ~3-4T params, "the next major architecture jump"). *Not* a Kimi
  K2.x point release — K2 through K2.7-Code are the same DeepSeek-V3 arch and already supported via
  `kimi_k2` (no work; shared-path parity in `docs/capability-matrix.md`). K3 itself is
  **PARTIAL per the same Phase-0 scoping doc as V4-Flash**: MLA is genuinely ours, but it runs on
  only 24 of 93 layers — the other 69 are a KDA (Kimi Delta Attention) mixer, closer to
  `qwen3_5_moe`'s DeltaNet family than to anything MLA-shaped, plus a latent-MoE wrapper, a new
  activation, and MLA-with-NoPE + output gate. Also blocked on unread weights today: K3 ships
  `mxfp4-pack-quantized` in a `compressed-tensors` layout that has not been confirmed to match
  `decoder/mxfp4.go`'s (gpt-oss-derived) packing. Re-check the arch when weights actually drop;
  default assumption is DeepSeek-V3-shaped (alias-first) only until proven otherwise.

- **Qwen3.8-Flash-Next (Qwen4 architecture preview)** — 180B on disk / 6B active, explicitly
  previews Qwen4's shape: Gated DeltaNet + a new sparse-attention mixer (QSA) + a 4-branch gated
  residual stream + a 20M-entry n-gram embedding table + a native MTP head. The checkpoint itself
  doesn't fit either rig — bigger than `Qwen3-Next-80B-A3B`, which already can't get a full
  reference forward on the 62GB box (G5). Scoped in
  [`docs/scoping-qwen38-flash-next.md`](scoping-qwen38-flash-next.md): the GDN and
  MoE-expert-paging pieces ride substrate already shipped and parity-gated; QSA, the gated
  residual, and the n-gram embed are the real new work, sized there for a synthetic-tiny bring-up
  rather than the 180B itself. Revisit when a real Qwen4 checkpoint ships at a size that fits.

### The framing: families are no longer the high-leverage work

*(Historical — the two action items below predate this doc's 2026-08-17 update and are believed
resolved: GLM/Granite/Nemotron-H/DeepSeek-V2/V3/gpt-oss/Llama-4/Qwen3.5 all show current
`parity_manifest.json` rows in `docs/capability-matrix.md` today, and downstream docs reference a
landed v1.0 — **that second claim was wrong: no v1.0 tag exists; the latest is v0.17.2 as of
2026-09-12, and item 2 below is still open.** Left in place rather than deleted so the reasoning stays visible — verify before
citing either as still-open.)*

With the axes covered, the marginal family is a descriptor + loader + golden on an
*existing* path. That's cheap, but it's no longer where the leverage is. Two things
matter more than a sixth/seventh family and should land **before** this backlog:

1. **Validate what just shipped.** GLM, Granite, Nemotron-H, DeepSeek-V2/V3 arrived
   fast. Each needs its `parity_manifest.json` row + T1 golden + capability-matrix
   entry (`task-parity-coverage.md`, `task-capability-matrix.md`), or the support
   claims are unbacked and green CI is hiding asset-skips. Backfilling those five is
   worth more than any new family.
2. **Cut v1.0.** The roadmap's v1.0 trigger was "a second hybrid family survives the
   hybrid-cache abstraction." We now have *two* Mamba-2 hybrids **and** MLA — the
   abstraction absorbed far more than the bar. The evidence is overwhelming; freeze
   v1.0 and ship the capability matrix. **This backlog is explicitly post-v1.0.**

### The backlog (ordered by popularity-per-effort, all post-v1.0)

#### Tier A — ride a just-built path for nearly free

DeepSeek-V4 and Kimi-K3 (formerly here) moved to the "Next up" priority list at the top of this
doc (`G8` for V4-Flash; Kimi K3 stays a watch item there) once real scoping work existed for them.
The generalized lesson from that scoping is worth keeping visible regardless of where the specific
items live:

- **"De facto standard" MoE shapes are diverging, not converging — verify every time.** Both 2026
  DeepSeek-family flagships moved off the V3 shape within one generation (V4 replaced the KV
  latent outright; K3 demoted MLA to a quarter of its layers). "Alias-first" is still the right
  *starting hypothesis* — cheap, and sometimes right (Kimi-K2 was) — but it's a hypothesis to
  config-verify every time, never an estimate to fund. The `scoring_func`-keying gotcha
  generalized in an unexpected direction too: K3 renames the key entirely
  (`moe_router_activation_func`), V4 introduces a third value.

#### Tier B — cheap dense/MoE on existing rails (popularity filler)

- **Phi (Phi-4 / Phi-MoE)** — dense + MoE, no new primitive; popular for the
  on-device / efficient-deployment crowd. Pure descriptor + golden.
- **Newer Qwen dense/MoE** — the Qwen ecosystem keeps shipping; new dense or
  standard-MoE Qwen variants are descriptor-close to what we already run. (Qwen
  *hybrid* variants would ride DeltaNet/Mamba — also cheap.)
- **Yi / InternLM / other Llama-shaped families** — mostly `llama`-shaped; add only
  on concrete demand.

#### Tier C — more Mamba hybrids, with a real caveat

These compound the Granite/Nemotron machinery (hybrid cache + per-layer-kind + MoE),
**but most use Mamba-1, not Mamba-2**, so they are *not* automatically free:

- **Jamba** (AI21, Apache-2.0, Mamba + attention + MoE, 256K ctx) and **Falcon
  Mamba** (pure SSM) — both **Mamba-1**. Riding them needs a **Mamba-1 scan variant**
  alongside `mamba2.go` (a small new primitive, not a descriptor add). Worth it only
  if a Mamba-1 family is actually in demand.
- **Hunyuan-TurboS** and other Mamba-Transformer hybrids — check the Mamba version
  first; Mamba-2 ones are near-free (Nemotron-class), Mamba-1 ones carry the variant
  cost above.

Decision rule for Tier C: **confirm Mamba-1 vs Mamba-2 before estimating** — it's
the difference between "alias + config" and "a new scan kernel + parity."

### Explicitly out of scope (not "families")

- **Diffusion LMs** (LLaDA/Mercury/Gemini-Diffusion class) — a different *decode
  loop* (parallel iterative refinement, bidirectional attention), not a registry
  adapter. It'd be a second decode engine and a GPU/compute-bound play — squarely
  against goinfer's CPU single-binary lane. Not a family; revisit only as a separate
  initiative if ever.
- **Full multimodal early-fusion / vision-out / audio** — the text-decoder-only call
  (drop the vision tower) stays the policy; full multimodal is its own track.
- **A third linear primitive (RWKV / RetNet / lightning-attention)** — we already
  cover gated-linear + state-space; a third recurrent mixer is diminishing returns on
  the axis. Only on strong, specific community demand.

### Selection rule going forward

Popularity is now the tiebreaker, not the lead — and "rides a shipped path" beats
"prestigious but new primitive." Concretely: **MLA-shaped MoEs (Tier A) > dense
fillers (Tier B) > Mamba-1 hybrids (Tier C, gated on the variant cost) >
out-of-scope.** And every one of them goes through the `parity-coverage-policy.md`
definition of done before it earns a README support row — no exceptions, especially
for the cheap ones.

</details>

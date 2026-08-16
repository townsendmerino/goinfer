# 08 — DSpark / DFlash block drafters

> Status: **proposal — not built** (cut 2026-08-15, queued as **P10** in
> [queue-performance](../queue-performance.md)). This is the α lever
> [05](./05-eagle3-head.md)/[06](./06-acceptance-analysis.md) named, arriving pretrained:
> block drafters whose draft cost is per-*block* instead of per-token, with published
> acceptance far above EAGLE's measured ~1.6 tok/verify. Depends on
> [00-core](./00-core.md); composes with [03](./03-router-tree.md); its verify shape is
> exactly the short linear draft [07](./07-stageb-gemm-verify.md)'s large-dim re-measure
> asked for. Numbers quoted below from the upstream projects are **their claims on their
> hardware**, not ours — reproducing them on our harness is increment 0 of nothing and
> gate for everything.

## Provenance

Surfaced via `ARahim3/mlx-dspark` (an MLX port measured on an M4 Pro; r/LocalLLaMA post
`1vokrcy`, 2026-08). **The upstream is `deepseek-ai/DeepSpec` (github, MIT)** — DeepSeek's own
training + inference implementation for DSpark, DFlash *and* Eagle3, which mlx-dspark's `NOTICE`
names as its source and whose README lists the HF checkpoint repos as its released outputs
(paper: arXiv:2607.05147). Found during increment 2; it is the authoritative reference for the
parity gate, and it documents far more of the wiring than this page assumed. The methods are
DeepSeek's **DSpark** and z-lab's **DFlash**; both
publish trained drafter checkpoints on HF for targets goinfer already runs (Gemma-4,
Qwen3, Nemotron — confirmed pairs listed under increments; the post's headline model,
Qwen3.8-27B, implies newer pairs — re-check HF at spike time). We would import **weights
and method**, not code: the MLX/PyTorch implementations serve as the parity oracle, same
as AngelSlim did for 05.

## Idea

Both are lossless drafters under the standard verify — everything 00-core already
enforces (greedy bit-exact, sampled in-distribution) holds regardless of drafter quality.
What is new is the draft-side economics:

- **DSpark** (semi-autoregressive block drafter). A small parallel backbone (~5
  transformer layers) reads the target's hidden states — our `ForwardCapture` seam — and
  proposes a **7-token block** in essentially one pass; a **rank-256 Markov head** then
  adds cheap sequential token-to-token dependency across the block (correcting the
  suffix decay that pure parallel prediction suffers), and a **confidence head** scores
  each position so block length adapts per round (our [04](./04-adaptive-depth.md), but
  learned). Draft cost per round ≈ one pass of a 5-layer backbone + a rank-256 chain —
  versus EAGLE's head forward *per drafted token* plus a per-round correction forward,
  the exact overhead that 05's scorecard says ~1.6 accepted tokens couldn't pay back.
- **DFlash** (block-diffusion drafter). Denoises a **16-token block in one parallel
  pass**, and reuses the **target's own embedding and LM head** — the drafter ships only
  the denoiser trunk. Upstream reports acceptance length ~**6.0 on code/math** (up to
  ~2.1× end-to-end on the M4 Pro) and ~**0.98× on open chat** — a drafter that is
  strongest precisely on goinfer's marquee traffic (structured output, code, tool
  calls), where our own calibrated floor is α̂_grammar ≈ 0.20 (06).

Per the [README thesis](./README.md#thesis) — `speedup ≈ accepted/verify ÷ draft
cost/verify` — these attack **both terms at once**, which no source in the current
router does: n-gram (02) has a free draft but only fires on copy traffic; EAGLE (05) has
model-quality α on novel text but a draft cost that sank it; grammar (01) is the floor.

## Why goinfer is well-placed

- **The substrate is built and gated.** `Drafter` interface, batched verify
  (`forwardN` / `PrefillLastN`), `cache.TruncateTo` rollback, the 03 router with
  calibrated α̂ + online correction, and the hidden-state seam (`Model.ForwardCapture`,
  gated logits-byte-identical) that 05 already paid for. A new drafter is a new leaf,
  not new plumbing.
- **The verify amortization already exists where it matters.** On the resident CUDA
  path the batched M=k verify is measured **2.5–3.6× cheaper than k decodes at k=4–8**
  (`TestSpecVerifyCeiling`, D1) — and a 7- or 16-token **linear** block is exactly the
  short-linear-draft shape 07's large-dim re-measure said amortizes (M=4 → 1.58× at 70B
  dims), where trees were the wrong shape. What D1 lacks is a drafter with model-quality
  α on novel text; that is precisely what these are.
- **Same regime as the source numbers.** mlx-dspark's setting is single-user batch-1
  local decode on unified memory — goinfer's declared axis — so the published speedups
  are at least the right *kind* of number, unlike datacentre-batching results.
- **DFlash × `constrain` is a differentiated story.** The verify already applies the
  grammar mask, so constrained speculative decode stays lossless for free; and because
  DFlash reuses the target's LM head, the *same* mask can later be applied to the
  drafter's logits — the tokenization-aware forced drafter 01/06 said fundamentally
  needs the model, without training one.

## What it requires (the honest cost)

- **Two new drafter forwards.** DSpark: a 5-layer backbone (ops we have: attention +
  MLP + norms) + a rank-256 Markov head + a confidence head — closest to existing code.
  DFlash: **bidirectional (non-causal) attention over a masked block** — a genuinely new
  forward type for `decoder`, though bounded (one trunk, one pass, no KV politics).
- **Per-target trained checkpoints, instruct-matched.** These help specific presets,
  not every family; upstream reports base-vs-instruct mismatch craters acceptance
  (~47% vs ~82%). Availability is the same gating problem 05 named.
- **Protocol extraction, again.** As with EAGLE, the repos will not document every
  wiring choice (which layers are tapped, norm placement, mask schedule). The 05 lesson
  is priced in: a from-scratch forward that is "structurally correct" can still sit at
  α≈0.3 — so **reference-parity against the upstream implementation comes before any
  acceptance measurement**, not after a disappointing one.
- **Loader + fixture work.** Safetensors loaders for both drafter formats, converted
  fixtures, and a dumped-tensor CONFIRMED section in this doc once shapes are in hand
  (the 05 pattern).

## Licensing / IP note

Not yet checked — a **gate, not a footnote** (the 05 discipline). Verify the license on
the *specific* HF checkpoint repos (`deepseek-ai/dspark_*`, `z-lab/*DFlash*`) before
importing weights; reimplementing the method in Go is the clean path for the code side,
and we import no Python either way. Record the decision in the PR as 05 did.

## Where it can win — and where it will lose (pre-registered)

- **CPU: expect the 05 kill-gate to fire again, but measure once.** The verify node
  costs ~0.5 of a target step; even a *perfect* 7-token block tops out near ~2.2× (a
  ceiling mlx-dspark states for M-series and our 00-core economics reproduce). DSpark's
  per-block draft cost is the one term that changed since 05's CPU loss (~0.4×) — so one
  measured run documents whether it changes the verdict. Prediction: **no** on small
  targets; the entry is the GPU paths.
- **Resident CUDA (`linux`, 2070S 8 GB): Qwen3-4B int4/int8 + DSpark block7.** The D1
  verify amortization is already measured on this box. This is the cheapest end-to-end
  venue and the first gate.
- **Metal resident (`mac`, M1 Pro 16 GB): Gemma-4 12B int4 + either drafter.** The
  bigger target makes the draft relatively cheaper (06's "translates only when the
  target step is expensive enough"); 12B at 8-bit does not fit 16 GB with a drafter
  alongside, so int4 target + 4-bit drafter (~1.8 GB) is the configuration.
- **DFlash routes to constrained/tool traffic; DSpark to open text.** Both enter as
  router sources with trace-fit α̂ (06's runbook), not as replacements for n-gram —
  which stays the copy-traffic winner and costs nothing.

## Kill-gates (pre-registered, in order)

1. **Parity gate.** The Go drafter forward matches the upstream reference on dumped
   fixtures (logit cosine + argmax at drafted positions). Not cleared → do not measure
   acceptance; debug or stop. (This is the gate 05 cleared too late.)
2. **Acceptance gate.** Measured tok/verify on the 00-core suites (`chat` / `code` /
   constrained), trace-fit α̂ per 06. Bar: **≥3.0 tok/verify** on the paired target on
   at least one suite — anything near EAGLE's 1.6 means the protocol is wrong (back to
   gate 1) or the claims don't transfer (stop, record).
3. **Wall-clock gate.** End-to-end ≥**1.3×** vs plain resident decode (the 07 funding
   bar) on ≥1 real workload on ≥1 GPU backend, lossless gates green
   (`TestEagleSpecParity` shape). Miss broadly → park with numbers, like Stage B.
4. **Router gate.** With the drafter routed by α̂, the *mixed*-workload number must not
   regress vs n-gram-only — a drafter that wins its suite but mis-routes is a 03
   calibration item, not a ship.

## Build increments (planned)

1. **License + checkpoint audit — DONE 2026-08-15, not a kill, but a real flag.** Enumerate the HF
   pairs (confirmed at cut time: `deepseek-ai/dspark_gemma4_12b_block7`,
   `deepseek-ai/dspark_qwen3_4b_block7`, `z-lab/gemma4-12B-it-DFlash`,
   `z-lab/Qwen3-4B-DFlash-b16`), licenses, sizes. Kill here was free; the result doesn't kill it,
   but it isn't clean either:

   | repo | exists | license | size | notes |
   |---|---|---|---|---|
   | `deepseek-ai/dspark_gemma4_12b_block7` | yes | **NONE — no model card at all** | ~3.43B params, BF16, ~6.86GB | drafts for Gemma4; `Gemma4DSparkModel` |
   | `deepseek-ai/dspark_qwen3_4b_block7` | yes | **NONE — no model card at all** | ~1.39B params, BF16, ~2.79GB | drafts for Qwen3; `Qwen3DSparkModel` |
   | `z-lab/gemma4-12B-it-DFlash` | yes | **NONE — no model card at all** | ~0.73B params, BF16, ~5.82GB | drafts for Gemma4-12B-it; tagged `model_type: qwen3` (an internal arch label, not a Qwen target) |
   | `z-lab/Qwen3-4B-DFlash-b16` | yes | **MIT**, documented | ~0.54B params, BF16, ~1.08GB | explicitly "must be used in conjunction with the target model Qwen/Qwen3-4B"; needs `trust_remote_code=True`; cites arXiv:2602.06036 |

   **3 of the 4 confirmed pairs ship no license whatsoever** — not restrictive, not permissive,
   just absent (HF's bare "No model card" state). That is not automatically a blocker (DeepSeek and
   z-lab's OWN base-model releases are typically permissive, and the omission may just be a
   checkpoint-repo oversight), but it is not something to import and ship on the assumption it's
   fine either — **resolve before increment 2 touches these specific four weights**: either an
   explicit license appears upstream, or ask the authors, or treat it as blocking for those four
   and lean on the ones below that ARE licensed.

   **Re-checked HF for newer pairs, per the doc's own note.** The source post's headline model
   (Qwen3.8-27B) has **no official deepseek-ai or z-lab pair** — the only DSpark checkpoint
   targeting it is a **third-party community one**, `RadixArk/Qwen3.8-27B-DSpark` (license `other`,
   trained via SpecForge, not deepseek-ai/z-lab — do not treat as a confirmed pair). But deepseek-ai
   and z-lab both ship **more official pairs than the four originally listed**: deepseek-ai adds
   `dspark_qwen3_8b_block7`, `dspark_qwen3_14b_block7`, and full-size `DeepSeek-V4-{Flash,Pro}-DSpark`
   (~~**MIT**, documented~~ — **CORRECTED 2026-08-15: only the two `DeepSeek-V4-*` repos are MIT.
   `dspark_qwen3_{8b,14b}_block7` carry NO license and no model card, exactly like the 4b — this
   sentence's trailing "(MIT, documented)" read as covering the whole list and sent increment 2 at
   the 8b as a licensed checkpoint. The HF *list* endpoint reports `license=None` for every repo
   including the genuinely-MIT ones, so only per-repo detail queries separate them**); z-lab adds
   ~20 more DFlash targets (`Qwen3.6-27B-DFlash`,
   `Qwen3-8B-DFlash-b16`, `gpt-oss-120b-DFlash`, `Kimi-K2.6-DFlash`, `gemma-4-31B-it-DFlash`, among
   others — inconsistent naming, "gemma-4" vs "gemma4"). **No Nemotron pair exists from either
   deepseek-ai or z-lab** (confirms the doc's own honest gap) — but **NVIDIA ships its own
   first-party pair**, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-{DSpark,DFlash}` (license
   `other`), a genuinely new option this audit surfaced that the original scoping didn't know about.
2. **Protocol extraction + fixtures — PROTOCOL HALF DONE 2026-08-15 (`linux-62gb`); the
   checkpoint half is BLOCKED, see below.** Read the reference implementations
   (mlx-dspark being MLX, closest in spirit to our decode loop), dump per-position
   tensors for a short trace, convert one DSpark checkpoint to f32 safetensors, write
   the CONFIRMED-shapes section here.

   **The license gate increment 1 raised did not clear — it got worse on inspection, and it
   blocks exactly the half that needs weights.** Every `deepseek-ai/dspark_*_block7` drafter
   (4b, 8b, 14b, gemma4_12b) carries **no license and no model card** — three files each
   (`.gitattributes`, `config.json`, `model.safetensors`). The only MIT DSpark repos with a
   `LICENSE` are `DeepSeek-V4-{Flash,Pro}-DSpark` at **166.9 GB / 892.8 GB**, which are full-size
   V4 models rather than standalone drafters. Increment 1's own summary said the 8b/14b were
   MIT; that was an artifact of the HF *list* endpoint, which reports `license=None` for every
   repo including the two that really are MIT — corrected in `queue-performance.md`. **No weights
   were downloaded.**

   **What replaced the weights, and why it is stronger than a skim.** mlx-dspark's `NOTICE`
   names the upstream the design page did not know about: **`deepseek-ai/DeepSpec`
   (github, MIT)** — DeepSeek's own training + inference implementation, whose README lists
   these exact HF repos as its released checkpoints and whose `config/dspark/*.py` are the
   configs that produced them. So the protocol came from **first-party MIT source**, not from
   the paper and not only from a third-party port. See the CONFIRMED section below.
3. **DSpark forward + loader** (gate 1), then **Drafter wiring** into the existing
   verify on CPU — correctness first where debugging is cheap; take the one CPU
   wall-clock measurement while there.
4. **Resident CUDA end-to-end** on Qwen3-4B (gates 2–3). This is the go/no-go for the
   rest of the program.
5. **DFlash forward** (the new bidirectional block pass) + constrained-traffic
   measurement vs the 01/02 router baseline (gates 2–4).
6. **Metal 12B run** (`mac`), only if 4 clears.

## CONFIRMED architecture (`deepseek-ai/dspark_qwen3_8b_block7`, derived 2026-08-15)

> **Read the provenance line before the numbers.** 05's CONFIRMED section was *dumped from a
> checkpoint on this box*. This one is **derived from first-party MIT source and validated against
> the published checkpoint's exact byte count** — because the weights are unlicensed and were not
> downloaded (see increment 2). Every number below traces to one of:
>
> - **`deepseek-ai/DeepSpec`** (github, **MIT**) — `config/dspark/dspark_qwen3_8b.py` (the config
>   that produced this checkpoint), `deepspec/modeling/dspark/{common,markov_head}.py`,
>   `deepspec/modeling/dspark/qwen3/{config,modeling}.py`, `deepspec/eval/dspark/{draft_ops,
>   evaluator}.py`.
> - **`ARahim3/mlx-dspark`** (**MIT**) — the MLX inference port; read as the second source, and it
>   agrees with DeepSpec everywhere the two overlap.
> - **`Qwen/Qwen3-8B`** `config.json` (**Apache-2.0**) — the target dims the drafter inherits.
> - HF **blob metadata** for the drafter repo (file size only — no tensor data fetched).
>
> **What is NOT confirmed: any VALUE.** No per-position tensor dump, no f32 conversion, no logit
> cosine. That is the rest of increment 2 and it needs the license decision. Treat the shapes as
> load-bearing and the *forward* as reference-read; the parity gate (kill-gate 1) is still ahead.

**The byte-exact cross-check — why the derivation is trustworthy without the weights.** The
architecture below implies **2,371,081,729 parameters**, i.e. **4,742,163,458 bytes** at bf16.
The published `model.safetensors` is **4,742,170,330 bytes**. The **6,872-byte** difference is the
safetensors JSON header + its 8-byte length prefix, which is the right magnitude for the 64 tensors
listed below (~107 B/entry). A missing or mis-shaped tensor would move this by megabytes. So the
tensor inventory is complete and correctly shaped, established from published metadata alone.

**Config (DeepSpec `dspark_qwen3_8b.py`, verbatim):** `block_size=7`, `num_draft_layers=5`,
`target_layer_ids=[1, 9, 17, 25, 33]`, `mask_token_id=151669`, `num_anchors=512`,
`markov_rank=256`, `markov_head_type='vanilla'`, `confidence_head_alpha=1.0` (⇒
`enable_confidence_head=True`), `confidence_head_with_markov=True`.

`build_draft_config` **deep-copies the target config** and overrides only
`num_hidden_layers=5`, `tie_word_embeddings=False`, `architectures=["Qwen3DSparkModel"]`, plus the
DSpark fields — so every other dim is Qwen3-8B's: hidden 4096, heads 32/8 GQA, head_dim 128,
intermediate 12288, vocab 151936, `rms_norm_eps` 1e-6, `rope_theta` 1e6, no attention bias.

Tensors (`[out, in]`, 64 total, bf16):

| tensor | shape | note |
|---|---|---|
| `embed_tokens.weight` | [151936, 4096] | drafter's OWN copy, initialized from the target and **frozen** during training (`initialize_embeddings_and_head(freeze=True)`) |
| `fc.weight` | [4096, **20480**] | fuses the **5** captured target hidden states (5·4096) → 4096. 05's EAGLE-3 `fc` is the same idea at 3 layers |
| `hidden_norm.weight` | [4096] | RMSNorm on the fused target feature |
| `layers.{0..4}.self_attn.q_proj.weight` | [4096, 4096] | ×5 |
| `layers.{0..4}.self_attn.{k,v}_proj.weight` | [1024, 4096] | ×5 each, GQA 32/8 |
| `layers.{0..4}.self_attn.o_proj.weight` | [4096, 4096] | ×5 |
| `layers.{0..4}.self_attn.{q,k}_norm.weight` | [128] | ×5 each — Qwen3 per-head RMSNorm over head_dim |
| `layers.{0..4}.mlp.{gate,up}_proj.weight` | [12288, 4096] | ×5 each (SwiGLU) |
| `layers.{0..4}.mlp.down_proj.weight` | [4096, 12288] | ×5 |
| `layers.{0..4}.{input_layernorm,post_attention_layernorm}.weight` | [4096] | ×5 each — 2-norm (qwen/llama) layout, not gemma's 4-norm |
| `norm.weight` | [4096] | final, pre-head |
| `lm_head.weight` | [151936, 4096] | own copy, frozen; **full target vocab — no reduced draft vocab, so no `d2t`/`t2d`** (unlike 05's EAGLE-3 head) |
| `markov_w1.weight` | [151936, **256**] | rank-256 Markov: embedding indexed by the PREVIOUS token id |
| `markov_w2.weight` | [151936, 256] | projects the rank-256 latent back to full-vocab logit bias |
| `confidence_head.proj.{weight,bias}` | [1, **4352**], [1] | `AcceptRatePredictor` = one `Linear(→1)`; in_dim = hidden 4096 + markov_rank 256 because `confidence_head_with_markov=True` |

**Where the weight actually is:** embed + lm_head are **1.24 B of the 2.37 B params (52%)** and are
frozen copies of the target's. The 5 backbone layers are 0.96 B (193 M each); the Markov head is
77.8 M; the confidence head is 4,353 params. A loader that reuses the target's embed/head instead of
storing copies would halve the drafter's footprint — worth checking whether the two copies are
bit-identical to Qwen3-8B's own, since `freeze=True` says they must be.

### CONFIRMED forward (DeepSpec `eval/dspark/{evaluator,draft_ops}.py` + `mlx-dspark/model.py`)

Per drafting round, with the target's committed context already in the drafter's per-layer context
cache:

1. **Context update (once per accepted chunk).** `fused = hidden_norm(fc(concat(h[1], h[9], h[17],
   h[25], h[33])))` over the newly committed positions; each drafter layer projects it to K/V and
   appends to its own **`CtxCache`** — K is RoPE'd at its absolute position, **V is not**. This cache
   is append-only and is never rolled back (unlike the target KV cache).
2. **Block input.** `draft_input_ids = [last_committed_token, MASK ×6]` where MASK = `151669`, so
   the block is `block_size=7` wide.
3. **Backbone.** `noise_embedding = embed_tokens(draft_input_ids)` → 5 layers, each doing
   cross-attention with **Q from the block** and **K/V from `concat(ctx_cache, block)`**, then the
   standard Qwen3 2-norm layer → `norm`.
4. **The block attention is BIDIRECTIONAL — this is the load-bearing detail.** `draft_ops.py` calls
   the backbone with `is_causal=False`, and the training mask
   (`common.create_dspark_attention_mask`) allows `q_block_id == kv_block_id` with **no `q≥k`
   constraint** — every block position attends every other. Context keys are masked to
   `kv_idx < anchor_pos`. mlx-dspark independently confirms it (`mask=None`, "full bidirectional
   block"). **Consequence for us: the block width is not truncatable** — the trained distribution
   depends on all 7 positions, so "draft only 3" is not free (mlx-dspark's `draft_width` makes
   exactly this distinction and allows truncation only for causal DFlash-lineage heads).
5. **Logits.** `base_logits = lm_head(block_hidden[:, :7, :])` — `logits_start = 0`, so slot 0 both
   embeds the known token *and* predicts the first draft token. **All 7 positions are drafts**;
   a block7 head proposes 7 tokens/round and verifies 8.
6. **Markov correction — sequential, and not free.** For `i` in 0..6:
   `step_logits[i] = base_logits[i] + markov_w2(markov_w1[prev])`, `next = argmax/sample`,
   `prev = next`. So the block is parallel in the backbone and **serial in a 7-step scalar chain**.
   Each step is a [151936, 256] matvec ≈ **38.9 M MACs**, ×7 ≈ 272 M per round — small against a
   backbone pass but *not* the "cheap" the Idea section implies; it is ~0.3 of one 4096-dim layer's
   projection work, and it is latency-serial. Budget it explicitly on the CPU path.
7. **Confidence → adaptive block length.** `confidence_head(concat(block_hidden,
   markov_w1[prev_ids]))` → per-position logit → `sigmoid` → the proposal is truncated at the
   **first position below `confidence_threshold`** (`_confident_prefix_length`). This is 04's
   adaptive depth, learned and per-round, exactly as the Idea section claims.

### What this changes in the plan

- **The seam requirement is 5 layers, not 3, and it is config-driven per target.**
  `Model.ForwardCapture(id, cache, layers []int)` already takes an arbitrary layer list, so Qwen3
  needs no seam change. **But it hard-rejects architectures with their own `runLayers`** —
  including `gemma4` — so `dspark_gemma4_12b_block7`, the `mac` half of this program, needs the
  seam extended before it can run at all. That is a real increment-3 prerequisite the design page
  did not list.
- **Capture-point convention must be checked, not assumed.** DeepSpec indexes
  `hidden_states[layer_id + 1]` with `-1` meaning the embedding output, i.e. `target_layer_ids=[1,
  9, ...]` means *the output of layers 1, 9, …*. Confirm our seam's off-by-one against this before
  the parity gate — 05's lesson was that a structurally-correct forward can still sit at α≈0.3.
- **"The repos will not document every wiring choice" (the What-it-requires section) is now too
  pessimistic for DSpark.** DeepSpec documents the tapped layers, norm placement, mask schedule and
  block protocol outright. The remaining unknowns are values, not wiring — which is precisely what
  the blocked half would settle.

## Validation plan

- Correctness: lossless by construction (verifier owns `p`); output ≡ non-speculative,
  bit-exact greedy + sampled in-distribution, per backend — the standing 00-core bar.
- Acceptance: reproduce upstream acceptance on a matching model/workload before
  claiming transfer; publish α̂ tables per 06 with the same held-out discipline.
- Speed: 00-core harness suites on both boxes; every number lands in
  [experiments.md](./experiments.md) dated, with the losing runs kept.

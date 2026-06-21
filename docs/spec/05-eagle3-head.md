# 05 — EAGLE-3 feature head

> Status: **proposal**. Depends on [00-core](./00-core.md). Highest effort, but it is
> the general-purpose SOTA and the strongest moat. Tackle after the cheap wins
> ([01](./01-grammar-fused.md), [02](./02-cache-ngram.md), [04](./04-adaptive-depth.md)).

## Idea

The cheap drafters cover structure (01) and copied text (02) but not **novel
prose/reasoning**, where the target distribution is the model's own and nothing
external matches it. The proven way to draft there is a small **feature-level**
head that autoregresses over the target model's **hidden states** and predicts the
next token directly — EAGLE / EAGLE-2 / EAGLE-3. As of early 2026 EAGLE-3 is the
production standard (reported ~0.8+ acceptance, 3–4× throughput) and is merged into
vLLM, SGLang, and TensorRT-LLM.

By [00-core](./00-core.md) §1, the head wins by directly minimizing `TV(q, p)`: it is
*trained* to match the target, so it pushes acceptance up exactly in the region the
other sources cannot reach.

## Why goinfer is well-placed

- It **owns the full forward pass in pure Go**, so exposing the target's hidden
  states (the head's input) is a clean internal seam, not a hack — unlike wrapping a
  closed runtime.
- "Pure-Go EAGLE-3 in a single static binary" is a position no other maintained
  runtime occupies; it pairs with the [03 router](./03-router-tree.md) and tree
  verification already specified.
- The head is tiny (roughly one transformer layer) — feasible to run on the existing
  SIMD / WebGPU matmul paths without new heavy dependencies.

## What it requires (the honest cost)

- **A trained head per model family** (and often per checkpoint). This is the real
  expense — either train heads (needs data + a training pipeline, currently outside
  goinfer's pure-inference scope) or import/convert existing open EAGLE-3 heads where
  license permits. Decide import-vs-train early; see IP note below.
- A clean **hidden-state seam** in `decoder` to feed the head (read-only export of
  the residual/feature stream at the chosen layer(s)).
- Head **autoregression + tree drafting** (EAGLE-2/3 build a draft tree from the head;
  this is the natural input to [03](./03-router-tree.md)).
- **Numerics parity** held to goinfer's existing bar: the head is part of the draft
  path only (losslessness is still enforced by verification), but its outputs should
  be parity-checked against the reference head implementation to ensure acceptance
  matches the published numbers.

## Licensing / IP note

EAGLE's reference implementation (SafeAILab/EAGLE) and Medusa are released as open
source — last checked Apache-2.0, which carries an **express patent grant** for the
contributors' contribution; **verify the LICENSE on the specific repo/weights before
importing**. Reimplementing the *method* from the papers is the cleaner path for an
MIT project; importing *weights* pulls in their license. Either way this is the spoke
most worth a deliberate license decision — note it in the PR. (See the project-level
patent discussion; the foundational draft/verify loop has separate Google patent
activity independent of EAGLE's grant.)

## Risks / open questions

- **Per-family head availability** is the gating problem; without a head a family
  falls back to 01/02/plain decode (graceful, but no novel-text speedup).
- Hidden-state seam must not perturb the target forward numerics (read-only).
- Training pipeline, if we train, is a new capability area for the project — scope it
  separately or rely on imports initially.
- Acceptance is workload- and family-dependent; reproduce published numbers on our
  harness before claiming them.

## Feasibility verdict (2026-06-21) — IMPORT a general head; build is large but lossless-safe

Researched the import-vs-train decision (the doc's gating question). Decision: **import**,
since we cannot train.

**A usable head exists and the gates clear:**
- **`AngelSlim/Qwen3-4B_eagle3`** (and a size ladder: 1.7B / 4B / 8B / 14B). Single
  `model.safetensors` (437 MB ≈ 0.2 B params). `config.json`:
  `Eagle3LlamaForCausalLM`, `num_hidden_layers: 1`, `hidden_size: 2560`,
  `num_attention_heads: 32`, `num_key_value_heads: 8`, `head_dim: 128`,
  `intermediate_size: 9728`, silu, `rope_theta: 1e6`, **`draft_vocab_size: 32000`**
  (reduced draft vocab + a t2d/d2t map to the full 151936) — a single Qwen3/llama-style
  transformer layer using only ops goinfer already has (GQA, RoPE, SwiGLU, RMSNorm) +
  a feature-fusion linear + the draft-vocab head.
- **License OK:** Apache-2.0 / MIT / CC-BY-4.0 components — commercial use,
  redistribution, modification permitted; attribution required; no copyleft. Importable
  into goinfer with attribution.
- **Base is goinfer-runnable:** Qwen3 dense is already resident-eligible (Lever C1). The
  head is *general* (not code-trained), but a head drafts whatever the base emits, so it
  still accelerates code on a Qwen3 base. (Code-specific heads exist only for
  Qwen3-Coder-**Next** = Gated DeltaNet, which goinfer guards OUT of speculation, and in
  FP8 — not usable. No EAGLE head exists for qwen2.5-coder.)

**The reframe that de-risks it:** the head is a *Drafter*. The existing spec verify is
lossless regardless of draft quality, so an imperfect head **cannot break correctness** —
it only yields low acceptance (no speedup). So we can build incrementally and *measure*
acceptance, with no losslessness risk and no need for a bit-exact parity gate to ship
safely.

**The real risk = matching the protocol for good ACCEPTANCE, not correctness.** The
EAGLE-3 wiring that the model repo does NOT document — **which 3 target layers are fused**
(low/mid/high), the fusion detail, the head's autoregression, and the t2d/d2t mapping —
lives in the vLLM / SGLang / SpecForge source. Reproducing it wrong gives poor acceptance,
not wrong output. So the build extracts the protocol from that framework code and tunes to
maximize measured acceptance.

## Build increments (planned)

1. **Hidden-state seam** (DONE): `Model.ForwardCapture(id, cache, layers)` exports the
   residual stream after each listed layer (read-only; `TestForwardCaptureSeam` gates
   logits-byte-identical + last-layer capture reproduces logits exactly). Generic
   decode path; special-family archs return an error (not yet wired). `forwardN`
   (batched verify) capture is a follow-up for the full draft loop.
2. **Head loader** (DONE): `decoder.LoadEagleHead(dir)` → `EagleHead` (fc, midlayer
   q/k/v/o + gate/up/down, hidden/input/post-attn/final norms, draft `lm_head`, `d2t`),
   via the aikit safetensors reader. `TestLoadEagleHead` gates the confirmed shapes
   (fc 3*hidden, attn 2*hidden, lm_head draft-vocab, d2t in-range). Head converted
   from the AngelSlim `.bin` to f32 safetensors with `~/.venv-vl` (torch+safetensors).
3. **Head forward** (next): per the protocol below.

### EAGLE-3 protocol (extracted from vLLM `llama_eagle3.py` + SpecForge)

Submodules / tensors in the checkpoint:
- `embed_tokens` — the head's token embedding [vocab, hidden] (ships its own).
- `fc` — ReplicatedLinear, **in = target_hidden × num_aux_hidden_states (=3·hidden), out = hidden**: fuses the 3 concatenated target hidden states into one feature.
- `input_norm` — RMSNorm applied to the **concatenated 3-layer hidden** *before* `fc` (when `norm_before_fc`). (optional `fc_norm` per-aux RMSNorms in some variants.)
- `layers` — ONE `LlamaDecoderLayer` (self_attn q/k/v/o + mlp gate/up/down + input_layernorm + post_attention_layernorm). **Its attention projects from 2·hidden** (see forward).
- `norm` — final RMSNorm before the head.
- `lm_head` — [draft_vocab(32000), hidden].
- `draft_id_to_target_id` (d2t) — int map scattering draft-vocab logits into the 151936 target vocab.

Forward (drafting one token from the target's hidden at the current position):
1. Target forward exposes aux hidden states from **3 layers** (the inc-1 seam).
2. `h3 = concat(low, mid, high)` → `input_norm(h3)` → `feature = fc(h3)`  → [hidden].
3. `e = embed_tokens(prev_token)` → [hidden].
4. **`layer_in = concat(e, feature)`** → [2·hidden] → the decoder layer (so its q/k/v_proj have `in_features = 2·hidden`) → `norm` → `lm_head` → draft logits [32000].
5. Map to target vocab: `target_logits[d2t[i]] = draft_logits[i]`; argmax/sample → a target token id.
6. Autoregress K steps: the head keeps its OWN KV cache; each step feeds the just-drafted token's embed concatenated with the SAME `feature` (the target hidden is fixed for this draft block) — confirm vs source in inc 4.

OPEN: which 3 target layer indices feed step 1 (a training choice; vLLM reads
`eagle_aux_hidden_state_layer_ids` or defaults). Confirm from the checkpoint/config
or tune by measured acceptance (inc 5). Reuses goinfer's GQA/RoPE/SwiGLU/RMSNorm.

### CONFIRMED tensor structure (AngelSlim/Qwen3-1.7B_eagle3, dumped 2026-06-21)

Test target: local `qwen3-1.7b-q8_0.gguf` (hidden 2048, 28 layers, heads 16/8, hd 128).
Head shipped as `pytorch_model.bin` → converted to f32 safetensors at
`~/models/qwen3-1.7b-eagle3/model.safetensors` (via `~/.venv-vl` torch+safetensors).
Tensors (`[out,in]`):
- `fc.weight` [2048, **6144**] — fuse 3 concatenated target hidden (3·2048) → 2048.
- `midlayer.self_attn.q_proj.weight` [2048, **4096**], `k_proj`/`v_proj` [1024, 4096],
  `o_proj` [2048, 2048] — GQA 16/8 over `concat(embed, feature)` (2·2048=4096 in).
- `midlayer.mlp.{gate,up}_proj` [6144, 2048], `down_proj` [2048, 6144] (SwiGLU).
- `midlayer.input_layernorm` [2048] (norms the **embed**), `midlayer.hidden_norm`
  [2048] (norms the **feature**), `midlayer.post_attention_layernorm` [2048] (pre-MLP),
  `norm` [2048] (final, pre-head).
- `lm_head.weight` [**32000**, 2048] (draft vocab). `d2t` (32000, int64), `t2d`
  (151936, bool). **No `embed_tokens`** → reuse the TARGET's embedding (hidden matches).

Refined forward: `feature = fc(concat(h_lo,h_mid,h_hi))`;
`x = concat(input_layernorm(target_embed(tok)), hidden_norm(feature))` →
midlayer (attn in=4096, residual over the 2048 feature) → `norm` → `lm_head` →
draft logits[32000] → `target_id = draft_id + d2t[draft_id]` (confirm delta-vs-direct
in inc 2). Weights local; no further download needed.
4. **Autoregressive drafting + Drafter integration**: the head generates K tokens from
   the target's last hidden state (via the seam) and its own outputs; wire as a `Drafter`
   into the spec verify (it composes with the 03 router).
5. **Measure acceptance** on `chat`/`reasoning`/`code`; tune the fused-layer indices to
   match the framework. Extract exact protocol from vLLM/SGLang EAGLE-3 source.

### Build status (2026-06-21) — full pipeline BUILT + WORKING at ~1.6 tok/verify

Increments 1–4 done on the real AngelSlim/Qwen3-1.7B head + a local Qwen3-1.7B base:
- **inc 1 seam** ✅ `Model.ForwardCapture` (gated).
- **inc 2 loader** ✅ `LoadEagleHead` (gated; head `.bin`→f32 safetensors).
- **inc 3 forward** ✅ `Fuse`/`Step`/`attend` — validated structurally correct
  (~41% head/base single-token agreement at capture layers {2, L/2, L-3} vs ~0% wrong).
- **inc 4 autoregressive draft** ✅ `Prefill` (head KV over the context — the piece that
  unlocked multi-token) + `DraftFrom`. Realized **accepted length 0.64 → ~1.64
  tok/verify** (greedy, K=6), histogram [17,23,2,…].

**Findings on the acceptance gap (vs published ~3–4×):**
- NOT base precision: a bf16 base gave 0.64 vs the q8 base's 0.62 — no change.
- NOT capture layers: agreement plateaus at 38–41% across all reasonable low/mid/high
  triples; {2,14,25}/{3,14,25} are the best.
- The gap is **method + metric**: EAGLE's headline numbers use **tree drafting**
  (multiple head candidates per position) + **sampled rejection acceptance** (1−TV),
  whereas this is a single linear chain measured by **greedy top-1 match** (the
  strictest case). ~40% greedy top-1 → ~1.6 tok/verify linear is consistent with that.

**inc 5 DONE — end-to-end lossless EAGLE spec decode.** `GenerateEagleSpeculative`
(+ `forwardN` per-position hidden-capture, the batched seam) wires the head as the
drafter: head drafts K → target verifies in one batched pass capturing each
position's hidden → matching prefix + the target's correction committed → head
re-seeds from the verified hidden, its KV rebuilt over the confirmed context.
`TestEagleSpecParity`: token-identical to plain greedy, **1.60 tok/verify**. The full
pure-Go EAGLE-3 pipeline runs lossless. The acceptance is modest (greedy single
chain, ~40% top-1 head match + an extra correction-forward per round); the path to
the published ~3–4× is EAGLE **tree drafting** (multiple head candidates per
position, composes with [03](./03-router-tree.md)) + **sampled** rejection
acceptance — future work, not a correctness gap.

## Validation plan

- Correctness: lossless by construction (verifier owns `p`); output ≡ baseline.
- Acceptance parity: reproduce reference EAGLE-3 acceptance on a matching
  model/workload before integrating, to confirm the head/seam is faithful.
- Speed/acceptance: `chat` / `reasoning` suites in the [00-core](./00-core.md)
  harness, plus combined runs with [03](./03-router-tree.md) (head + grammar + n-gram
  in one tree) to measure the full-stack number.

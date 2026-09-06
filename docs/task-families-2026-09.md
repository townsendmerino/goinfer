# Task: four families, 2026-09 (F1–F4)

> **Status: DONE, 2026-09-06.** All four items landed in order (F1 → F2 → F3 → F4), each committed
> and pushed independently. Summary below; read each section's own status line for the detail.
>
> - **F1 `qwen3_moe`** — new registry key, T1 tiny-golden (cosine 0.9999999999999462, argmax
>   exact), GGUF support, resident-admitted cuda/metal/webgpu. T3 real-checkpoint owed (Linux box).
> - **F2 Nemotron 3.5 Lightning** — confirmed NOT a new family: config-identical to the already-T3'd
>   Nemotron 3 Nano in every architecturally meaningful field. No registry change; a real-checkpoint
>   gate + pin script + asset were added, execution owed (Linux box, bf16 ~60GB).
> - **F3 `granite`** (dense) — new registry key, reuses `llamaTensorSchema` verbatim. Of Granite's
>   four scalar multipliers: `embedding_multiplier` → `EmbedScale`, `attention_multiplier` →
>   `AttnScale`, `logits_scaling` → `LogitScale` (all three already-generic fields, all exercised
>   non-trivially by the T1 fixture below); `residual_multiplier` has no generic hook and is
>   HARD-REJECTED unless 1.0 (the identity value every released 4.2 size ships — see the F3 section
>   below for the check). T1 tiny-golden with the three non-trivial multipliers (cosine
>   0.999999999999987, argmax exact), GGUF support, resident-admitted cuda/metal/webgpu. T3
>   real-checkpoint owed (this Mac or Linux box; 8B is the sized target).
> - **F4 Ling-3.0-tiny (`bailing_hybrid`)** — Phase 0 only, by design: no stop condition fired, but
>   the brief scoped this item to a synthetic rehearsal, not a shipped family. MLA + MoE are pure
>   `deepseekArchitecture` composition; KDA (Kimi Delta Attention) is a genuinely new primitive — a
>   per-channel-gated delta rule, one row of the existing Gated DeltaNet's decay generalized from a
>   scalar to a vector — proven against `fla-org/flash-linear-attention`'s real reference at cosine
>   1.00000000 (maxAbsDiff 2.98e-08, float32 rounding). No registry key, no CHANGELOG entry. Filed
>   explicitly as the KDA rehearsal for Kimi K3's own eventual bring-up.
> - **Bench rows**: none this pass for any of the four — every family's T3 (the gate a peer-bench
>   row is conditioned on, per this doc's own rule for F1/F2/F3) is owed, and F4 shipped no loadable
>   family at all. Zero peer-bench claims were made, which is the correct state given that.

## F1 · qwen3_moe (Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-Instruct) — DONE at T1 (mac, 2026-09-05/06)

**The hole this closes:** `decoder/registry.go` had `qwen3` (dense), `qwen2_moe`, `qwen3_5_moe` and
`qwen3_next`, but no `qwen3_moe` — Qwen3-30B-A3B and Qwen3-Coder-30B-A3B-Instruct, both released
under that exact `model_type`, had no adapter at all.

### Phase 0 (config-verified against the real releases, not a model card)

Fetched both real `config.json` files directly from the hub:

- **`Qwen/Qwen3-30B-A3B`**: `architectures: ["Qwen3MoeForCausalLM"]`, `model_type: "qwen3_moe"`,
  `head_dim: 128`, `hidden_size: 2048`, `num_attention_heads: 32`, `num_key_value_heads: 4`,
  `num_hidden_layers: 48`, `num_experts: 128`, `num_experts_per_tok: 8`,
  `moe_intermediate_size: 768`, `norm_topk_prob: true`, `decoder_sparse_step: 1`,
  `mlp_only_layers: []`, `rope_theta: 1000000.0`, `attention_bias: false`, `tie_word_embeddings:
  false`. **No `shared_expert_intermediate_size` field at all.**
- **`Qwen/Qwen3-Coder-30B-A3B-Instruct`**: byte-identical on every field above except
  `max_position_embeddings` (262144 vs 40960) and `rope_theta` (1e7 vs 1e6). Confirms the scoping
  assumption directly: **the Coder variant is the same registry key**, a config delta only.

This is exactly the hypothesized composition: qwen3 dense's attention (per-head QK-norm, GQA, no
q/k/v bias, single-base RoPE) with `qwen2_moe`'s router shape (top-k of `num_experts` at
`moe_intermediate_size`, `norm_topk_prob`) — **minus the shared expert**, confirmed by the absent
config field rather than assumed from the "no shared expert" phrasing in the original scoping note.

### GGUF — header and tensor names/dims verified against a real file; VALUES are not (T3 item)

Fetched the first 30 MB of `unsloth/Qwen3-30B-A3B-GGUF`'s `Q2_K.gguf` via HTTP Range (well under
the ~12 GB full file — GGUF metadata + tensor-info lives at the front) and parsed it with goinfer's
own `aikit/embed.GGUFFile` reader, the same ground-truth method the Nemotron 3 Nano T3 handoff used
— but be precise about what that method proves here: it read `general.architecture`, the metadata
keys, and every tensor's NAME and DIMS. It did **not** dequantize a single tensor or run a forward
pass, so it cannot rule out the exact failure mode `docs/completed/queue-correctness.md` G4 already
named for this shape of loader: "a fused expert stack read with the wrong stride gives finite
values, correct shapes and confident nonsense." That check — hand-dequantizing a real
`ffn_*_exps` tensor's expert slices, or a real forward — is not done and belongs to T3, not this
pass.

What WAS confirmed this way:

- `general.architecture` is literally `"qwen3moe"`.
- Metadata is the plain, un-exotic `{arch}.attention.head_count/head_count_kv/key_length/
  layer_norm_rms_epsilon`, `{arch}.expert_count/expert_used_count/expert_feed_forward_length`,
  `{arch}.rope.freq_base` — **no sliding-window or YaRN keys at all** (unlike Mellum's per-layer-type
  split; qwen3_moe is single global RoPE).
- Tensor set and dims: `blk.N.attn_{q,k,v,output}`, `attn_{q,k}_norm`, `ffn_norm`, `ffn_gate_inp`,
  `ffn_{gate,up,down}_exps` (dims `[768 2048 128]` / `[2048 768 128]`, matching
  `moe_intermediate_size`/`hidden_size`/`expert_count`) — **no `ffn_*_shexp` tensors**, confirming
  the no-shared-expert finding independently on the GGUF side, at the name/shape level.

Because both findings are "this is exactly the generic shape, just missing the shared-expert
piece," **zero new code was needed in the generic GGUF `loadLayer` path** (`decoder/gguf.go`) — it
already gates QK-norm on `arch.QKNorm`, the router+experts on `arch.MoE != nil`, and the shared
expert on `arch.MoE.SharedIntermediateDim > 0`. Only a per-family `Config` builder
(`ggufQwen3MoeConfig`) and a `case "qwen3moe":` dispatch line were needed. That reasoning is sound
independent of the T3 gap above — it follows from the generic loader's existing gating logic, not
from this file's tensor values — but the T3 gap is the honest place a stride/layout bug would
actually be caught, and isn't closed yet.

### Adapter decision: pure composition, no new primitive, no new forward path

- `decoder/registry.go`: `qwen3MoeArchitecture` — qwen3's `Architecture` literal (`QKNorm: true`,
  `AttnScale: 1/√head_dim`, single-base RoPE, `NormPlacement: NormPre2`) with a `MoEConfig` shaped
  like `qwen2_moe`'s but with `SharedIntermediateDim` left at its zero value.
- `decoder/config.go`: `validateQwen3Moe` — qwen3's dense checks plus MoE bounds, no
  `shared_expert_intermediate_size` requirement.
- `decoder/weights.go`: `qwen3MoeTensorSchema` — qwen3's attention tensor names + `qwen2_moe`'s
  router/expert names, with every `Shared*` field left empty.
- `decoder/gguf.go`: `ggufQwen3MoeConfig` + one dispatch line, per the GGUF section above.

**Because qwen3_moe is a standard uniform-layer family** (not added to `decoder/arch.go`'s
`ownForwards` table), every dispatch surface named in the task brief answers correctly **for
free**, the same way `qwen2_moe`/`mixtral`/`mellum` already do:

- `canBatchN`/`specRollbackSafe`/`hasRecurrentState` (`decoder/forwardn.go`) all derive from
  `ownForward()`, which qwen3_moe never enters — confirmed by inspection, not a new test. (One of
  `spec_adaptive.go`'s own comments literally uses "Qwen3-30B-A3B" as its worked example of a
  resident MoE with `specRollbackSafe` true — this family was already an implicit assumption in
  that code before it had an adapter.)
- `decoder/features.go`'s `archFeatureProfile`/`FeatMoEGatedShared` derivation is generic on
  `arch.MoE.SharedIntermediateDim > 0`, so the absent shared expert falls out correctly with no new
  branch.
- `canSerialize` (`decoder/serialize.go`) and the CUDA `nonBatchableKind` (`cuda/prefill.go`) are
  both keyed on architecture SHAPE (MoE/MLA/Mamba presence), not a per-family name list, so neither
  needed an entry.

### Chokepoints exercised directly (not just via the family's own validator)

Registering the family surfaced three registry-wide tests that fail until a new `model_type` is
classified everywhere required — exactly the "walk the registry" guard the brief asked for,
already built into this repo rather than new:

1. `TestResidentAdmission_registryCovered` / `archFeatureProfile` (`decoder/features_test.go`) —
   classified as `{FeatMoE, FeatQKNorm}`, the union of `qwen3`'s and `mixtral`'s profiles.
2. `TestResidentAdmission_matrix` / `admissionGolden` — hand-verified `{cuda, metal, webgpu}`: this
   is a strict SUBSET of what `qwen2_moe` (`{FeatMoE, FeatMoEGatedShared}`) and `qwen3`
   (`{FeatQKNorm}`) each already require and are already admitted on, on all three backends, so no
   new kernel work is implied by the admission.
3. `TestParityManifest_fresh` / `TestCapabilityMatrix_CoverageComplete` — a manifest row and a
   `representativeConfig`/`familyDoc` entry are both required before CI is green; both added.

`resolveArchitecture` → `validateResolved()` runs for `qwen3_moe` like every family (not a new
test — the existing generic chokepoint test already covers every registered `model_type`).

### T1 — tiny-golden, DONE

`scripts/pin_qwen3moe_tiny.py` (transformers 5.15.0, `Qwen3MoeForCausalLM`, 4 layers / 8 experts /
top-2 / head_dim 8): `TestQwen3Moe_forwardParity` — **argmax exact, cosine 0.9999999999999462.**
Exercises QK-norm composed with a no-shared-expert routed MoE together for the first time.

**One real, if minor, finding along the way:** transformers 5.15.0's
`Qwen3MoeConfig.save_pretrained` writes the field as `num_local_experts`, but the real released
`Qwen/Qwen3-30B-A3B` config.json (fetched directly, not from a save) uses `num_experts`. goinfer's
adapter reads `num_experts` to match the real release, so the pin script rewrites the freshly-saved
config's field name post-save — the same class of transformers-serialization-drift-vs-release
mismatch `docs/completed/queue-correctness.md`'s G4 (`layers_block_type` canonicalization) and G5
(fused `in_proj_qkvz`) already catalogued, caught here before it could cause a silent config-parse
failure on anyone else's freshly-saved fixture.

`parity_manifest.json` row: `status: experimental`, `method: tiny-golden`, `argmax_pct: 100.0`,
`cosine_min/mean: 1.0`. Added to `cmd/gate/parity.go`'s required `parityGates` list
(`{"qwen3moe", "TestQwen3Moe_forwardParity"}`) so a regression here fails the sweep like every
other family's tiny gate.

### What was deliberately not done

- **T3 real-checkpoint parity — infra built, not yet run.** Qwen3-30B-A3B bf16 is ~61 GB; the
  doc's own order-of-operations ("if F1's T3 cannot run, ship at T1 and move on") applied at first
  landing. `scripts/pin_qwen3moe_real.py` + `decoder/qwen3moe_real_test.go` (tag `realckpt`,
  `TestQwen3MoeReal_oracle`, `GOINFER_QWEN3MOE_HF` asset registered) now exist, mirroring
  `decoder/nemotron3nano_real_test.go`'s shape — starting quant `int8` (weights) / f32
  (activations), not `int8int8`, per Nano's own measured router-flip cliff at similar sparsity
  (G4). Added to both `parityRealckptGates` and `emitGates` (unlike F2's Lightning gate, this one
  has no prior real-checkpoint evidence in the manifest to protect, so a passing sweep may
  legitimately promote the row). Row stays `experimental` until the run actually happens.
- **One decode benchmark row** (the brief's "one cell each" bench requirement) is gated on T3 —
  measuring a resident quant path before the numerics are validated against a real oracle would be
  exactly the kind of ungated claim `docs/benchmarks.md`'s Methodology section exists to prevent.
- **GGUF real-file round-trip test — this is a T3 item, not a T1 gap.** The header, metadata, and
  tensor names/dims were verified against a real file (above); the tensor VALUES were not — no
  dequantization, no forward pass. A full tiny-GGUF fixture + parity gate (mirroring
  `TestGGUF_qwen3_parity`) is not built, and neither is the hand-dequant spot-check G4 used for
  Nemotron 3 Nano's fused expert tensors. The fused-expert stride/layout is exactly where a GGUF
  MoE loader goes wrong per that precedent, so this is named as an open correctness question, not
  a formality.
- **CUDA/Metal/WebGPU resident kernel work.** None is needed — see the admission section above —
  but no resident-parity test was run on real hardware for this family specifically; it rides the
  same generic MoE dispatch `mixtral`/`qwen2_moe` already exercise on real backends elsewhere.

## F2 · Nemotron 3.5 Lightning 30B-A3B — Phase 0 DONE: same family, not a new one; T3 code ready, run pending

**The question this section answers, per the brief's own framing ("probably a parity pin, not a
family"): is it?** Yes — confirmed, not assumed.

### Phase 0 (config-verified against both real releases, field-by-field, not from the brief's own transcription)

Fetched `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`'s real `config.json` directly (the
brief's own quoted fields checked out, but transcriptions get re-verified here the same as every
other family in this doc) and diffed it programmatically, field-by-field, against
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`'s real `config.json` — the checkpoint `nemotron_h`'s
existing T3 (`docs/completed/queue-correctness.md` G4) already validated.

**Every architecturally meaningful field is identical**, not merely similar: `hidden_size` 2688,
`num_attention_heads` 32, `num_key_value_heads` 2, `head_dim` 128, `mamba_num_heads` 64,
`mamba_head_dim` 64, `n_groups` 8, `ssm_state_size` 128, `n_routed_experts` 128,
`num_experts_per_tok` 6, `moe_shared_expert_intermediate_size` 3712, `routed_scaling_factor` 2.5,
`n_group`/`topk_group` both 1 (degenerate, plain top-k — same as Nano), `rope_theta` 10000,
`partial_rotary_factor` 1.0 (present in BOTH configs, and — checked directly against mainline
transformers' `modeling_nemotron_h.py`, not assumed from Nano's T3 alone — genuinely unused by the
reference forward; Nemotron-H's attention is NoPE in practice regardless of these fields, which is
why `nemotronhArchitecture`'s `Architecture` literal hardcodes `RotaryDim: 0` unconditionally).

**Every one of those matching values is READ FROM CONFIG, not hardcoded** — which is the actual
claim "no registry change" rests on, not just that Lightning happens to match Nano. Named against
the Go `Config` field each maps to, all already wired by the existing `nemotronhArchitecture` /
`MoEConfig` construction:

| config.json field | Go `Config` field | consumed as |
|---|---|---|
| `n_routed_experts` | `NRoutedExperts` | `MoEConfig.NumExperts` |
| `num_experts_per_tok` | `NumExpertsPerTok` | `MoEConfig.TopK` |
| `routed_scaling_factor` | `RoutedScalingFactor` | `MoEConfig.RoutedScale` |
| `n_groups` (mamba) | `NGroups` | `nemotronParams.NGroups` (Mamba-2 group count — a DIFFERENT
  field from the MoE router's own `n_group`/`topk_group`, which both configs set to 1) |
| `moe_shared_expert_intermediate_size` | `MoeSharedExpertIntermediateSize` | `MoEConfig.SharedIntermediateDim` |

None of these is a value the adapter assumes or defaults to — if Lightning's release had shipped
different numbers here, the loader would have read and used THOSE, not Nano's. The identity is a
fact about the two checkpoints, not about what the code does with whatever it's given.

**Decoded and diffed the actual layer PATTERN, not just its summary counts.** Nano ships the
compact `hybrid_override_pattern` string (`"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"`,
52 chars); Lightning ships the expanded `layers_block_type` array directly. Decoded Nano's string
with the same M/E/*/− alphabet `decoder/config.go`'s `normalizeNemotronBlocks` uses and compared
position-by-position against Lightning's array: **byte-identical sequence**, 23 mamba / 23 moe / 6
attention, same order. `normalizeNemotronBlocks` already no-ops when `layers_block_type` is
present (its very first line), so Lightning's array-shaped release needs no loader change either.

**The only field-level differences are vestigial or presentational, each checked against source
rather than assumed inert:**

- `moe_latent_size: null` on both — checked against `modeling_nemotron_h.py` directly: non-null
  would add an `fc1_latent_proj` bottleneck before the expert FFN; null (both checkpoints) means
  that branch never executes. No LatentMoE on either family member.
- `moe_shared_expert_overlap: true` (Lightning only) — grepped mainline transformers'
  `modeling_nemotron_h.py`: **zero occurrences**. Not read by the HF reference forward at all — an
  inference-engine (vLLM/TensorRT-LLM-class) scheduling hint to overlap the shared-expert compute
  with routed-expert dispatch, with no effect on the computed result.
- `num_nextn_predict_layers: 1` / `mtp_layers_block_type` (Lightning only) — also zero occurrences
  in `modeling_nemotron_h.py`; the MTP head isn't implemented in the mainline forward class at all.
  **This is the one field pair the Nano-shaped tiny fixture cannot exercise**, since it never had
  an MTP head to begin with — checked what actually happens, not assumed benign:
  `decoder/config.go`'s `Config` struct has **no field mapped to either JSON key**, so Go's
  `encoding/json` silently drops both on unmarshal (unknown-field behavior, not a parsing error).
  `cfg.NumLayers` is derived purely as `len(cfg.LayersBlockType)` (52) inside
  `nemotronhArchitecture` — a count that never reads `num_nextn_predict_layers`, so its presence
  or value cannot inflate or shrink the layer count the loader iterates. That reasoning is
  code-verified (both fields are genuinely absent from `Config`, checked by grep), but **it is
  reasoned from code, not confirmed against the real checkpoint's tensor index** — whether the
  actual safetensors shards carry a 53rd layer's worth of MTP-head tensors under some naming the
  main load loop could accidentally touch is unknown until the checkpoint is actually inspected or
  loaded. If T3 fails on a tensor-count or shape mismatch, this is the first place to look.

### Adapter decision: no registry change

Because every field above already round-trips through `nemotronhArchitecture` /
`validateNemotron` / `nemotronTensorSchema` unmodified, **this is not a new registry entry** — the
manifest's `nemotron_h` row (keyed by registry `model_type`, per G4's own precedent: "recorded on
the existing nemotron_h manifest row rather than a sibling") already covers this checkpoint's
architecture. The existing `nemotron3nano-tiny` T1 fixture already exercises this exact geometry
(identical shape), so no new tiny fixture was built — doing so would duplicate a fixture that
tests nothing new, which the repo's own anti-duplication instinct argues against.

### What T3 here actually checks (and what's ready, vs run)

A tiny fixture proves the *shape*; it cannot catch a wrong tensor name, a transposed expert stack,
or a router-bias read from the wrong key on **these specific trained weights** — every one of
which produces correct shapes and plausible values regardless of the training run behind them.
That is the one thing Phase 0's config-identity finding cannot establish, so it is still worth a
real-checkpoint gate.

**Built and merged, not yet run:**

- `scripts/pin_nemotron35lightning_real.py` — `pin_nemotron3nano_real.py` with the checkpoint path
  swapped; same prompt, same greedy-continuation-6 shape.
- `decoder/nemotron35lightning_real_test.go` (tag `realckpt`), `TestNemotron35LightningReal_oracle`
  — reuses the same `realLogitOracleQuant` helper Nano's own gate uses, same `int8`
  weights/f32-activations quant (Nano's own measured finding: int8 *activations* cliffed this
  family's router from cosine 0.997668 to 0.978086 at 6-of-128 sparsity — expected to transfer
  given the identical router shape, but that transfer is itself unverified until this gate runs).
  **Deliberately left out of `cmd/gate/parity.go`'s `emitGates`** (added to `parityRealckptGates`
  instead): the manifest's `nemotron_h` row is already `validated` from Nano specifically, and a
  routine sweep re-emitting this gate's `PARITY_ROW` would silently overwrite Nano's own
  already-recorded numbers with Lightning's on every `EMIT_MANIFEST=1` sweep, discarding real
  evidence for no gain. `TestNemotron3NanoReal_oracle` itself sets the precedent — it isn't in
  `emitGates` either.
- Asset registered: `GOINFER_NEMOTRON35LIGHTNING_HF` → `$MODELS/nemotron35lightning-30b-bf16`
  (`testdata/assets.json`), same `dir` shape as Nano's own entry.

**Not run this pass:** the checkpoint is bf16 ~60 GB+, well past this Mac's 29 GB free disk / 16 GB
RAM. The Linux box (`nobara-pc`, 229 GB free / 62 GB RAM) is the intended target per the brief's
own "30B-A3B class on the Linux box" rule, matching where Nano's own T3 ran. **No CHANGELOG entry
for F2**: nothing loadable changed (no registry key, no adapter code, no tensor schema) — only
test/asset infrastructure, which is not a capability change.

## F3 · Granite 4.2 (dense, 3B/8B/30B) — DONE at T1 (mac, 2026-09-06)

### Phase 0 (all three released sizes fetched and diffed)

`ibm-granite/granite-4.2-{3b,8b,30b}`: `model_type: "granite"` (`GraniteForCausalLM` — distinct
from `granitemoehybrid`'s `GraniteMoeHybridForCausalLM`), no key registered for it. All three
carry `attention_bias: false`, `rms_norm_eps: 1e-05`, both a nested `rope_parameters` object and a
redundant flat `rope_theta` (transformers 4.57.1 writes both), `tie_word_embeddings: false` on
**every size** (the brief's own caution — "confirm tie_word_embeddings per size" — checked and
found NOT to vary here), and Granite's four scalars: `embedding_multiplier`, `attention_multiplier`
(the only one that varies meaningfully: 0.0078125 / 0.015625 / 0.0078125 for 8b/3b/30b — NOT
1/√head_dim, e.g. 8b's head_dim 128 would give 0.0884), `residual_multiplier`, `logits_scaling`.

**`embedding_multiplier`, `residual_multiplier` and `logits_scaling` are 1.0 on all three sizes.**
Only `attention_multiplier` deviates from its HF default. This is a real, checked-not-assumed
finding with a direct consequence below.

**Confirmed the tensor names are byte-identical to llama's**, not by reading `modeling_granite.py`
and eyeballing similarity but by instantiating a real `GraniteForCausalLM` (transformers 5.15.0)
and reading its `state_dict()` keys directly: `model.embed_tokens.weight`,
`model.layers.N.self_attn.{q,k,v,o}_proj.weight`, `model.layers.N.mlp.{gate,up,down}_proj.weight`,
`model.layers.N.{input,post_attention}_layernorm.weight`, `model.norm.weight`, `lm_head.weight` —
no bias tensors, no QK-norm tensors. `llamaTensorSchema` is reused verbatim.

### GGUF — verified against a real file

Fetched the first 15 MB of `bartowski/granite-4.2-3b-GGUF`'s `Q2_K.gguf` via HTTP Range and parsed
it with `aikit/embed.GGUFFile` directly: `general.architecture` is `"granite"` (distinct from the
hybrid's `"granitehybrid"`), and llama.cpp bakes the four scalars directly into metadata —
`granite.attention.scale` (the **resolved** attention_multiplier, 0.015625 for the 3B, matching
its released config exactly — no 1/√d default-resolution needed on the GGUF side),
`granite.embedding_scale` / `granite.logit_scale` / `granite.residual_scale` (all 1, confirming the
Phase 0 finding independently). Tensor set: `blk.N.attn_{q,k,v,output}`, `ffn_{gate,up,down}`,
`attn_norm`, `ffn_norm`, `token_embd`, `output`, `output_norm` — exactly llama's, no bias, no
QK-norm tensors.

### Adapter decision: pure composition on three already-generic scalar fields, one hard guard

`EmbedScale` (embedding_multiplier), `AttnScale` (attention_multiplier, in place of the
1/√head_dim default), and `LogitScale` (logits_scaling) are **already generic** on `Architecture` —
each is just "multiply by this scalar," applied at the same call sites regardless of which family
supplies the value or how that value was derived (Gemma's √hidden `EmbedScale` and Cohere's
`LogitScale` already prove the mechanism is family-agnostic). **`residual_multiplier` is the one
exception**: `granitemoehybrid`'s own dedicated forward (`runLayersGranite`) applies it via
`graniteParams.ResidMul`, and the generic uniform-layer path this family rides has no such hook.
Since every released 4.2 size ships it at the identity value 1.0, `validateGraniteDense` **rejects
any other value** rather than silently dropping it — the same discipline `validateLlama` already
applies to scaled RoPE, and cheap today (it costs nothing on any currently-published checkpoint).

`decoder/registry.go`: `graniteDenseArchitecture` (reuses `ropeBaseFlatOrNested`, same as the
hybrid). `decoder/config.go`: `validateGraniteDense`. `decoder/gguf.go`: `ggufGraniteDenseConfig` +
one dispatch line. No new tensor schema (reuses `llamaTensorSchema`).

### Chokepoints and resident admission

Registering `"granite"` surfaced the same three registry-wide gates F1 did:
`TestResidentAdmission_registryCovered`/`archFeatureProfile`, `TestParityManifest_fresh` (manifest
row), `TestCapabilityMatrix_CoverageComplete` (`representativeConfig`/`familyDoc`) — all satisfied.

**`archFeatureProfile["granite"]` is `{}` — empty on purpose, not by omission.** Checked against
`FeatEmbedScale`'s and `FeatLogitScale`'s own derivation (`add(a.EmbedScale > 1, ...)`,
`add(a.LogitScale != 0 && a.LogitScale != 1, ...)`): since every released size's
`embedding_multiplier`/`logits_scaling` are exactly 1.0, **neither feature flag ever fires for a
real checkpoint of this family** — the derivation and the Phase 0 finding agree independently.
`attention_multiplier` isn't gated by any `ResidentFeature` at all; it's baked into `AttnScale`,
which every backend reads generically regardless of value. So `admissionGolden["granite"]` is
`{cuda, metal, webgpu}` — the same set `llama` gets, for the same reason: nothing this family needs
is missing anywhere.

### T1 — tiny-golden, DONE

`scripts/pin_granite_dense_tiny.py` (transformers 5.15.0, `GraniteForCausalLM`, 4 layers, **all
three tunable multipliers set to non-trivial values** — 12.0 / 0.5 / 6.0 for embedding/attention/
logits, residual pinned to 1.0 as the adapter requires): `TestGraniteDense_forwardParity` —
**argmax exact, cosine 0.999999999999987**, plus direct assertions that the resolved
`EmbedScale`/`AttnScale`/`LogitScale` equal the fixture's non-default values (not just that the
final logits happen to match, which a dropped-then-cancelled multiplier could also produce).

`parity_manifest.json` row: `status: experimental`, `method: tiny-golden`. Added to
`cmd/gate/parity.go`'s `parityGates` (`{"granite-dense", "TestGraniteDense_forwardParity"}`) and
`awaitingFirstConfirmation` (learned from F1's CI break — see below).

### What was deliberately not done

- **T3 real-checkpoint parity.** Same order-of-operations as F1: ship at T1, move on. Granite 4.2
  8B (bf16 ~16 GB) is the natural T3 target for this Mac per the brief's own sizing note, but
  wasn't run this pass — no CHANGELOG claim is made about real-checkpoint validation.
- **30B real-checkpoint / load check.** Explicitly out of scope per the brief ("30B is a load
  check only") and not attempted.
- **GGUF real-file round-trip test.** Same posture as F1: the header, metadata, and tensor
  names/dims were verified against a real file; the tensor values were not (no dequant, no
  forward). No committed tiny-GGUF fixture exercises this in CI.

### Process note carried over from F1

F1's push broke CI: `TestParity_everyRequiredGateIsConfirmed` (audit G-04's ledger guard) correctly
flagged `TestQwen3Moe_forwardParity` as a required gate with no ledger entry. Fixed same-day by
adding it, and `TestNemotron35LightningReal_oracle`, to `cmd/gate/parity.go`'s
`awaitingFirstConfirmation`. F3's own new gate was registered there in the same commit that added
it, closing the loop this once cost a red `main` to discover.

## F4 · Ling-3.0-tiny (inclusionAI, model_type "bailing_hybrid") — Phase 0 DONE, KDA rehearsal DONE, NO registry work (mac, 2026-09-06)

**Verdict up front, per the brief's own framing: does not STOP, but does not become a registered
family this pass either.** No stop condition fires (weights are posted, real, and BF16-readable;
the checkpoint composes from exactly three primitives — nothing beyond KDA+MLA+MoE), so Phase 0
proceeds; per the brief's explicit scope for this item ("proceed ONLY to a synthetic-tiny bring-up
… no real-checkpoint work in this pass"), what follows is that rehearsal, not a shipped adapter.

### Phase 0

**(a) Weights are posted, not announced-only** — the specific trap `docs/post-v1.0-models.md`
already recorded for Ling 3.0's July announcement. Confirmed via the HF API, not a model card:
32 real safetensors shards (`model-00000-of-00032.safetensors` … `-00031-of-00032`) plus
`model.safetensors.index.json`, `torch_dtype: "bfloat16"` — readable today (no fp8/mxfp4 blocker,
the thing that stopped DeepSeek-V4-Flash and gates Kimi K3). MIT license. `custom_code`
(`trust_remote_code` needed): `configuration_bailing_moe_v3.py` + `modeling_bailing_moe_v3.py`
ship in the repo.

**(b) Exact mixer inventory**, from the real `config.json` and the real `modeling_bailing_moe_v3.py`
(fetched directly, not summarized from a description): `model_type: "bailing_hybrid"`,
`architectures: ["BailingMoeV3ForCausalLM"]`.

- **MLA**: `q_lora_rank` 256, `kv_lora_rank` 512, `qk_nope_head_dim` 128, `qk_rope_head_dim` 64,
  `qk_head_dim` 192 (= nope+rope), `v_head_dim` 128, `rope_interleave: true` — the SAME field
  names and shapes `deepseekArchitecture` (`decoder/registry.go`) already reads for DeepSeek-V2/V3.
  One real delta: `gated_attention_proj_granularity_type: "head_wise"` adds a per-head output gate
  (`g_proj: Linear(hidden, num_heads)`) that DeepSeek's own MLA doesn't have — structurally the
  same primitive Laguna's `FeatAttnOutputGate` already ships (`docs/task-laguna.md`), not a new one.
- **MoE router**: `topk_method: "noaux_tc"`, `scoring_func`/`score_function: "sigmoid"`,
  `routed_scaling_factor: 2.5`, `n_group` 8 / `topk_group` 4, `num_experts` 128,
  `num_experts_per_tok` 8, `num_shared_experts` 1, `moe_shared_expert_intermediate_size` 512,
  `first_k_dense_replace: 1` (one dense-prefix layer) — byte-for-byte DeepSeek-V3's `noaux_tc`
  shape, already implemented by `routeExperts`/`moeMLP`. `modeling_bailing_moe_v3.py`'s
  `BailingMoeV3Gate.forward` confirms the shared expert is added UNGATED
  (`y = y + self.shared_experts(identity)`), the same convention DeepSeek/GLM already use.
- **KDA:MLA ratio**: `layer_group_size: 4`, `num_hidden_layers: 24`. `BailingMoeV3DecoderLayer`'s
  actual layer-type decision (quoted from source): every layer where
  `(layer_idx+1) % layer_group_size == 0` is MLA ("attention"); every other layer is KDA
  ("linear_attention") — a **repeating 3-KDA-then-1-MLA pattern**, confirming the brief's own
  "3:1 alternating stacking" description exactly: 18 KDA / 6 MLA over 24 layers.
- **KDA itself — the one genuinely new primitive, verified against fla-org/flash-linear-attention's
  actual source** (`fla/ops/kda/{naive,gate}.py`, MIT), not the HF modeling file's paraphrase
  (which only calls the opaque Triton kernels `chunk_kda`/`fused_recurrent_kda`). Structurally
  **identical to the Gated DeltaNet this repo already ships** for `qwen3_5_moe`
  (`gatedDeltaNetStep`, `decoder/deltanet.go`): a q/k/v projection through a depthwise short conv
  (`short_conv_kernel_size: 4`), a beta write-gate (`sigmoid(b_proj(h))`), an outer-product
  delta-rule state update, q/k L2-norm-in-kernel, a gated RMSNorm before `o_proj`
  (`FusedRMSNormGated(..., activation='sigmoid')` — Gated DeltaNet's own gated norm uses SiLU, a
  real but small delta). **The one load-bearing difference**: Gated DeltaNet's decay is ONE SCALAR
  per head, multiplying the WHOLE `[head_k_dim, head_v_dim]` state block
  (`gatedDeltaNetStep`'s `gt := exp(negExpA[headV]*softplus(...))`, applied uniformly to every
  element of `S`); KDA's decay is **PER-CHANNEL** — shape `[H, head_k_dim]`, one independent value
  per ROW of the state matrix (`naive_recurrent_kda`: `S = S * g_i[..., None].exp()`, where `g_i`
  is `[H,K]` and only broadcasts over the value dimension). This is the real, sourced distinction
  between "Gated DeltaNet" and "Kimi Delta Attention" the literature describes — not a
  renaming, and not reducible to a config flag on the existing DeltaNet code.
- Ling-3.0-tiny's own config selects the SIMPLEST variant of two optional wrinkles:
  `no_kda_lora: true` (a single `f_proj`/`g_proj` linear per gate, not the LoRA'd `a_proj`/`b_proj`
  split `modeling_bailing_moe_v3.py` also supports) and `kda_safe_gate: true` with
  `kda_lower_bound: -5` (the gate saturates through `naive_kda_lowerbound_gate`:
  `g = lower_bound · sigmoid(exp(A_log) · (raw_gate + dt_bias))`, rather than the plain
  `naive_kda_gate`'s unbounded `-exp(A_log)·softplus(...)`).

**(c) Mapping onto existing primitives** — MLA + MoE ride `deepseekArchitecture` as config
composition (plus one small new output-gate primitive already shipped via Laguna); KDA needs one
genuinely new sequence-mixer, structurally a variant of the already-shipped Gated DeltaNet with a
per-channel instead of per-head-scalar decay.

**(d) dtype**: BF16 primary. Readable — no blocker.

### The rehearsal: KDA's per-channel-gated delta rule, proven against a real reference

Per the brief's scope for this item, built and verified ONLY the new piece — not a full layer, not
a registered family, no MLA, no MoE, no conv/gated-norm wrapper (those are already-shipped
primitives with their own existing proof).

- `scripts/pin_kda_rehearsal.py`: `naive_kda_lowerbound_gate` and `naive_recurrent_kda` copied
  **verbatim** (MIT license) from `fla-org/flash-linear-attention`'s actual source on GitHub —
  fetched directly, not reconstructed from a description, and not run through the full `fla`
  package (which pulls in Triton, unavailable without a CUDA toolchain on this Mac; the two
  reference functions themselves are plain PyTorch/einops). A tiny synthetic input (1 batch, 6
  timesteps, 2 heads, head_k_dim=4, head_v_dim=4) drives both functions and dumps a golden.
- `decoder/kda_rehearsal.go`: `kdaLowerBoundGate` + `kdaRecurrentStep`, **not wired to any
  registered family** — this is deliberately a standalone rehearsal, matching the brief's own
  framing that no real-checkpoint or registry work belongs in this pass.
  `TestKDARehearsal_matchesReference` (`decoder/kda_rehearsal_test.go`) checks the gate
  computation AND the recurrence output independently (so the two could not cancel out and hide
  behind one passing final comparison): **maxAbsDiff 2.98e-08, cosine 1.00000000** — float32
  rounding, not an approximation.

### Estimate for the full family (the deliverable this rehearsal earns)

- **KDA layer, full**: the rehearsal proves the core recurrence + gate exactly; remaining work is
  the wrapper this pass deliberately didn't re-prove — the depthwise short conv (LFM2/Mamba-2
  precedent exists; `short_conv_kernel_size` 4 matches LFM2's own convention), the gated RMSNorm
  with **sigmoid** activation (Gated DeltaNet's own gated norm uses SiLU — a real, small,
  config-driven delta to thread through, not a new primitive), and the **chunked/parallel** scan
  (`naive_chunk_kda`/`chunk_kda`) for prefill throughput — this rehearsal only covers the
  sequential form, matching `gatedDeltaNetStep`'s own current scope (no chunked Gated DeltaNet
  path exists in Go today either, per `decoder/deltanet_chunked.go`'s own scope — worth checking
  before assuming that work is novel to KDA). Estimate: **small-to-moderate**, closely bounded by
  Gated DeltaNet's own bring-up cost, now that the one genuinely new piece is de-risked.
- **MLA + MoE**: composition on `deepseekArchitecture`, config-mapping delta only, plus the
  Laguna-precedented output gate. Estimate: **cheap**, same class as F1/F3 this doc.
- **Registry + loader wiring, tiny fixture, T1**: same shape as every family in this doc — a new
  `model_type` key, a tensor schema (three tensor sets: MLA's, KDA's, MoE's — all individually
  precedented), `representativeConfig`/`familyDoc`/`archFeatureProfile` entries, a
  `BailingMoeV3ForCausalLM` tiny fixture via `trust_remote_code=True` (the repo ships the modeling
  code needed). Estimate: **one focused session**, similar order to F1/F3.
- **T3 real-checkpoint**: 7.9B total / 1.3B active — small enough to co-reside with an HF f32
  reference even on this Mac's 16GB, unlike every 30B-A3B-class family in this doc. The cheapest
  real-checkpoint validation of the four families touched this pass, if and when the family lands.
- **Total, rough order of magnitude**: comparable to one of F1/F3 above PLUS a Gated-DeltaNet-sized
  new-primitive cost — not a `qwen3_next`-scale undertaking, because the hardest primitive (KDA's
  core math) is now de-risked by this rehearsal rather than an open question.
- **Why this matters beyond Ling itself**: this rehearsal is explicitly filed as the dry run for
  Kimi K3 (`docs/post-v1.0-models.md`'s "Watch — Kimi K3": MLA on only 24 of 93 layers, the rest
  KDA, plus a latent-MoE wrapper and a new activation — "closer to qwen3_5_moe's DeltaNet family
  than to anything MLA-shaped"). K3's own weights are still `mxfp4-pack-quantized` in an unconfirmed
  layout and its arch should still be re-checked when they actually drop (per that doc's own
  caution) — but the one piece independent of K3's own quirks, the KDA recurrence itself, is now
  proven end-to-end against a real upstream reference rather than an open question.

### What was deliberately not done

No registry key, no tensor schema, no adapter function, no CHANGELOG entry (nothing loadable
changed), no real-checkpoint work, no chunked-scan implementation, no MoE/MLA code (both already
exist and weren't re-exercised here). This is Phase 0 plus one scoped, verified rehearsal — exactly
the brief's stated ceiling for this item.

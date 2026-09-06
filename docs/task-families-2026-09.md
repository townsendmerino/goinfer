# Task: four families, 2026-09 (F1–F4)

> **Status: IN PROGRESS.** One section per family, filled in as each lands (order F1 → F2 → F3 → F4,
> each independently shippable per `docs/post-v1.0-models.md` "Next up"). Read a family's own
> status line before citing it as done.

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

### GGUF — verified against a real file, not assumed

Fetched the first 30 MB of `unsloth/Qwen3-30B-A3B-GGUF`'s `Q2_K.gguf` via HTTP Range (well under
the ~12 GB full file — GGUF metadata + tensor-info lives at the front) and parsed it with goinfer's
own `aikit/embed.GGUFFile` reader, the same ground-truth method the Nemotron 3 Nano T3 handoff used:

- `general.architecture` is literally `"qwen3moe"`.
- Metadata is the plain, un-exotic `{arch}.attention.head_count/head_count_kv/key_length/
  layer_norm_rms_epsilon`, `{arch}.expert_count/expert_used_count/expert_feed_forward_length`,
  `{arch}.rope.freq_base` — **no sliding-window or YaRN keys at all** (unlike Mellum's per-layer-type
  split; qwen3_moe is single global RoPE).
- Tensor set: `blk.N.attn_{q,k,v,output}`, `attn_{q,k}_norm`, `ffn_norm`, `ffn_gate_inp`,
  `ffn_{gate,up,down}_exps` — **no `ffn_*_shexp` tensors**, confirming the no-shared-expert finding
  independently on the GGUF side.

Because both findings are "this is exactly the generic shape, just missing the shared-expert
piece," **zero new code was needed in the generic GGUF `loadLayer` path** (`decoder/gguf.go`) — it
already gates QK-norm on `arch.QKNorm`, the router+experts on `arch.MoE != nil`, and the shared
expert on `arch.MoE.SharedIntermediateDim > 0`. Only a per-family `Config` builder
(`ggufQwen3MoeConfig`) and a `case "qwen3moe":` dispatch line were needed.

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

- **T3 real-checkpoint parity.** Qwen3-30B-A3B bf16 is ~61 GB; the doc's own order-of-operations
  ("if F1's T3 cannot run, ship at T1 and move on") applies — this pass didn't have Linux-box CUDA
  time budgeted. Row stays `experimental` until a real oracle run lands, same posture `qwen3_5_moe`
  and `qwen3_next` shipped at before their own T3s.
  `models-pull Qwen3-30B-A3B` (or the Coder variant) + a real-checkpoint gate mirroring
  `decoder/nemotron3nano_real_test.go`'s shape is the follow-up.
- **One decode benchmark row** (the brief's "one cell each" bench requirement) is gated on T3 —
  measuring a resident quant path before the numerics are validated against a real oracle would be
  exactly the kind of ungated claim `docs/benchmarks.md`'s Methodology section exists to prevent.
- **GGUF real-file round-trip test.** The header was verified (above); a full tiny-GGUF fixture +
  parity gate (mirroring `TestGGUF_qwen3_parity`) is not built — the generic loader path gives high
  confidence, but "high confidence" is not the same claim as "gated," and this is named here rather
  than silently skipped.
- **CUDA/Metal/WebGPU resident kernel work.** None is needed — see the admission section above —
  but no resident-parity test was run on real hardware for this family specifically; it rides the
  same generic MoE dispatch `mixtral`/`qwen2_moe` already exercise on real backends elsewhere.

# Prompt: Nemotron 3 Nano T3 real-checkpoint parity (run on the CUDA box)

> **STATUS: DELIVERED.** `TestNemotron3NanoMoEReal_gate` (`decoder/nemotron_moe_real_test.go`)
> exists and the v0.14.0 CHANGELOG records the T3 real oracle at **cosine 0.997668**, continuation
> exact, plus a real Q4_K_M GGUF gate.


Paste the block below into a Claude Code session on the CUDA box (`nvidia-rtx2070s`). Everything
above the line is context for whoever is dispatching it; the prompt itself is self-contained.

**Why this exists.** Nemotron 3 Nano (`G4` in `docs/queue-correctness.md`) has Phase 0/1 done on
the Mac: the adapter, forward path, safetensors loader, GGUF loader, and a real HF-oracle T1
golden (tiny random-weight checkpoint, cosine 1.000000) all landed and verified. What's left is
T3 — a forward pass through the ACTUAL released model compared against a real oracle — and this
one genuinely doesn't fit on the Mac: the checkpoint is 20GB+ and the Mac has ~24GB free with no
memory headroom to load a 30B model twice (goinfer's loader isn't streaming/paged for this family
yet — every layer loads fully into memory). This is a disk/RAM problem, not a work-scoping one.

---

In this repo (`goinfer`, branch `main`), I need T3 (real-checkpoint parity) for Nemotron 3 Nano —
step 3 of the family-onboarding checklist (`docs/parity-coverage-policy.md`, "Definition of done
for a new family"). This is a validation task: the adapter code is already written and works
against a real HF oracle at tiny scale. Your job is to prove it also works at real scale, record
the result in `testdata/parity_manifest.json`, and update `docs/queue-correctness.md`'s `G4` entry
— not to redesign anything, unless real-scale testing finds a bug the tiny fixture couldn't catch
(entirely possible — say so plainly if it does, that's a legitimate and useful outcome here).

## What's already true, verified — don't re-derive these

**The Go implementation** (all landed, `go test ./decoder/...` green):
- `decoder/arch.go`: new `nemoMoE` block-kind.
- `decoder/config.go`: pattern parser accepts `E` (MoE-FFN) alongside `M`/`*`/`-`; new
  `MoeSharedExpertIntermediateSize` field (verified NOT derivable as
  `NSharedExperts*MoeIntermediateSize` — the real checkpoint ships 3712, not 1856).
- `decoder/registry.go`: `nemotronhArchitecture` populates `arch.MoE` when the pattern has an
  `E` block. Router is DeepSeek-V3's `noaux_tc` shape (sigmoid + `e_score_correction_bias` +
  group-limited top-k), `RouterSigmoid: true` unconditionally (verified against NVIDIA's real
  `modeling_nemotron_h.py` — no `scoring_func` config key exists for this family). Also accepts
  `layers_block_type` entries spelled `"linear_attention"`/`"full_attention"` (transformers'
  canonicalized names) as aliases for `"mamba"`/`"attention"` — found empirically generating the
  T1 fixture, not theoretical.
- `decoder/forward_nemotron.go`: `nemotronMoE`/`nemotronExpertFFN` — non-gated relu² experts
  (confirmed against the real safetensors tensor index: `up_proj`/`down_proj` only, no
  `gate_proj`), NOT `moeMLP` (which hard-requires gated SwiGLU and Nemotron-H never used it
  anyway — separate dedicated forward loop).
- `decoder/weights.go`: safetensors loader, tensor names
  `backbone.layers.N.mixer.{gate,gate.e_score_correction_bias,experts.I.{up,down}_proj,shared_experts.{up,down}_proj}.weight`.
- `decoder/gguf.go`: GGUF loader. **llama.cpp's GGUF arch string is `nemotron_h_moe`, not
  `nemotron_h`** (normalized to `nemotron_h` on load since that's what the registry dispatches
  on). Verified directly against a real file's header (HTTP Range fetch, not a secondhand PR
  summary — which turned out to be wrong about one key). Real metadata keys: `expert_count`,
  `expert_used_count`, `expert_feed_forward_length`, `expert_shared_feed_forward_length`,
  `expert_shared_count`, `expert_weights_norm`, `expert_weights_scale`, `expert_group_count`,
  `expert_group_used_count`. Tensor names: `blk.N.ffn_gate_inp.weight` (router),
  `blk.N.exp_probs_b.bias` (correction bias), `blk.N.ffn_{up,down}_exps.weight` (fused 3-D, ALL
  experts in one tensor per projection — different layout than safetensors), `blk.N.ffn_{up,down}_shexp.weight`
  (shared expert). Also spot-checked dequantization directly against ~3GB of real Q4_K_M data
  (one full MoE layer, all six tensor kinds, expert indices 0/1/127) — correct dims, sane values,
  zero NaN/Inf. **Not yet run: a full forward pass through the whole file.** That's this task.
- `decoder/residency.go`: GPU residency correctly DECLINED for this family (both the dense parent
  and the MoE variant) — confirmed `--backend metal`/`--backend cuda` currently give it zero
  speed benefit over `--backend cpu` (`metalBackend.MatmulBT` is literally the CPU SIMD kernel;
  `*metalBackend` doesn't implement `QuantBackend`). Not your problem to fix — just don't expect
  GPU accel for this one yet, on either backend.

**T1** (real HF oracle, tiny scale): `decoder/nemotron_moe_test.go`'s
`TestNemotron3NanoMoE_textParity` — cosine 1.000000, exact argmax, against transformers 5.15.0's
real `NemotronHForCausalLM` forward (no `trust_remote_code` needed, mainline support). Fixture:
`scripts/pin_nemotron3nano_tiny.py`.

**Real config, verified against the actual release** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`):
`hidden_size=2688, num_hidden_layers=52, num_attention_heads=32, num_key_value_heads=2,
head_dim=128, n_routed_experts=128, num_experts_per_tok=6, moe_intermediate_size=1856,
moe_shared_expert_intermediate_size=3712, n_shared_experts=1, routed_scaling_factor=2.5,
n_group=1, topk_group=1, vocab_size=131072`. `hybrid_override_pattern` (52 chars):
`MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` — 23 mamba, 23 moe, 6 attention.

## The one real gotcha for T3 specifically: no fp8 support

**Do not use the `-FP8` checkpoint variant.** goinfer has NO fp8 reader anywhere in the tree —
this was the exact blocker that stopped DeepSeek V4-Flash's scoping
(`docs/completed/task-model-family-deepseek-v4-kimi-k3.md`), and it applies here too if you reach
for the FP8 repo without checking first. Use one of:
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (native release, safetensors, ~55-60GB — goinfer
  requantizes on load same as every other family, e.g. `Options{Quant:"int4int8"}`).
- A GGUF quant from `bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF` (or
  `lmstudio-community`/`moxin-org`'s mirrors) — smaller on disk (Q4_K_M is ~20GB; smaller quants
  like Q2_K/IQ variants exist if disk is tighter), and the loader side is already spot-checked
  against real data from this exact repo (see above). **This is probably the better default** —
  check disk/RAM on the box first and pick based on what actually fits with room for the
  reference oracle too.

## What T3 needs

Per `docs/parity-coverage-policy.md`'s step 3: a real-checkpoint parity run against a real oracle,
recorded in `testdata/parity_manifest.json` at the landing commit. **`nemotron_h`'s own entry is
already there** (`status: "validated"`, `method: "real-model-oracle"`, `uses: [core, loaders,
quant, mamba2]`, `own: [decoder/forward_nemotron.go]`) — it's the dense checkpoint's T3, and it's
the closest possible template since it's literally the same `model_type`. Find whatever test file
backs it (`grep -rn nemotron_h testdata/parity_manifest.json` then trace to the real-checkpoint
test that produced that row) and mirror its method for the MoE checkpoint — either extend that
entry's coverage or add a sibling row, whichever the manifest's actual schema makes more natural
once you're looking at it; `deepseek_v3`/`glm4_moe` are secondary references for how a large MoE
family's row looks if the mamba-hybrid template doesn't transfer cleanly.

**If the full HF oracle (`transformers`, running the real 30B model) doesn't fit the box's
RAM/VRAM** — plausible; the RTX 2070S is 8GB VRAM, and a 30B model even 4-bit-quantized in
`bitsandbytes` is a real ask — the checklist has a named fallback: `weightDiff` (comparing loaded
weight statistics/checksums rather than a full forward pass) is explicitly sanctioned for exactly
this case ("or `weightDiff` if the oracle OOMs"). Use it if the full oracle genuinely doesn't fit,
and say so plainly in the manifest entry and the commit — don't silently downgrade the proof
without recording that it happened.

## Deliverable

1. A T3 real-checkpoint parity result (full oracle comparison, or `weightDiff` with the OOM reason
   stated) for Nemotron 3 Nano, recorded in `testdata/parity_manifest.json`.
2. Whatever test file backs it, committed so it can be re-run (matching this family's other test
   files' naming: `decoder/nemotron_moe_*_test.go`).
3. `docs/queue-correctness.md`'s `G4` entry updated with the result — if it passes clean, note
   that T3 is now done and only T2 (adding this to the sweep list) + a capability-matrix row
   remain before "supported" is a backed claim; if it surfaces a real bug the tiny fixture missed,
   record what and where, the same honest-negative-is-a-good-outcome standard the rest of this
   family's scoping has followed.

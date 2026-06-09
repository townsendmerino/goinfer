# Task: Qwen3.5/3.6-MoE real-checkpoint gate — measurable, not coherence

> **North star (this defines "done"):** this is an **aikit milestone** — "aikit
> can load and correctly run a frontier-class MoE + DeltaNet hybrid in pure Go" —
> **not a ken feature.** ken's generation is BYO-endpoint by decision; a 35B at
> single-digit tok/s is not an interactive backend. The gates below *are* the
> milestone. The algorithm is already bit-exact (cosine 1.0 on
> `testdata/qwen35-tiny`); **the unproven thing is the LOADER** on real tensors
> (names, shapes, 256-expert stacking, DeltaNet geometry). That is testable
> without ever holding the whole 35B in RAM.

Real model: **`Qwen/Qwen3.6-35B-A3B`** (HF `model_type: qwen3_5_moe`, 40 layers,
hidden 2048, 256 experts, 3:1 Gated-DeltaNet/softmax hybrid; ~72 GB bf16
safetensors). Box: 62 GB RAM, 8 GB GPU, 1.5 TB disk. HF rig: `~/g4venv`
(transformers 5.10.2, accelerate 1.13.0, `Qwen3_5MoeForCausalLM` imports).

## Why not "f32 HF parity on the full model"

The recon caught it: the 35B at f32 (140 GB) / bf16 (70 GB) won't fit 62 GB RAM,
and the full forward isn't interactive here — so a single full-model f32 HF run is
the **infeasible-gate trap**. But "loads + decodes coherent tokens" gives up
measurability. So: **two right-sized, measured gates**, not one infeasible run and
not vibes.

## Gate 1 — slice f32 parity (proves the loader; cheap, bit-exact)

Load the **first 4 real layers** (embed + layers 0–3) from the real checkpoint at
**f32**, run goinfer's forward on that slice, compare the hidden state after the
slice **bit-exact (cosine ≥ 1−1e-5)** against HF on the same slice. No full-model
run, no offload. **This is the gate that proves the loader.** Run it BEFORE
building the int8 path: if the loader is wrong on real tensors, you find out
cheaply here.

**Slice composition is deliberate — it must exercise the parts that actually
differ from a vanilla transformer, or it passes for the wrong reason.** Layers
0–3 of qwen3_5_moe cover both novel surfaces:
- **DeltaNet / linear-attention** layers 0,1,2 (full_attention_interval 4 → the
  3:1 mix: 0,1,2 linear, 3 softmax) — the recurrent state geometry, the easiest
  thing to get subtly wrong.
- **256-expert MoE** on every layer — tests the **expert-weight stacking** tensor
  mapping (`stackedExperts`), not just dense projections.
- the softmax (full-attention) layer 3 + per-head QK-norm, for completeness.
A slice of only embed + a dense block would pass while proving nothing about the
two things that matter (MoE expert-stacking, DeltaNet recurrence). Both the
router/expert path and the deltaState path must be inside the bit-exact compare.

**Dtype honesty (do not fudge the bar against a bf16 golden).** Gate 1 is f32 on
goinfer's side; run **HF on the slice in f32 too** — the slice is small (4 layers,
a few GB), so force f32, no offload, and hold the tight **1−1e-5** bar. Do NOT
compare goinfer-f32 against the bf16 offloaded reference at 1e-5: bf16's ~3-decimal
mantissa floors achievable cosine at ~1−1e-4, so a 1e-5 bar would either fail for
the wrong reason or get quietly fudged. f32-vs-f32 on the small slice is feasible,
so do that and keep the tight bar. (The bf16 offloaded golden is for Gate 2, where
there's no choice — and there the bar is the int8 quant tolerance, looser anyway.)

## Gate 2 — full-model int8 token-agreement (proves the quant path end-to-end)

goinfer int8 (streamed load, ~35 GB, fits 62 GB) vs an **offloaded bf16 HF
reference** on ~10 fixed prompts: greedy **top-1 token agreement** + logit cosine
within the expected int8 quant tolerance. "Coherent tokens" rides on top as a
**human smoke test — NOT the gate.** The gate is the agreement number.

**Golden banked (2026-06-08):** `~/models/qwen35_real_golden/` (outside the repo
— ~80 MB) — 10 fixed prompts × 8 greedy steps, full last-position logits per step
(`promptNN_logits.npy` [8, vocab]) + `manifest.json` (prompt_ids, gen_ids,
gen_text). Captured while HF was warm in page cache (~7 s/step). This is the
Gate-2 reference; goinfer int8 greedy-decodes the same prompts and compares
top-1 token + per-step logit cosine.

## The reference is achievable (don't despair)

`transformers` + `accelerate` `device_map="auto"` + a disk offload folder streams
the 72 GB bf16 model on a 62 GB / 8 GB box — slow (minutes per forward), but a
golden needs only a handful of passes and disk is 1.5 TB. The arch already runs in
HF (the tiny golden went through it); scaling to the real checkpoint via offload
is the only delta.
- Gate 1 reference: HF on just the slice (trivial, no offload).
- Gate 2 reference: offloaded HF on ~10 short prompts (slow but feasible).
- (A Qwen3.5/3.6 q4 GGUF would give an external greedy-token reference via
  llama.cpp — but that needs the GGUF DeltaNet loader, the bigger build. **Don't
  build the GGUF path just to get a reference;** offloaded HF is cheaper.)

## Sequencing — spec + probe before any build

0. **Precondition:** fetch `Qwen/Qwen3.6-35B-A3B` (~72 GB; downloading).
1. **PROBE (the "can we even measure this" gate):** confirm **offloaded HF loads
   and forwards the real 35B on this box** — `device_map="auto"` + offload-folder,
   a few forward passes, capture logits + per-pass time. This decides whether
   Gate 2 has a reference at all. **Do not build the int8 path until this passes.**
2. **Gate 1 (slice f32 parity):** real checkpoint on disk + a slice loader + HF on
   the slice. No int8, no offload. Fast, bit-exact, proves the loader. **Run
   before building the full int8 path.**
3. **Build the int8 streaming quant path** for qwen3_5_moe (experts already go
   through `matmulWeights` → quantizable; stream into int8, no whole-model f32
   spike) — only after Gate 1 is green AND the probe confirms a Gate-2 reference.
4. **Gate 2 (full-model int8 token-agreement)** vs the offloaded HF golden.

## Loader gap found by Gate-1 prep (2026-06-08) — the real build, scoped

Inspecting the real checkpoint's tensor index **before** running Gate 1 surfaced
that the released `Qwen/Qwen3.6-35B-A3B` is a **multimodal (VL) model** whose text
decoder is packed differently from the tiny text-only golden goinfer's loader was
built and parity-tested against. The DeltaNet half matches; the MoE packing and
the prefix do not. Precise deltas (real → what goinfer's safetensors loader
expects):

| surface | real checkpoint | goinfer expects | gap |
|---|---|---|---|
| prefix | `model.language_model.layers.N.*`, `…embed_tokens` | `model.layers.N.*`, `model.embed_tokens` | inject `language_model.` (lm_head stays top-level `lm_head.weight`) |
| **MoE experts** | **fused + stacked**: `mlp.experts.gate_up_proj` (one 3-D `[256, 2·moe_inter, hidden]`) + `mlp.experts.down_proj` (one 3-D) | per-expert **separate** `…experts.%d.{gate,up,down}_proj` | **split fused gate_up + de-stack 256 experts** — the real lift |
| shared expert | `mlp.shared_expert.{gate,up,down}_proj`, `mlp.shared_expert_gate` | `SharedExpert.{Gate,Up,Down}`, `SharedGate` | minor name mapping |
| DeltaNet | `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,conv1d,A_log,dt_bias,norm,out_proj}` | **same** | ✅ none |
| MTP / vision | `mtp.*`, `model.visual.*` | — | ignore (don't request) |

**So Gate 1 is blocked on a real, well-scoped loader build** — not "slice-load and
compare." The scary novel surface (DeltaNet geometry) is already name-compatible;
the work is (1) the `language_model.` prefix and (2) **fused-`gate_up` + stacked-3-D
expert unpacking in the safetensors path** (goinfer's `stackedExperts` exists for
the GGUF MoE path — that logic is the model for the safetensors side, plus the
gate_up split). This is the genuine Track A loader task; the recon's "just load
real weights" understated it. Once built, Gate 1 (slice f32 parity) validates it,
then int8 → Gate 2.

### Loader build — DONE + GREEN (2026-06-08)

Built and validated bit-exact. Three deltas, all in `decoder/`:
1. **VL `text_config` flattening** (`config.go loadConfig`) — a third gap the
   table missed: the VL config nests every text dim under `text_config`, so a flat
   parse read zeros. Flatten it (text_config fills the dims, top-level keys stay
   authoritative); flat configs are unaffected.
2. **`language_model.` prefix injection** (`weights.go`, `mp()`/`tn()` helpers,
   auto-detected from `model.language_model.embed_tokens.weight`; `lm_head.weight`
   stays top-level).
3. **Fused-stacked expert unpacking** (`loadFusedExperts`): splits
   `gate_up_proj [256, 2·moe_inter, hidden]` on the row axis and de-stacks all 256
   experts from the two 3-D tensors, copying each slice so the big arrays free per
   layer. Auto-selected when `mlp.experts.gate_up_proj` is present.

Reference: `scripts/pin_qwen35_slice.py` (HF, `num_hidden_layers=4`, f32, loads
only the 4 shards holding the slice — no offload, ~17 GB). **Subtlety that cost a
debug loop:** HF's `output_hidden_states` applies the final `model.norm` to its
LAST entry (the standard decoder loop appends post-norm). `model.norm` here has
`‖1+w‖≈119`, so comparing goinfer's pre-norm `runLayers` output against
`hidden_states[N]` failed at cosine 0.90 / a 60× magnitude gap — a reference bug,
not a loader bug (layers 0–2 already matched bit-exact). The script now captures
the pre-norm input to `model.norm` via a forward-pre-hook.

## Known v1 limit (keep visible)

Hybrid models opt out of prefix-reuse / spec-decode — the recurrent `deltaState`
isn't position-truncatable (`deltanet.go`). State it; the gate doesn't cover it.

## Don'ts

- **Coherence is the smoke test, not the gate.** The gate is a number (slice
  cosine + int8 token-agreement).
- **Don't build the int8 path before Gate 1 is green and the offloaded-HF probe
  confirms a Gate-2 reference exists.** Building the measured thing before the
  measurement is the one move that breaks the streak.
- **Don't build the GGUF DeltaNet loader just to get a reference** — offloaded HF
  is cheaper.

## Status

- ✅ **Precondition:** `Qwen/Qwen3.6-35B-A3B` downloaded (67 GB, 26 shards) at
  `~/models/qwen3.6-35b-a3b`.
- ✅ **PROBE (green):** offloaded HF (`device_map=auto`, cpu 23 / disk 21 layers,
  no GPU) loads and forwards the real 35B — **~52 s cold / ~5–7 s warm per pass**,
  correct argmax ("The capital of France is" → `Paris`). The box *can* hold the
  reference → **spec-and-build**, not re-scope.
- ✅ **Gate-2 golden banked** (`~/models/qwen35_real_golden/`) while warm.
- ✅ **Loader build (VL prefix + fused-stacked experts + text_config flatten)** —
  loads the real 4-layer slice at f32, all shapes validated against real tensors.
- ✅ **Gate 1 (slice f32 parity) GREEN** — `decoder/qwen35_realckpt_test.go`
  (behind the `realckpt` build tag — needs the 67 GB checkpoint, too heavy for the
  default 10-min `go test` budget; run
  `go test -tags realckpt ./decoder/ -run TestQwen35Real -v`): embed + layers 0–3,
  f32 both sides, **cosine = 1.0000000000 bit-exact at every layer boundary**
  (0.7140 / 0.9128 / 1.2667 / 1.3515, goinfer ≡ HF). The slice covers DeltaNet
  recurrence (layers 0–2), the 256-expert fused-stacked MoE (every layer), and
  softmax + partial-rotary + output-gate + QK-norm (layer 3). Reference banked at
  `~/models/qwen35_real_golden/slice4_f32.json` (`scripts/pin_qwen35_slice.py`).
- ▶ **NEXT: int8 streaming quant path** for qwen3_5_moe — Gate 1 is green and the
  Gate-2 reference exists, so the precondition is met. Then **Gate 2** (full-model
  int8 token-agreement vs the offloaded-HF golden).

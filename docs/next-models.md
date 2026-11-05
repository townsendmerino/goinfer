# Next model families — the breadth backlog, and what became of the last list

> **Audience:** internal planning. Renamed from `post-v1.0-models.md` on 2026-09-12: v1.0 has not
> been cut (the latest tag is v0.17.2), the "post-v1.0" backlog got built anyway, and the old name
> answered a question nobody was asking. What has *landed* is never read from here —
> [`docs/capability-matrix.md`](capability-matrix.md) (families, generated from the registry) and
> [`docs/hardware-matrix.md`](hardware-matrix.md) (which backend runs each resident, generated from
> the admission predicate) are the truth. This doc is for *which family next, and why*.
> Last field check: 2026-10-15. Ollama's 2026-10-11 to 10-25 list read 2026-10-26 (below).

## Where the 2026-08-17 list ended up (status 2026-09-12)

| # | item | status | record |
|---|---|---|---|
| 1 | Nemotron 3 Nano (`G4`) | **landed**, T3'd on a real checkpoint; Nemotron 3.5 Lightning confirmed config-identical, no registry change | [`completed/task-families-2026-09.md`](completed/task-families-2026-09.md) F2 |
| 2 | Qwen3-Next (`G5`) | **landed**, resident on WebGPU, CUDA and Metal | matrix |
| 3 | Laguna (`G6`) | **landed**, CPU path (softplus attention gating and per-layer RoPE scales were the new primitives, as scoped) | matrix |
| 4 | gpt-oss residency (`G7`) | **landed**, resident on all three backends. The one real forward its gate demanded found three silent wiring defects that a 100%-green kernel suite could not see — the lesson `G8` now rests on | [`completed/queue-correctness.md`](completed/queue-correctness.md) |
| 5 | DeepSeek V4-Flash (`G8`) | **PARKED — unvalidatable here.** 0.16 TB against 62 GB of host RAM; no configuration runs one whole forward on hardware this project has. The fp8 prerequisite is half lifted: e4m3 with blockwise f32 scales reads (`decoder/fp8.go`), V4's `ue8m0` scale format does not | [`queue-correctness.md`](queue-correctness.md) |

From the watch list: **Ling 3.0 landed** as `bailing_hybrid` (v0.17.0) — KDA became the fifth
sequence-mixing family, the exact thing the old text said would be "natural" and told us to
verify the weights for first. **Qwen3.8** (the dense 27B) landed. From Tier B, **InternLM2** and
**`qwen3_moe`** landed. Of the seven families in v0.17.0 (`qwen3_moe`, dense Granite, Ministral 3,
SmolLM3, Olmo 3, Olmo Hybrid — where Metal is int4 only and WebGPU is read only on the M1 Pro's adapter — and Ling 3.0), none came from this doc's tiers — they came from
`docs/completed/task-families-2026-09.md` and the G-series. That is the actual finding of the period: this doc
stopped being where families came from once the axes were covered, which is what it predicted.

## Next up — 2026-10-15 field check

A sweep of open-weight releases from mid-August to 2026-10-15, filtered by the selection rule
below. Every config fact here was read from the posted `config.json` or model card on
2026-10-15, not from coverage. The field's headline releases this period are 300B–1T and fail
rule 2 outright (listed under Watching). What is left is short:

1. **Muse Glimmer 30B (Meta, 2026-08-10, Apache 2.0) — the one new family worth building.**
   `model_type` `muse_glimmer`, text `muse_glimmer_text`. Dense, 52 layers, 32 query / 2 KV
   heads, head_dim 128, vocab 202,048, untied head. `layer_types` is three
   `sliding_attention` (window 2,048) to one `full_attention`, and `layer_rope_theta` is 0 on
   the full layers — NoPE globals, the interleave Command-R7B and SmolLM3 already run. Also in
   the config: `final_logit_softcapping` 20 (Gemma 2 has the softcap), `output_multiplier`
   0.196 (a logit scale, like Command-R's), `qk_scale_factor` 3.87 and a separate
   `post_norm_eps` (suggests sandwich norms; verify). Meta's blog describes **gated** GQA and
   per-head QK RMSNorm, and neither appears in the config — the modeling file
   (`modeling_muse_glimmer.py`) is the source for both before any estimate. If the gate is the
   sigmoid head-wise output gate Spark-X2.5 already has, this is adapter composition, not new
   kernels. Why it ranks first: GGUF and llama.cpp support from day 0, an official DFlash
   block-diffusion drafter (block 16) that is the vendor-blessed demo
   [`spec/08-dspark-dflash.md`](spec/08-dspark-dflash.md) wanted (Laguna's was the last), and
   Meta positions it for local always-on agents, which is this engine's lane. **Hardware:**
   int4 is ~15–16 GB, so not resident on the 16 GB M1 Pro; the Linux box runs it. An HF
   reference forward at bf16 is ~60 GB against 62 GB of host RAM — plan the oracle (layer
   streaming, or the llama.cpp route in
   [`tasks/task-llamacpp-inproc.md`](tasks/task-llamacpp-inproc.md) I2) before estimating.
   Text decoder first; the 2B Perception Encoder tower follows [`multimodal.md`](multimodal.md)'s
   policy.

2. **Low-bit read formats — a format bet, not a family bet.** goinfer reads 15 ggml types
   ([`quantization.md`](quantization.md)); it does not read `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`,
   `IQ1_S`, `IQ1_M`, `TQ1_0`, `TQ2_0`, or NVFP4. That now blocks real files, not hypothetical
   ones: Nemotron 3.5 Bolt's official 24 GB build (experts at IQ2_XXS, Mamba layers NVFP4,
   embeddings and head Q8_0), and Unsloth's dynamic quants, which mix the IQ types. NVFP4 was
   already flagged as L-08 in [`completed/audit-2026-09-02.md`](completed/audit-2026-09-02.md);
   the MXFP4 and fp8 e4m3 readers are two-thirds of it. Reading a low-bit file is policy-clean
   here — quantization.md reads Q2_K for the GLM-4.5-Air and Llama-4-Scout gates and declines
   only to *recommend* it. **One cheap guard first, independent of the rest:** PrismML's
   Ternary Bonsai 2 27B (a `qwen35` GGUF — Qwen3.8-27B, an architecture goinfer already runs —
   at 1.75 bits/weight, 5.95 GB) ships in fork-only packings `PTQ1_0` and `PQ2_0` that the GGUF
   metadata labels `TQ1_0` and `Q2_0`, and its card warns stock llama.cpp loads the `Q2_0`-labelled
   file silently into garbage. Check what goinfer does with those two files today, and make it a
   named decline if it is anything else. Supporting PrismML's packings themselves means chasing
   a fork; wait until they are upstreamed.

3. **Mistral Small 4 (119B-A6.5B, Apache 2.0) — a routing guard now, an adapter later.** The
   checkpoint's outer `model_type` is `mistral3` (`Mistral3ForConditionalGeneration`) and its
   `text_config.model_type` is **`mistral4`**: MLA (`kv_lora_rank` 256, `q_lora_rank` 1024),
   128 routed experts top-4, `first_k_dense_replace` 0, 36 layers. goinfer resolves the outer
   key (`decoder/arch_mistral3.go:ministral3Architecture` documents that loadConfig re-applies it
   last), so this checkpoint is handed to the dense Ministral 3 adapter. The guard: decline a
   `mistral3` wrapper whose text config is `mistral4`, with a message that says so. The
   adapter is DeepSeek-V3-shaped substrate (MLA + routed MoE) — verify router scoring and
   tensor names before calling it an alias. Int4 is ~60 GB: Linux box with expert streaming
   only. Mistral Large 4 (2026-10-06) is likely the same text type — check its config at
   pickup, not before.

4. **Nemotron 3.5 Bolt 74B-A9B (NVIDIA, 2026-10-15, OpenMDW-1.1) — watch.** A Mamba-2 /
   attention hybrid with **LatentMoE** (top-9) and a C-RADIO v4-H vision tower. LatentMoE is
   the primitive this doc already deferred for Nemotron Super; the 24 GB build needs item 2's
   formats; a BF16 checkpoint exists. Ships its own DFlash drafter. Re-rank if item 2 lands.

**Already served by an existing adapter (no work, confirmed by config):** MiniCPM5-2B is a
plain `LlamaForCausalLM`; Gemma 4 12B is `gemma4_unified_text`, registered under
`decoder/arch_gemma4.go:gemma4Architecture`; Mistral Medium 3.5 (below). Xiaomi's
MiMo-V2.6-Distill-Qwen-9B is a Qwen3.5-9B fine-tune, so expected to be `qwen3_5` — not checked.

## Field check — 2026-10-26 (an Ollama engineer's list of releases, 2026-10-11 to 10-25)

Read from the list and the sources named here, 2026-10-26. Nothing new to build now; the owner keeps new work behind the
current tracks.

- **Already handled:**
  - EmbeddingGemma-2: shipped in v0.22.0.
  - Flint: `tasks/task-flint-2026-10.md`, F0 in progress.
  - Clef-flash: route C.
  - Nemotron 3.5 Bolt E4B: NVIDIA's post-training of Gemma 4 E4B (8B total, 4B effective, 64K context, Q4_K_M GGUF of
    6.18 GB). Expected to be a `gemma4` checkpoint swap; its Gate 0 checks the architecture, the license and the chat
    template's reasoning toggle and tool format. Its DFlash drafter is a candidate windowed target for SP0
    (`tasks/task-forward-architecture-2026-10.md` §6a).
  - Bolt 74B-A9B: item 4 above, unchanged.
- **Out:**
  - Nemotron 3.5 Super (120B-A12B, vision, ~76 GB in NVFP4): rule 2.
  - Phonon (Gradium, a 40M text-to-speech model): speech output is out of scope.
  - Cloud-only on Ollama: Beam, Muse-spark-open, MiniMax M3.1 Flash, Mistral Large 4, DeepSeek V4.1 Flash. No action;
    Mistral Large 4's config check stays at pickup (item 3 above).
- **Bonsai-2: the format question changed (item 2's guard).**
  - Ollama serves PrismML's 2B, 4B, 9B and 27B as `qwen35` GGUFs in `Q2_0` only.
  - OrcaRouter reads upstream llama.cpp as running `Q2_0` and rejecting `PQ2_0` and `PTQ1_0`.
  - Item 2 says PrismML's `PQ2_0` files carry a `Q2_0` label and load silently into garbage. Whether Ollama's files are a
    true upstream `Q2_0` or the relabelled `PQ2_0` is what the guard must check first.
  - The other reason `completed/task-ternary-weights.md` closed (quality figures that are PrismML's own, unreproduced)
    still stands.
- **Decision models: 11 of the 17 local releases.** Ollama 0.35 serves `/v1/systemone`. The facts, and the owner's call
  to keep the work parked until the current tracks are done, are in `tasks/parked/task-constrained-confidence.md`
  (2026-10-26 note). For this backlog:
  - most are Qwen3.5 or Gemma 4 E-model recipes on families goinfer runs;
  - Laya, NLI, Decima and Largha need an encoder family (ModernBERT, DeBERTa or mmBERT) goinfer does not have;
  - d1 needs `lfm2` on a GPU.

## Field check — 2026-11-03 (releases 2026-10-26 to 11-03)

> Drafted by Gemini from posted config.json files and model cards, 2026-11-03, and read by the coordinator; not
> independently re-fetched. Each candidate's Gate 0 re-reads its config before any estimate.
> Summary: CHEAP: LFM3-1.7B (new `lfm3`, the `lfm2` family plus sliding attention, partial RoPE, a head-wise gate and
> sandwich norms; CPU until a GPU has short-conv), Nanbeige4.3-33B (plain `llama`, nobara only), the MiniCPM-OCR-1.0
> text decoder (`qwen3_5_text` behind a new outer key), Nemotron 3.5 Bolt E4B (confirmed `gemma4`). EXPENSIVE:
> SenseNova-Loop-9B (looped layers with per-pass cache state, plus mHC). Mistral Large 4 posted and is out by rule 2;
> the Mistral Small 4 guard stands. The Bonsai-2 guard must key on its Hadamard-rotated weights, not only the tensor types.

Field check for this doc covering open-weight releases from **2026-10-26 to 2026-11-03** and updates to **Next up** (2026-10-15 and 2026-10-26) and **Watching**. Selection rule (this doc): (1) weights and `config.json` posted, not announced; (2) fits the **16 GB M1 Pro** or **8 GB RTX 2070 SUPER** at int4 (or 62 GB host RAM for an HF oracle forward), or is a new version of a family `goinfer` already runs ([`decoder/registry.go`](../decoder/registry.go)). All config facts below come from the posted `config.json`, `modeling_*.py`, or model card.

---

## Releases passing the selection rule (2026-10-26 – 2026-11-03)

### 1. LFM3-1.7B / LFM3-1.7B-Base and LFM3-VL-2B

- **Name:** `LiquidAI/LFM3-1.7B` ([HF card](https://huggingface.co/LiquidAI/LFM3-1.7B)), `LiquidAI/LFM3-1.7B-Base` ([HF card](https://huggingface.co/LiquidAI/LFM3-1.7B-Base)), `LiquidAI/LFM3-VL-2B` ([HF card](https://huggingface.co/LiquidAI/LFM3-VL-2B)); Ollama `lfm3` ([library](https://ollama.com/library/lfm3)) and `lfm3-vl` ([library](https://ollama.com/library/lfm3-vl)).
- **Maker:** Liquid AI ([LiquidAI org](https://huggingface.co/LiquidAI)).
- **Date:** 2026-10-27 ([HF card](https://huggingface.co/LiquidAI/LFM3-1.7B), [Ollama v0.40.11–v0.40.13](https://github.com/ollama/ollama/releases)).
- **License:** LFM Open License v1.0 (`lfm1.0`; free for research/personal and commercial under $10M annual revenue; [LICENSE](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)).
- **`model_type` / `architectures` (`config.json`):**
  - Text ([`LFM3-1.7B/config.json`](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/config.json)): `"model_type": "lfm3"`, `"architectures": ["Lfm3ForCausalLM"]`; 24 layers (`hidden_size: 2048`, `intermediate_size: 7680`, `num_attention_heads: 16`, `num_key_value_heads: 4`, `head_dim: 128`, `vocab_size: 128000`, `norm_eps: 1e-05`, `tie_word_embeddings: true`, `use_sandwich_norm: true`, `conv_L_cache: 7`, `conv_bias: false`, `sliding_window: 256`); tri-hybrid `layer_types` with 13 `"conv"`, 5 `"sliding_attention"`, and 6 `"full_attention"` layers; `rope_parameters` sets `full_attention` (`rope_theta: 1000000.0`, `partial_rotary_factor: 0.5`) and `sliding_attention` (`rope_theta: 10000.0`, full rotary).
  - VL ([`LFM3-VL-2B/config.json`](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/main/config.json)): outer `"model_type": "lfm3_vl"`, `"architectures": ["Lfm3VlForConditionalGeneration"]`, nested `text_config.model_type: "lfm3"`, plus 14-layer hybrid conv/attention `vision_config`.
- **Size:** 1.73B text (~1.0 GB int4, ~3.5 GB BF16) and 2.0B VL (~1.3 GB int4) ([HF card](https://huggingface.co/LiquidAI/LFM3-1.7B), [Ollama `lfm3`](https://ollama.com/library/lfm3)). Fits both the 8 GB RTX 2070 SUPER and 16 GB M1 Pro, and is the new version of the shipped `lfm2` family ([`decoder/arch_lfm2.go`](../decoder/arch_lfm2.go)).
- **Existing `goinfer` adapter check:** **No.** [`decoder/registry.go`](../decoder/registry.go) registers `"lfm2"` and `"lfm2_vl"`, not `"lfm3"` or `"lfm3_vl"`, and `validateLFM2` in [`decoder/config.go`](../decoder/config.go) rejects any `layer_types` entry outside `"conv"` and `"full_attention"`.
- **New primitives needed:** **None (composition over shipped primitives + 1 program hook).** Per [`modeling_lfm3.py`](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/modeling_lfm3.py), LFM3's four deltas over `lfm2` ([`decoder/arch_lfm2.go`](../decoder/arch_lfm2.go)) already exist in `decoder`: (1) `"sliding_attention"` layers (`SlidingWindow: 256`, `IsGlobalLayer`); (2) per-layer-type RoPE theta and `partial_rotary_factor: 0.5` (`parseRopeParameters`, `RotaryDim`/`RotaryDimLocal` as in [`decoder/arch_laguna.go`](../decoder/arch_laguna.go) and [`decoder/arch_spark2_5.go`](../decoder/arch_spark2_5.go)); (3) head-wise sigmoid attention gate `gate_proj: Linear(2048, 16, bias=False)` (`AttnGate: GateSigmoid` as in [`decoder/arch_spark2_5.go`](../decoder/arch_spark2_5.go)); (4) `NormSandwich4` (`operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`), where [`decoder/program.go`](../decoder/program.go) (`a.isConvLayer(l)`) currently hardcodes `NormPre2` on conv layers and needs to emit `slotPostAttn`/`slotPostMLP` when `NormSandwich4` is active. (Like `lfm2`, `lfm3` stays CPU-only until a GPU backend implements `FeatShortConv`.)
- **Cost class:** **CHEAP** (days — adapter wiring, conv-layer `NormSandwich4` emission in [`decoder/program.go`](../decoder/program.go), and tensor-name mapping; VL vision encoder follows [`docs/multimodal.md`](multimodal.md)).

### 2. MiniCPM-OCR-1.0 and MiniCPM-OCR-Jev

- **Name:** `openbmb/MiniCPM-OCR-1.0` ([HF card](https://huggingface.co/openbmb/MiniCPM-OCR-1.0)) and `openbmb/MiniCPM-OCR-Jev` ([HF card](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)).
- **Maker:** OpenBMB ([openbmb org](https://huggingface.co/openbmb)).
- **Date:** 2026-10-28 ([OCR-1.0 card](https://huggingface.co/openbmb/MiniCPM-OCR-1.0), [OCR-Jev card](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)).
- **License:** Apache-2.0 ([OCR-1.0 README](https://huggingface.co/openbmb/MiniCPM-OCR-1.0/blob/main/README.md), [OCR-Jev README](https://huggingface.co/openbmb/MiniCPM-OCR-Jev/blob/main/README.md)).
- **`model_type` / `architectures` (`config.json`):**
  - `MiniCPM-OCR-1.0` ([`config.json`](https://huggingface.co/openbmb/MiniCPM-OCR-1.0/blob/main/config.json)): outer `"model_type": "minicpm_ocr"`, `"architectures": ["MiniCPMOcrForConditionalGeneration"]`; nested `text_config` has `"model_type": "qwen3_5_text"`, 24 layers (`hidden_size: 1024`, `intermediate_size: 3584`, `head_dim: 256`, `full_attention_interval: 4`, `linear_conv_kernel_dim: 4`, `linear_key_head_dim: 128`, `attn_output_gate: true`, 3:1 `"linear_attention"` / `"full_attention"` Gated DeltaNet hybrid); `vision_config` has `"model_type": "minicpm_ocr_vision"` (SigLIP + window-attention resampler with `insert_layer_id: 6`).
  - `MiniCPM-OCR-Jev` ([`config.json`](https://huggingface.co/openbmb/MiniCPM-OCR-Jev/blob/main/config.json)): outer `"model_type": "qwen3_5"`, `"architectures": ["Qwen3_5ForConditionalGeneration"]`, `"problem_type": "single_label_classification"`; nested `text_config` has `"model_type": "qwen3_5_text"`, 24 layers (`hidden_size: 2048`, `intermediate_size: 6144`, `head_dim: 256`, `full_attention_interval: 4`, `linear_conv_kernel_dim: 4`).
- **Size:** 1.3B (`MiniCPM-OCR-1.0`, ~0.8 GB int4 / ~2.6 GB BF16) and 2.0B (`MiniCPM-OCR-Jev`, ~1.2 GB int4 / ~4.0 GB BF16) ([OCR-1.0 card](https://huggingface.co/openbmb/MiniCPM-OCR-1.0), [OCR-Jev card](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)). Both fit the 8 GB RTX 2070 SUPER and 16 GB M1 Pro, and both text decoders are `qwen3_5_text` ([`decoder/arch_qwen3_5.go`](../decoder/arch_qwen3_5.go)).
- **Existing `goinfer` adapter check:**
  - `MiniCPM-OCR-1.0`: **Text backbone yes (with alias); full VLM no.** `loadConfig` ([`decoder/config.go`](../decoder/config.go)) flattens `text_config` and re-applies outer `"model_type": "minicpm_ocr"`, which is not in [`decoder/registry.go`](../decoder/registry.go). Aliasing `"minicpm_ocr"` to `qwen35DenseArchitecture` ([`decoder/arch_qwen3_5.go`](../decoder/arch_qwen3_5.go)) serves the text decoder.
  - `MiniCPM-OCR-Jev`: Outer `"model_type": "qwen3_5"` **is** registered ([`decoder/registry.go`](../decoder/registry.go)), **but** it replaces `lm_head` with a binary classification head (`problem_type: "single_label_classification"`) for OCR quality scoring (`0..1`), making it a trained-head verifier ([`docs/tasks/parked/task-constrained-confidence.md`](tasks/parked/task-constrained-confidence.md) Route B/C).
- **New primitives needed:** None for the `qwen3_5_text` decoder; `MiniCPM-OCR-1.0` needs the SigLIP + mid-layer window-attention resampler (`insert_layer_id: 6`, `4x` token compression) in `aikit` for vision-in OCR, and `MiniCPM-OCR-Jev` needs its binary classifier readout head.
- **Cost class:** **CHEAP** for text-decoder / verifier-head wiring on `qwen3_5` (days); vision encoder + resampler follows [`docs/multimodal.md`](multimodal.md).

### 3. SenseNova-Loop-9B-Preview

- **Name:** `sensenova/SenseNova-Loop-9B-Preview` ([HF card](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)).
- **Maker:** SenseTime / SenseNova ([sensenova org](https://huggingface.co/sensenova)).
- **Date:** 2026-10-30 ([HF card](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)).
- **License:** Apache-2.0 ([README](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/blob/main/README.md)).
- **`model_type` / `architectures` (`config.json`):** Outer `"model_type": "sensenova_loop"`, `"architectures": ["SenseNovaLoopForConditionalGeneration"]` ([`config.json`](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/blob/main/config.json)); `text_config` has 32 physical layers (`hidden_size: 4096`, `intermediate_size: 12288`, `num_attention_heads: 16`, `num_key_value_heads: 4`, `head_dim: 256`, `linear_conv_kernel_dim: 4`, `attn_output_gate: true`, `full_attention_interval: 4`, 3:1 `"linear_attention"` / `"full_attention"` Gated DeltaNet hybrid, `max_position_embeddings: 262144`) plus `loop_settings: {"counts": [2], "start": 4, "end": 28, "eval_count": 2}` (physical layers 4..27 execute twice in sequence for 56 effective layer steps).
- **Size:** 9B parameters (~5.5 GB int4, ~18 GB BF16) ([HF card](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)). Fits both the 16 GB M1 Pro and 8 GB RTX 2070 SUPER at int4, and 62 GB host RAM at BF16.
- **Existing `goinfer` adapter check:** **No.** `"sensenova_loop"` is not registered in [`decoder/registry.go`](../decoder/registry.go).
- **New primitives needed:** While per-layer mixers match `qwen3_5` ([`decoder/arch_qwen3_5.go`](../decoder/arch_qwen3_5.go)), [`modeling_sensenova_loop.py`](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview/blob/main/modeling_sensenova_loop.py) introduces two architectural primitives `goinfer` lacks:
  1. **Looped layer schedule with per-visit cache state:** 32 physical weight layers unroll to 56 execution steps (`loop_settings`), with distinct KV and Gated DeltaNet recurrent slots per loop pass — breaking the 1:1 physical-layer-to-cache-slot invariant in `kvgeometry.go`, `statebytes.go`, and the runners.
  2. **2-stream Manifold-Constrained Hyper-Connections (mHC):** Sinkhorn / log-space row-column doubly-stochastic mixing across two parallel residual streams in the looped block (the same hyper-connection primitive family noted on DeepSeek V4-Flash and GLM-5.3-Flash in this doc).
- **Cost class:** **EXPENSIVE** (weeks — decoupled weight-layer vs. state-slot indexing + 2-stream mHC residual mixer).

### 4. Nanbeige4.3-33B / Nanbeige4.3-33B-Base

- **Name:** `Nanbeige/Nanbeige4.3-33B` ([HF card](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)) and `Nanbeige/Nanbeige4.3-33B-Base` ([HF card](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-Base)).
- **Maker:** Nanbeige LLM Lab / BOSS Zhipin ([Nanbeige org](https://huggingface.co/Nanbeige)).
- **Date:** 2026-10-25/26 ([HF card](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)).
- **License:** Apache-2.0 ([README](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/blob/main/README.md)).
- **`model_type` / `architectures` (`config.json`):** `"model_type": "llama"`, `"architectures": ["LlamaForCausalLM"]` ([`config.json`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B/blob/main/config.json)); 52 layers, `hidden_size: 7168`, `intermediate_size: 22016`, `num_attention_heads: 56`, `num_key_value_heads: 8` (`head_dim: 128`), `vocab_size: 166912`, `rms_norm_eps: 1e-05`, `rope_theta: 120000000`, `max_position_embeddings: 524288`, `attention_bias: false`, `mlp_bias: false`, `hidden_act: "silu"`.
- **Size:** 33B dense (~17 GB int4, ~66 GB BF16) ([HF card](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)). Above the 8 GB card and tight on the 16 GB M1 Pro at int4 (fits the 62 GB Linux host at int4), **and** passes as a new checkpoint on a family `goinfer` already runs (`"llama"`).
- **Existing `goinfer` adapter check:** **Yes.** Directly served by `"llama": llamaArchitecture` ([`decoder/arch_llama.go`](../decoder/arch_llama.go), [`decoder/registry.go`](../decoder/registry.go)).
- **New primitives needed:** **None** (verify chat template and `eos_token_id: [2, 166101]` at Gate 0).
- **Cost class:** **CHEAP** (days — zero-code checkpoint swap on `llama`).

---

## Inspected releases in window filtered out by scope

- **`DiffuSpace/Clover-35B-A3B-Preview`** (DiffuSpace, 2026-10-28, Apache-2.0; [`config.json`](https://huggingface.co/DiffuSpace/Clover-35B-A3B-Preview/blob/main/config.json)): `"model_type": "clover_moe"`, `"architectures": ["CloverMoeModelLM"]`, 40 layers, 256 experts top-8 + 1 shared (35B total / 3B active, converted from `Qwen3.6-35B-A3B`). Per [`modeling_clover_moe.py`](https://huggingface.co/DiffuSpace/Clover-35B-A3B-Preview/blob/main/modeling_clover_moe.py) and [`llama.cpp` PR #31306](https://github.com/ggml-org/llama.cpp/pull/31306), it is a **block-diffusion LM** — excluded by this doc ("Explicitly out of scope: Diffusion LMs").
- **`tencent/Hy-SimulMT-30B-A3B`** (Tencent Hunyuan, 2026-10-26, Apache-2.0; [`config.json`](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B/blob/main/config.json)): `"model_type": "hy_v3"`, `"architectures": ["HYV3SpeechForCausalLM"]`. Speech/voice-cloning simultaneous translation — audio/speech output is out of scope (this doc).
- **`LiquidAI/LFM3-Audio-2B` & `LiquidAI/LFM3-Omni-2B`** (Liquid AI, 2026-10-27, `lfm1.0`; [Audio](https://huggingface.co/LiquidAI/LFM3-Audio-2B), [Omni](https://huggingface.co/LiquidAI/LFM3-Omni-2B), [Ollama `lfm3-audio`](https://ollama.com/library/lfm3-audio), [Ollama `lfm3-omni`](https://ollama.com/library/lfm3-omni)) and **`MayaResearch/Maya-2-Native`** ([HF card](https://huggingface.co/MayaResearch/Maya-2-Native)): speech/audio generation — out of scope (this doc).

---

## Changes to items already in "Next up" and "Watching"

1. **Mistral Large 4 ("Le Chonk", `mistralai/Mistral-Large-4-1T-A52B`) — weights posted 2026-10-29; refutes the `mistral4` alias hypothesis ("Next up" #3 & 2026-10-26 check):**
   - **Update:** Open weights dropped 2026-10-29 at [`mistralai/Mistral-Large-4-1T-A52B`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) ([`config.json`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/main/config.json), [Ollama `mistral-large-4`](https://ollama.com/library/mistral-large-4)) under the **Mistral Ecosystem Open-Weight License** (MIT-derived with a MaaS restriction, not Apache-2.0).
   - **Config check:** Outer `"model_type": "mistral_large4"` (`"architectures": ["MistralLarge4ForConditionalGeneration"]`) and `text_config.model_type: "mistral_large4"` (`"architectures": ["MistralLarge4ForCausalLM"]`). Unlike Mistral Small 4 (`119B-A6.5B`, outer `"mistral3"` / inner `"mistral4"`), Large 4 **does not** hit the `"mistral3"` outer-key collision in [`decoder/arch_mistral3.go`](../decoder/arch_mistral3.go); `resolveArchitecture` cleanly rejects `"mistral_large4"`. Architecturally it is **not** plain `mistral4`: its 72-layer `text_config` (`hidden_size: 6144`, `kv_lora_rank: 512`, `q_lora_rank: null`, 128 routed experts top-4 + 1 shared) adds **DeepSeek Sparse Attention (DSA)** with a 32-head (`head_dim: 128`) top-2048 token-selection `indexer` (`"full"` and `"shared"` indexer modes after 17 dense layers, NoPE every 4th layer), **cross-layer KV sharing** (every 4th layer from layer 39 reuses layer 35's KV latent), and **LeRoPE** + YaRN to 1M context ([`config.json`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/main/config.json)). Moves to **Out by rule 2** (1.05T total / 52B active), and item 3's `mistral3` -> `mistral4` guard only needs to cover Mistral Small 4.
2. **Ternary Bonsai 2 (`PrismML/Ternary-Bonsai-2-*`) — 2026-10-26 `Q2_0` guard question resolved ("Next up" #2 & 2026-10-26 check):**
   - **Update:** PrismML ships three GGUF packings (`PTQ1_0` ~1.75–1.83 bpw, `PQ2_0` ~2.13–2.20 bpw, and standard `Q2_0` ~2.28–2.33 bpw; [`PrismML/Bonsai-2-27B-GGUF`](https://huggingface.co/PrismML/Bonsai-2-27B-GGUF), [prismml.com](https://prismml.com)). Even standard `Q2_0` files output gibberish on stock `llama.cpp` prior to [`b11556`](https://github.com/ggml-org/llama.cpp/releases/tag/b11556) because Bonsai-2 folds a **blockwise Hadamard rotation** into the weights that requires a matching runtime Hadamard activation transform on every forward pass. Item 2's GGUF guard must decline Bonsai-2 / Hadamard-rotated GGUFs via metadata, not only `TQ1_0`/`PQ2_0` tensor types.
3. **Nemotron 3.5 Bolt E4B (`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`) — Gate 0 config and license verified (2026-10-26 check):**
   - **Update:** [`config.json`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/blob/main/config.json) (and [`Bolt-E4B-4GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB), featured in NVIDIA's 2026-10-30 [Halloween post](https://huggingface.co/blog/nvidia/nemotron-halloween)) has outer `"model_type": "gemma4"`, `"architectures": ["Gemma4ForConditionalGeneration"]`, and `text_config` (`hidden_size: 2560`, `num_hidden_layers: 42`, `enable_moe_block: false`) under **NVIDIA OpenMDW-1.1** — confirming it resolves directly through `"gemma4": gemma4Architecture` ([`decoder/arch_gemma4.go`](../decoder/arch_gemma4.go)) as a **CHEAP** checkpoint swap.
4. **Decision models (`d1` and `d3`) — new 2B `d1` on LFM3 and `d3` on Qwen3.5 (2026-10-26 check):**
   - **Update:** On 2026-10-27 Liquid AI posted smaller 2B decision models [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B/blob/main/config.json) (`"model_type": "lfm3_vl"`, `"architectures": ["D1Model"]`) and [`LiquidAI/d1-Omni-2B`](https://huggingface.co/LiquidAI/d1-Omni-2B/blob/main/config.json) ([Ollama `d1`](https://ollama.com/library/d1), [Ollama v0.40.12](https://github.com/ollama/ollama/releases)), moving `d1` from 3B `lfm2` to 2B `lfm3_vl` + `D1Model` head. `vllm-sr` posted `d3-nano`/`mini`/`flash`/`ultra` (2B/4B/9B/27B; [`vllm-sr`](https://huggingface.co/vllm-sr), [Ollama `d3`](https://ollama.com/library/d3), [Ollama v0.40.11](https://github.com/ollama/ollama/releases)) with `"model_type": "qwen3_5"` (`Qwen3_5Model`, served by `qwen35DenseArchitecture` in [`decoder/arch_qwen3_5.go`](../decoder/arch_qwen3_5.go)).
5. **Watching & Out-by-rule-2 items (2026-10-26 – 2026-11-03):**
   - **Kimi K3 / Kimi K3.1 Pro:** Moonshot announced **Kimi K3.1 Pro** (same 2.8T total / 104B active `kimi_k3` hybrid: 69 KDA + 24 gated-MLA layers, 896 routed + 2 shared LatentMoE experts top-16; [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3), [kimi.ai](https://kimi.ai)) with weights scheduled for 2026-11-04. Still out by rule 1 (until posted) and rule 2 (size), with no smaller KDA+MLA checkpoint released.
   - **DeepSeek V4.1 Flash:** NVIDIA posted [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) on 2026-10-30; still 552B/763B causal encoder-decoder MoE, out by rule 2 with no Lite checkpoint posted.
   - **GLM-5.x (`zai-org`), Qwen 4 (`Qwen`), Muse Glimmer 30B (`meta-models`), Xiaomi MiMo-V3 (`XiaomiMiMo`):** No change — `GLM-5.3-FlashX` is an API tier of the 320B-A18B [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (still no Air-class checkpoint); Qwen 4 remains in training with no smaller open-weight carrier after [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) has no smaller size; [`XiaomiMiMo`](https://huggingface.co/XiaomiMiMo) MiMo-V3 (HySparse2) remains paper-only.

## Watching — current list

- **Kimi K3** — reportedly ~3–4T params; *not* a K2.x point release (K2 through K2.7-Code are
  DeepSeek-V3-shaped and already run as `kimi_k2`). Phase-0 scoping
  ([`completed/task-model-family-deepseek-v4-kimi-k3.md`](completed/task-model-family-deepseek-v4-kimi-k3.md))
  found MLA on only 24 of 93 layers, the other 69 a KDA mixer — which `bailing_hybrid` now
  implements — plus a latent-MoE wrapper, a new activation, and MLA-with-NoPE + output gate. Weights
  ship `mxfp4-pack-quantized` in a `compressed-tensors` layout not yet confirmed to match the
  MXFP4 packing goinfer reads (that reader moved to aikit in 7a8e38150). Same hardware wall as V4-Flash (1.56 TB). Re-check when weights drop.
- **GLM-5.x** — still no Air-class size. GLM-5.3-Flash (weights posted, MIT) turned out to be
  320B-A18B with a sparse + linear attention hybrid and Manifold-Constrained Hyper-Connections,
  not the Air-class model this entry was waiting for. Shares the native-sparse-attention
  direction with V4-Flash; if the DSA path is ever built, evaluate against it.
- **MiniMax M3** — 428B MoE, ~22B active, natively multimodal. Too big for the niche, and the
  multimodal training makes text-only support less compelling than the size alone suggests.
- **Qwen3.8-Flash-Next (Qwen4 architecture preview)** — `qwen4_exp`, 125B on disk / ~6B active:
  Gated DeltaNet + a new sparse-attention mixer (QSA) + a 4-branch gated residual + an n-gram
  embedding + native MTP. Doesn't fit either rig. Scoped in
  [`scoping-qwen38-flash-next.md`](scoping-qwen38-flash-next.md). As of 2026-10-15 Alibaba says
  Qwen 4 is "in training" with no sizes or date; no smaller carrier of the architecture exists.
  Qwen3.8-27B remains the newest open-weight Qwen that fits.
- **Mistral Medium 3.5 (128B dense) — config check done 2026-10-15.** Outer `mistral3`, text
  `ministral3`: 88 layers, 96 query / 8 KV heads, head_dim 128, no sliding window, YaRN
  (factor 64, `llama_4_scaling_beta` 0). That is exactly what
  `decoder/arch_mistral3.go:ministral3Architecture` already expresses — a config alias, no new
  forward. Two blockers, neither architectural: the safetensors ship per-tensor static fp8
  (`weight_block_size` null), which `decoder/fp8.go:fp8Reconstruct` does not read (it needs
  block-wise `weight_scale_inv`), so GGUF is the load route; and 128B dense fails rule 2 except at
  Q2_K–Q3_K (~45–55 GB) on the Linux box. Parked on hardware, not on code.
- **Out by rule 2 (released this period, weights posted, far past both rigs):** Meta's Muse
  Spark Open (816B-A42B, 2026-10-15; GQA + attention output gate + learned residual gate,
  BF16 and NVFP4), DeepSeek V4.1 Flash (552B; a causal encoder-decoder, so *not* a V4-Flash
  delta), Xiaomi MiMo-V2.6 Flash (309B-A15B) and Pro (~1T), and inclusionAI's Ling 3.1 Flash
  (560B-A25B, MIT; architecture not checked against `bailing_hybrid`).
- **Qwen3.8-Max** — no open-weight repo found on 2026-10-15; treat as API-only until one appears.
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
  itself is a plumbing change one of the three could unlock. (Muse Spark Open's learned
  residual gate is a fourth gated-residual design; whether it is the same primitive was not
  checked.)

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
   entry (`completed/task-parity-coverage.md`, `completed/task-capability-matrix.md`), or the support
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

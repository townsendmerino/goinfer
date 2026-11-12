# Task: Recursive's Flint (40B-A2.5B MoE) on goinfer (F0–F5) — 2026-10

> **Status 2026-11-11: All items (F0, F1, F1b, F2, F3, F4, F5) complete on Mac (16 GiB M1 Pro) and `nobara` (8 GiB RTX 2070 SUPER, 62.7 GiB RAM).**
> Flint is Recursive's first model: a 39.8B-total, ~2.5B-active decoder-only MoE for local coding agents, trained from
> scratch, Apache-2.0, released 2026-10-21 (`RecursiveSI/flint`, pinned at `2a247be0c0aec895ccb0ac7ca92229ec6df37288`).
> Every one of its layer types is already in goinfer in some family; what is new is the combination, the standalone
> attention sink (`FeatAttnSink` split), the 3-step `flint` pre-tokenizer (`\p{N}{1,3}` + CJK/Kana + `o200k`), and the
> chat template (`<|system|>`, `<|developer|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`, `<think>` preserved across
> all historical assistant turns, and GLM-4.7-style XML tool calls). Both upstream `transformers` (`modeling_flint.py`,
> PR `#49824`) and `llama.cpp` mainline (`src/models/flint.cpp`, PR `#30928`) now exist.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| F0 | done | desk read complete (F0.1–F0.5); zero `config.json` contradictions with the card; owner confirmed F0.3 parity tier and ordered F1b after F1–F3 | `docs/measurements/flint-f0-2026-10.md` |
| F1 | done | CPU/CUDA/Metal family descriptor (`decoder/arch_flint.go:flintArchitecture`), standalone `Architecture.AttnSinks` split from `FeatAttnSink`, GGUF `flint` config/loader (`decoder/gguf.go:ggufFlintConfig`), `splitFlint` pre-tokenizer (`tokenizer/splitshape.go:shapeFlint`, `tokenizer/split_o200k.go:splitFlint`, `tokenizer/gguf.go:byteLevelKnobs`), and `Flint()` chat template + tool renderer/parser (`chat/templates.go:Flint`, `chat/chat.go:Detect`, `chat/tools.go:parseFlintTools`) | `decoder/flint_parity_test.go`, `tokenizer/split_o200k_test.go`, `chat/streamable_test.go` |
| F1b | done | safetensors NVFP4 (`decoder/nvfp4.go:nvfp4Reconstruct`), MLX quantized safetensors (`affine` 4/8-bit, `nvfp4`, `mxfp4`, quantized `embed_tokens`/`lm_head`, and 3-D stacked `mlp.switch_mlp.{gate,up,down}_proj` in `decoder/mlx_safetensors.go`), GGUF `<name>.scale` companion multiplication (`decoder/gguf.go:buildWeightsFromGGUF`), and `decoder/fit/weights.go:SafetensorsWeightBytes` pricing (GGUF `Q4_1`/`NVFP4` block dequantizers skipped per owner scope decision as `.giw` is the primary runtime path) | `decoder/nvfp4_mlx_test.go` (`TestNVFP4_e2m1TableAndExhaustiveE4M3`, `TestNVFP4_reconstructAndEndToEndFlint`, `TestMLX_affineAndFP4ModesAndEndToEndFlint`, `TestFit_nvfp4AndMLXSafetensorsWeightBytes`) |
| F2 | done | Tier 1 tiny-oracle parity (`TestFlint_tinyForwardParity`: `maxAbs=3.576e-07`, `cosine=1.00000000`, greedy continuation exact match, `.giw` round-trip `maxAbs < 1e-6`), `sliding_window=513` boundary golden (`TestFlint_windowBoundaryParity` at `len=512, 513, 514, 1100`: `maxAbs <= 7.600e-07`), 4 break-it-first checks (`TestFlint_breakItFirst`), resident + expert-cache parity (`TestFlintResidentParityCUDA`, `TestFlintResidentParityMetal`), and Tier 2 real-checkpoint gate (`TestFlint_realCheckpointParity`) verified on both Mac (`Q2_K` `.giw`, `argmax=220`) and `nobara` (`Flint-Q4_K_M` `.giw`, `argmax=13`) | `decoder/flint_parity_test.go`, `backend/cuda/flint_resident_parity_test.go`, `backend/metal/flint_resident_parity_test.go`, `testdata/flint-tiny`, `testdata/flint_tiny_golden.json`, `testdata/parity/flint.json` |
| F3 | done | **Mac 16 GiB** (`RecursiveSI.flint.f16.gguf.Q2_K.int4.metal.v17.giw`, `21,689 MB`): `chat fit` matches F0.5 (`metal EXPERT-CACHED @ int4`: `dense 1.35 GB + 106/256 experts 8.21 GB + KV@4096 0.39 GB`); Metal `--moe-cache-experts` loads in `5.2s` and decodes at **`4.0 tok/s`** vs CPU `--stream-weights` **`1.6 tok/s`** (`2.5×`). **`nobara` 8 GiB RTX 2070 SUPER / 62.7 GiB RAM** (`Flint-Q4_K_M.int4.e4h.cuda.v17.giw`, `21,431 MB`): fits `-ctx 16384` (`48–54/256` slots), `-ctx 40960` (`36/256` slots, `3.2 GB`), and `-ctx 65536` (`20/256` slots, `1.8 GB`); vs `llama.cpp` `b988-a37cc46` (`-ngl 99 -ot ".ffn_.*_exps.=CPU"`, `pp64=52.7`, `pp512=237.6`, `tg64=33.0 tok/s`), `goinfer` CUDA (`--moe-cache-experts`, `54` slots, `ctx=16384`) gets `pp64=38.7 tok/s` (`0.73×`), `pp512=123.3 tok/s` (`0.52×`), and **`tg64=33.9 tok/s` cold (`1.03×`), `44.1–49.9 tok/s` warm (`1.34×–1.51×`)**, `30.8 tok/s` after `pp512` (`0.94×`); CPU resident `.giw` loads in `254ms` and decodes at `5.2 tok/s` | `decoder/capability_matrix_test.go`, `docs/capability-matrix.md`, `docs/hardware-matrix.md`, `docs/measurements/flint-f0-2026-10.md` §6.3 |
| F4 | done | replayed all 84 requests (29 B-shaped fixtures, rollback `9..271` tokens, median `31`) from `docs/measurements/kv0h-2026-10/study-h20-stats-h.json` against Flint (`sliding_window=513`, 35 sliding + 14 full layers): Metal resident retains full context (`29/29` full reuse, `0` tokens short); CUDA resident `--windowed-kv` (`cap=1025` + `576` extra = `1601`, `step=64`, `rollback floor=448 >= 271`) retains `29/29` full reuse (`0` tokens short); CPU modular ring (`513` slots) drops to `0/29` once wrapped past 513 | `decoder/flint_parity_test.go:TestFlint_kv0hBShapedWindowReuse`, `docs/measurements/flint-f0-2026-10.md` §6.4 |
| F5 | done | full 11-harness I4 first-request census tokenized against Flint (`codex 0.162.1`: `5,831` tok → `8,192`; `opencode 1.19.0`: `6,937` tok → `16,384`; `Claude Code`: `1,009` bare / `10,602` 12-tool / `19,234–20,331` 23-tool → `32,768`, or `40,960` for KV0-H headroom); added `flint-16k` and `flint-40k-cc` to `plan-i3.json` and verified live `bwrap` sandbox `read` cells on `nobara` (`-ctx 40960 --moe-cache-experts`): **`codex` PASS** (`71.7s`, turn 2 `5.7s`), **`opencode` PASS** (`106.5s`, turn 2 `5.6s`), and **`claude-code` PASS** (`183.0s`, `0` compactions, turns 2–4 reuse the `17.1K` prefix in **`1.6s–2.4s`** via windowed KV) | `docs/measurements/integration-harness-2026-10/matrix/plan-i3.json`, `docs/measurements/flint-f0-2026-10.md` §6.5 |

## 0. What Flint is (model card and launch coverage, read 2026-10-21; to be confirmed against `config.json` in F0)

- **Shape.** 49 layers: 2 dense, then 47 MoE. Each MoE layer: 256 routed experts, 6 chosen per token, plus one
  always-active shared expert. Router type not stated.
- **Attention.** 35 sliding-window layers and 14 full-attention layers. GQA with 32 query heads, 4 KV heads, head dim 128.
  Per-head QK RMSNorm and **learned per-head attention sinks**. The window is **513 tokens: the current token plus the
  512 before it**. RoPE on the sliding layers, **NoPE on the full-attention layers**.
- **Context.** 262,144 tokens on the card (the launch post says up to 512k; Recursive recommends staying within 256k).
- **Tokenizer.** Vocabulary of 150,144; family not stated. The card says to "use the checkpoint's chat template and
  preserve returned `reasoning_content` in conversation history". Tool calls are OpenAI-compatible.
- **Files.** BF16 safetensors, ~80 GB. Quantizations are separate repos: an INT4 MLX build (~22.6 GB), an NVFP4 GGUF
  (22.9 GB), and third-party quantizations the card links for llama.cpp, Ollama and LM Studio.
- **Runtimes named at launch.** llama.cpp, Ollama, LM Studio, Unsloth, vLLM, SGLang (a kit under `sglang/` in the repo),
  and Recursive's own `rec` engine (MLX, Apple Silicon; it targets 48 GB M4 Pro / M5 Pro / M5 Max).
- **Claims (self-reported).** SWE-bench Verified 74.2%, Terminal-Bench 2.1 54.9%, LiveCodeBench v6 80.3%, IFEval 95.1%.
  Recommended sampling: temperature 1.0, top-p 0.95, top-k 20.

## 1. What goinfer already has

| Flint's piece | Where goinfer has it | Note |
|---|---|---|
| Sliding-window attention interleaved with full attention | gpt-oss, mellum, gemma4, cohere2 (`FeatSlidingWindow`) | the 513 convention is F0's question |
| Learned per-head attention sinks | gpt-oss (`decoder/forward_gptoss.go`, `FeatAttnSink`) | **`FeatAttnSink` is defined as gpt-oss's bundle: "softmax sink, clamped-SwiGLU expert, bias-weighted router". Flint needs the sink on its own unless F0 finds it shares the rest** |
| NoPE on the full layers, RoPE on the sliding layers | cohere2 (`decoder/registry.go`, `layerNoPE: isGlobal`), smollm3, llama4_text (`FeatNoPE`) | cohere2 is exactly this split |
| Per-head QK RMSNorm | Qwen3, GLM, Mellum (`FeatQKNorm`) | — |
| Sparse MoE with a shared expert | laguna, llama4_text (`FeatMoE`; `FeatMoEGatedShared` if the shared expert is sigmoid-gated) | 256 experts, top 6: new settings, not new operations |
| Dense lead layers before the MoE layers | gemma4 ("dense ‖ sparse"), laguna | — |
| Routed experts kept in host memory | `--moe-cache-experts` (CUDA), `--moe-pager` (Darwin pool mode) | the path for the 8 GB card and a Mac under 48 GB |

The OP3 layer program (each family walks a program of registered operations) is how this should be built: a new
descriptor that composes existing operations, not a new forward.

## 2. What is new or risky

1. **Reference implementation (resolved in F0.3).** At release on 2026-10-21 no Transformers reference existed; as of
   F0 (2026-11-11), both upstream `huggingface/transformers` (`modeling_flint.py`, PR `#49824`) and `ggml-org/llama.cpp`
   mainline (`src/models/flint.cpp`, PR `#30928`) are merged and pinned in `docs/measurements/flint-f0-2026-10.md` §3.
2. **The window convention.** "513 = current + 512" (`pos - j < 513`) matches `decoder/attention.go:attendQueryP`
   (`j < pos-window+1`) exactly. Off by one, the output drifts with no error. F2 tests prompt lengths across the boundary.
3. **The sink alone.** `FeatAttnSink` couples the sink with gpt-oss's clamped SwiGLU and router. Splitting it touches
   the CUDA and Metal feature declarations (`decoder/features.go`) and gpt-oss's own gates, which must stay unchanged.
   Also, `decoder/gguf.go:buildWeightsFromGGUF` currently gates `exp_probs_b.bias` on `if arch.MoE.RouterSigmoid`; Flint
   has `RouterSigmoid: true` and `HasRouterBias: false`, so that check must use `arch.MoE.HasRouterBias`.
4. **Reuse across turns with a 513 window.** KV0-H (`docs/measurements/kv0h-hybrid-traffic-2026-10.md` §2.3) found that
   harnesses re-send the previous reply differently, so a follow-up diverges about 200 tokens before the end of what the
   slot holds (median 201 re-processed). With a 513-token window, reuse to that point needs the sliding layers' rows from
   before it. `Model.residentReuseFloor` serves a prefix whose window was dropped as 0, and the whole prompt is prefilled.
   F4 measures whether that happens on Flint.
5. **Formats and disk.** Standard GGUFs (`ggml-org/Flint-GGUF`: `Flint-Q4_K_M.gguf` at 23.39 GB, `Flint-Q8_0.gguf` at
   42.35 GB, `Flint-BF16.gguf` at 79.63 GB; plus `unsloth/flint-GGUF`) use types (`Q4_K`, `Q8_0`, `BF16`, `Q5_K`, `Q6_K`,
   `IQ3_S`, `IQ4_XS`) that `aikit/embed` already reads today and that run natively on `goinfer`'s `q4k` lane. Recursive's
   own three quantized repos (`RecursiveSI/flint-int4-gguf` using `GGML_TYPE_Q4_1 = 3`, `RecursiveSI/flint-nvfp4-gguf`
   using `GGML_TYPE_NVFP4 = 40` + companion `.scale` `F32` tensors, and `RecursiveSI/flint-int4` using MLX affine-INT4
   safetensors) are scoped as optional read-only loader work in **F1b**, coordinated with
   `docs/tasks/task-lowbit-formats-2026-11.md` (LB0c, LB1b, LB2b, LB5c). Every timed run reads from `~/models` on the box
   doing it, never the archive (`CLAUDE.md`); `nobara` has 286 GiB free on `~/models`, while the Mac (`24 GiB` free) needs
   ~25–30 GiB freed before staging a quantized checkpoint and its `.giw` sidecar.

## 3. Items

### F0 — the desk read (no weights beyond configs; minutes) — **DONE** (`docs/measurements/flint-f0-2026-10.md`)

Pin the repository and revision. Record into `docs/measurements/flint-f0-2026-10.md`, each item a stop point:

1. **`config.json` and the architecture name.** Layer types and their order, the dense-layer positions, the window field
   and its convention, the sink parameter's name and shape, the shared expert (gated or not), the router (softmax or
   sigmoid, normalization, bias), RoPE theta and any scaling on the sliding layers, tied embeddings. **Any field that
   contradicts the card → record it and stop for the owner.** *(Confirmed: zero contradictions.)*
2. **Tokenizer and chat template.** Tokenizer family and pre-tokenizer; the template's handling of `reasoning_content`
   and of tool calls, rendered through goinfer's template path for a two-turn tool conversation, compared with the
   reference rendering token for token. *(Recorded in `docs/measurements/flint-f0-2026-10.md` §2.)*
3. **The reference.** In this order of preference, record which exists today:
   (a) Transformers modeling code (`modeling_flint.py`, PR `#49824`, pinned by SHA-256);
   (b) the SGLang kit's or vLLM's modeling file (`vllm/model_executor/models/flint.py`, pinned by SHA-256);
   (c) llama.cpp mainline support for the architecture (`src/models/flint.cpp`, PR `#30928`, pinned by SHA-256).
   Propose the parity tier each allows and **stop for the owner's choice.** *(Confirmed by owner: Tier 1 against `transformers` PR `#49824` + Tier 2 real-checkpoint gate.)*
4. **Quantizations and disk.** Which published quantizations goinfer's loaders read, how NVFP4 and MLX fit in (`F1b`),
   their sizes, and free space in `~/models` on nobara and the Mac. *(Recorded in `docs/measurements/flint-f0-2026-10.md` §4.)*
5. **Fit, estimated.** For each box: weights resident versus routed experts in host memory, KV at 64K and 262K (only the
   14 full layers grow; the 35 sliding layers hold 513 rows each), and the host RAM the expert cache needs (the `.giw`
   fit guard prices it). Record the Mac's memory and nobara's RAM. *(Recorded in `docs/measurements/flint-f0-2026-10.md` §5.)*

### F1 — the family (CPU first)

A descriptor composing existing operations: layer types from the config, 2 dense lead layers, MoE 256/6 plus the shared
expert, NoPE on full layers and RoPE on sliding ones, QK norm, the sink. The sink split from `FeatAttnSink` happens here,
with gpt-oss's existing gates unchanged as the guard, plus changing `if arch.MoE.RouterSigmoid` to
`if arch.MoE.HasRouterBias` for `exp_probs_b.bias` in `decoder/gguf.go:buildWeightsFromGGUF`. Tokenizer (`splitFlint` in
`tokenizer/bytelevel.go:splitRegex` and `tokenizer/gguf.go:byteLevelKnobs`) and template support (`TmplFlint` in
`chat/chat.go:Detect`) from F0.2.

### F1b — optional read-only format admission for Recursive's official quantized repos

Coordinated with `docs/tasks/task-lowbit-formats-2026-11.md` so `goinfer` can load all three official `RecursiveSI`
quantized checkpoints directly (dequantizing to `f32` at load/pack time into `int4`/`int8` or `.giw`, without new GPU
kernels):
1. **`GGML_TYPE_Q4_1` (`type = 3`, LB5c)** in `aikit/embed`: ~25-line block dequantizer (`32` nibbles + `FP16 d` +
   `FP16 m`, `20 B/block`), admitting `RecursiveSI/flint-int4-gguf` (`flint-int4-q4_1.gguf`).
2. **`GGML_TYPE_NVFP4` (`type = 40`, LB0c / LB1b / LB2b)**: `block_nvfp4` (`64` values in `36 B`: four `UE4M3` sub-block
   scales + `32 B` `E2M1` nibbles in `j / j+8` sub-block order) in `aikit/embed`, plus companion `<name>.scale` `F32`
   tensor multiplication (`[1]` for 2D tensors, `[256]` per-expert for 3D `ffn_*_exps.weight`) in
   `decoder/gguf.go:buildWeightsFromGGUF`, admitting `RecursiveSI/flint-nvfp4-gguf` (`flint-nvfp4.gguf`).
3. **MLX affine-INT4 safetensors (`quantization.group_size = 64, bits = 4, mode = "affine"`)**: dequantizing packed
   `U32` weights (`8` nibbles per `uint32`) with sibling `.scales` (`BF16`) and `.biases` (`BF16`:
   `w = q * scale + bias`) and pricing them in `decoder/fit/weights.go:SafetensorsWeightBytes`, admitting
   `RecursiveSI/flint-int4` (which is bit-for-bit identical to `flint-int4-q4_1.gguf`).

### F2 — parity

The tier the owner chose in F0.3. Add a window-boundary golden: prompts of 512, 513, 514 and about 1,100 tokens,
compared position by position against the reference (the shape of Mellum 2.1's follow-up A,
`docs/completed/task-mellum21-2026-10.md`). Add the sink and NoPE layers to the break-it-first list: perturb each and
confirm the gate catches it.

### F3 — backends and fit

CUDA on nobara with `--moe-cache-experts` (resident attention, dense and shared parts; routed experts in page-locked host
memory). Metal on the Mac, resident if it fits, otherwise the MoE pager. **Registered before running:** the fit
measured against F0.5's estimate; decode and prefill tok/s against llama.cpp mainline (or Ollama) on the same box, same
quantization class. Bars set at registration, house style (the ratio and its noise).

### F4 — reuse across turns with a 513-token window

B-shaped fixtures from KV0-H (a conversation whose next prompt re-renders the previous reply, diverging 15 to 123
tokens into it): record the reuse applied (`reused=`) against the divergence point. **Rule, registered first:** if reuse
stops more than one window short of the divergence on most fixtures, that is a finding for the windowed-KV plan
(Mellum 2.1 follow-up C, "worth building") and KV4/KV10 in `docs/tasks/task-kv-architecture-2026-10.md`, not a Flint
bug. Record which backends retain enough rows.

### F5 — the harnesses

Add Flint rows to I3's real-harness matrix (`docs/tasks/task-integration-harness-2026-10.md`): opencode, Codex and
Claude Code. Flint's small KV may give Claude Code room where Qwen3.5-9B was compaction-bound at 32,768 on the 8 GB card
(KV0-H A5). Run I4's first-request census at the context F3 found.

## 4. Not in scope

- Native GPU kernels (kept resident lanes) for NVFP4 or MLX affine-INT4 (read-only dequant admission is in **F1b** and
  `docs/tasks/task-lowbit-formats-2026-11.md`; native `q4k` already covers `Flint-Q4_K_M.gguf`).
- Recursive's `rec` engine as anything but a reference point for speed on a Mac.
- Fine-tuning, or Flint as a drafter.

## 5. Order and size (estimates)

F0 (done, desk) → owner's reference choice → F1 + F2 (M, CPU) → F3 (M per backend; CUDA first) → F4 (S) → F5 (S, once
F3 passes); F1b (S–M, optional read-only loaders, independent of F1–F3).

## 6. Conventions

Cite code by declaration name and docs by path and heading. Checkpoints from `~/models` on the box doing the run. Pin any
reference module by sha256. Estimate before any run; runs go by day unless they genuinely need the night. Stage explicit
paths only, on a branch; readings stay unmerged for the owner's review.

## Sources

- [RecursiveSI/flint model card](https://huggingface.co/RecursiveSI/flint)
- [MarkTechPost: Recursive releases Flint](https://www.marktechpost.com/2026/10/21/recursive-flint-local-coding-model/)
- [TechCrunch: Recursive launches Flint](https://techcrunch.com/2026/10/21/recursive-launches-flint-a-laptop-coding-model-it-says-was-largely-built-by-ai/)
- [TestingCatalog: Recursive releases Flint](https://www.testingcatalog.com/recursive-releases-flint-40b-model-for-local-coding/)

<!-- doc-reviewed: 2026-11-11 -->

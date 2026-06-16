# goinfer architecture

A tour of how goinfer turns a prompt into tokens, and how the model gets from a
file (or the binary's own image) into RAM. Three diagrams: the **forward pass**,
the **load + memory paths**, and the **module map**.

> These diagrams are intentionally drawn at the *stage* level, not the
> struct-field level. goinfer is descriptor-driven — one generic decoder runs
> Gemma 3/4, Qwen 2.5/3 (+ Qwen2.5-VL), Llama, Mistral, GPT-2, Mellum/Mellum2, and
> the MoE families (Mixtral, Qwen-MoE, GLM-4.5/4.6, Granite-4.0-H) by reading an
> `Architecture` descriptor — so the per-model specifics (which norm, GQA ratio,
> RoPE scaling, tied vs separate head, **MoE routing variant**) are *config*, not
> separate code paths. Even the sparse-MoE FFN is mostly config: a router scores
> experts, the top-k run as gated MLPs, plus an optional always-on shared expert —
> one `moeMLP` covers Mixtral (softmax top-k), Qwen-MoE (sigmoid-gated shared), GLM
> (DeepSeek sigmoid routing + an `e_score_correction_bias` that steers selection +
> an ungated shared expert), and Granite (the same, fused experts). Drawing stages
> keeps these accurate across new families; the numeric contract is the parity gate
> against HuggingFace, not this doc.
>
> **The exceptions to "config, not code paths"** are the two **hybrid** families,
> which add genuinely new *sequence-mixing primitives* — descriptor-selected per
> layer, not config knobs — both running over a hybrid cache (KV for the softmax
> layers + a per-layer recurrent state):
>
> - **Qwen 3.6 / `qwen3_5_moe`**: most layers are **Gated DeltaNet** (linear
>   attention with a recurrent matrix state) — `deltanet.go`, state `deltaState`.
> - **Granite-4.0-H / `granitemoehybrid`** (and **Nemotron-H / `nemotron_h`**, a
>   single-op-per-block variant): a mix of **Mamba-2 selective state-space** layers
>   and softmax attention, MoE on every layer, plus four
>   Granite scalar multipliers (embedding / attention / residual / logits) —
>   `mamba2.go` (sequential scan + an equivalent chunked scan), state `mamba2State`
>   (`{conv window, SSM state}`), forward in `forward_granite.go` / `forward_nemotron.go`.
> - **DeepSeek-V2/V3 / `deepseek_v2` `deepseek_v3`**: **Multi-head Latent Attention**
>   — the third axis, *latent-KV*. K/V compress to a shared low-rank latent
>   (`kv_lora_rank`) which is the ONLY thing cached (a new **latent cache kind**,
>   `KVCache.mlaLatent`, alongside full-KV and recurrent state); per-head K/V are
>   reconstructed from it each step, with decoupled RoPE on a separate slice. The
>   per-head q·k width differs from the v width, so it runs its own attention rather
>   than the uniform path — `forward_deepseek.go`. MoE rides the shared `moeMLP`
>   (config-driven softmax-V2 vs sigmoid-group-V3 routing).
>
> Each has a dedicated forward and is
> excluded from multi-token batched prefill — their recurrence / latent
> reconstruction is inherently sequential. A new mixer (or attention-kind) is one new
> primitive + its parity test on the existing per-layer-kind scaffolding, which is
> what let Granite (Mamba-2) land on the shapes qwen3_5_moe (DeltaNet) first proved,
> and MLA's latent cache slot in beside them.

## 1. The forward pass (one decode step)

Each generated token runs the full stack once. Prefill runs the prompt tokens
through the same path (filling the KV cache) and only keeps logits for the last.

```mermaid
flowchart TB
  TOK["tokenizer · BPE / SentencePiece<br/>(id-exact vs HuggingFace)"] --> EMB["token embedding<br/>(+ optional embed scale, learned pos emb)"]
  EMB --> BLK

  subgraph BLK["decoder block × N  (Architecture descriptor)"]
    direction TB
    N1["norm · RMSNorm or LayerNorm"] --> ATT["sequence mixer (descriptor per layer)<br/>causal attention: RoPE · GQA · opt. sliding window · KV cache<br/>OR recurrent: Gated DeltaNet / Mamba-2 (hybrid families)"]
    ATT --> PA{"sandwich norm?"}
    PA -->|yes| N1b["post-attn norm"] --> R1(("+ residual"))
    PA -->|no| R1
    R1 --> N2["norm"] --> MLP["MLP · dense SwiGLU<br/>OR sparse MoE: router top-k experts + opt. shared expert"]
    MLP --> PB{"sandwich norm?"}
    PB -->|yes| N2b["post-mlp norm"] --> R2(("+ residual"))
    PB -->|no| R2
  end

  BLK --> FN["final norm"]
  FN --> HEAD["LM head<br/>tied embedding · or separate projection"]
  HEAD --> SC{"logit softcap?"}
  SC --> LOG["logits · [vocab]"]
  LOG --> PROC["LogitProcessor seam<br/>(constrain: JSON-grammar mask)"]
  PROC --> SMP["sampler · greedy / temp / top-k / top-p"]
  SMP --> NXT["next token id"]
  NXT -. "append to KV cache, feed back" .-> EMB
```

The **`LogitProcessor` seam** is where `constrain` lives: it masks each step's
logits *before* sampling, so a JSON grammar makes malformed output literally
unreachable — independent of model size.

## 2. Load + memory paths

The same resident weight bundle can be reached four ways. The differences are
purely *where the bytes come from* and *how much heap they cost* — the forward
pass above is identical afterward.

```mermaid
flowchart TB
  subgraph SRC["where the model comes from"]
    direction TB
    GIW["embedded .giw bundle<br/>(prequant build · default release)"]
    EGG["embedded raw GGUF<br/>(--gguf build)"]
    FILE["--model file.gguf"]
    TMP["--model-tmp / GOINFER_MODEL_TMP<br/>(stream to temp file)"]
  end

  GIW --> ALIAS["LoadSerializedWeights<br/>int8/int4 arrays ALIASED zero-copy from the image<br/>(scales/norms copied — alignment)<br/>magic+version+quant+CRC guard, lazy fallback"]
  EGG --> PARSE["LoadGGUFBytes · parse in place"]
  FILE --> MMAP["OpenGGUFMmap · mmap the file"]
  TMP --> MMAP
  PARSE --> REQ["dequant → requant to int8 resident<br/>(per-layer parallel)"]
  MMAP --> REQ

  ALIAS --> W["resident weightMats"]
  REQ --> W

  W --> DISP{"matmul dispatch<br/>(per weightMat precision · same kernel at every M)"}
  DISP -->|int4 decode + prefill| K4["MatmulBTW4A8 · CPU int4×int8 integer kernel<br/>(NEON/AVX2; every M, bit-identical decode==prefill)"]
  DISP -->|int8 W8A8| K8["MatmulBTW8A8 · CPU SDOT (fast default)"]
  DISP -->|int8| KQ["MatmulBTQ8 · CPU"]
  DISP -->|f32| BE["Backend.MatmulBT · pure-Go SIMD"]
  W --> GPUR["OR: full-residency GPU forward (-tags gpu, webgpu)<br/>whole token on the GPU via DecodeRunner<br/>W8A8 / W4A8 GPU kernels · dense Qwen2/Llama only"]
```

Why the `.giw` path is the headline: it skips decompress **and** dequant/requant
**and** the resident-weight heap copy — the int8 weights are mapped straight from
the binary's read-only image. Measured on Qwen2.5-Coder-0.5B (M-series CPU):

| metric | embedded GGUF | prequant `.giw` | win |
|---|---|---|---|
| cold start | 2.30 s | 0.48 s | ~5× |
| resident heap (`phys_footprint`) | 772 MB | 78 MB | ~10× |
| binary size | 475 MB | 617 MB | +30% |

The RAM win scales with model size, which is what lets the demo ship in **two
size tiers** from one program (prequant `.giw`, M-series CPU, fixed prompt+seed):

| tier | binary | cold start | resident heap | tok/s (demo) |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | 617 MB | 0.48 s | 77 MB | ~57 |
| Qwen2.5-Coder-1.5B | 1.72 GB | 1.23 s | 87 MB | ~26 |

(End-to-end demo tok/s, M1 Pro, after the v0.5.0 perf work — see
`docs/completed/perf-campaign.md`. Pure runtime decode is higher: ~70 / ~36 tok/s on
`BenchmarkDecode`; the gap is streaming/UI overhead. ±a few tok/s by thermal
state.)

The 1.5B has ~3× the weights but **near-identical resident heap** (77 → 87 MB) —
they're image-mapped, not heap-copied — and stays under GitHub's 2 GiB asset cap.
Without prequant the 1.5B would need a ~1.7 GB weight heap per launch; that's the
enabler. (Looking further, a 4B int8 model no longer needs a ~5 GB weight heap,
which is what makes a `FROM scratch` container demo viable.) Quant is fixed when
the `.giw` is built; the runtime `--quant` flag and
the GGUF fallback apply only to the `--model <gguf>` path. Any bundle
mismatch (magic / version / quant / CRC) returns a typed error — never a panic —
and the user falls back via `--model`.

**Running bigger than RAM (`--stream-weights`).** The `.giw` is also an mmap-able
substrate: with `--stream-weights` the weights are paged on demand out of a
read-only mapping under a RAM budget (`--weight-cache`) instead of all held
resident — MoE expert demand-paging (run a 35B-A3B in ~16–20 GB) or dense per-layer
streaming. Bit-exact (read-only re-fault), trading RAM for cold-miss fault latency.
A plain `.gguf` is transparently transcoded to a sidecar `.giw` on first use, and
that transcode is itself **streaming** — `StreamTranscodeGGUF` converts one layer at
a time, so a model far larger than RAM (e.g. a 106B-A12B at int4) prequantizes with
a peak of ~one layer, not the whole model. Validated on real GLM-4.5-Air (106B):
prequantizes and then loads + generates via expert-paging on a 62 GB host.

**Two GPU modes (`-tags gpu`, WebGPU).** (1) A per-matmul `Backend` that
substitutes for the f32 kernel — the original, arch-agnostic path. (2) **Full
residency** (v0.4.0): for dense Qwen2/Llama the *entire token forward* runs on
the GPU through `DecodeRunner`, with **quantized** GPU kernels — `W8A8` (int8)
and `W4A8` (int4). This is the headline: a **7B int4 fits and decodes pure-GPU on
an 8 GB card** (~51 tok/s, ~71% of llama.cpp-CUDA at equal 4-bit quant) — the
model class that does *not* fit at int8. v1 residency limits: stateless
`Generate` only (Session/prefix-reuse fall back to the staged path), 16k context
(f32 KV) — or **~32k with the opt-in f16 KV cache (`--kv f16`, v0.5.0)** at the
same VRAM — dense Qwen2/Llama only (MoE / Gemma / hybrid → staged). Full numbers:
`docs/gpu-assessment.md`.

## 3. Module map (and where cgo is quarantined)

goinfer is the LLM-runtime half; the tensor/embedding primitives live in
`aikit`. On top of the `decoder` sit `chat` (per-family chat templates + tool
calling), `constrain` (schema-constrained decoding), and `cmd/serve` — an
HTTP server speaking both the OpenAI surface (chat/completions, completions,
Responses, with cross-call KV reuse) and the **Anthropic Messages API**
(`/v1/messages`), plus embeddings via aikit's `encoder`. Everything in the default
build is pure Go, no cgo. The one cgo dependency (`cogentcore/webgpu`) is sealed
inside the opt-in `goinfer/gpu` submodule, built only under `-tags gpu`.

```mermaid
flowchart TB
  subgraph GOINFER["github.com/townsendmerino/goinfer  (pure Go)"]
    direction TB
    DEC["decoder<br/>forward (softmax + Gated-DeltaNet + Mamba-2 hybrids · dense + sparse MoE) · quant kernels<br/>hybrid cache (KV + deltaState + mamba2State) · cross-call reuse · samplers · LoRA<br/>safetensors / GGUF / .giw loaders · streaming transcode + weight paging"]
    TKN["tokenizer<br/>byte-level BPE · SentencePiece byte-fallback"]
    CON["constrain<br/>logit-mask grammars · JSON Schema + Go-struct"]
    CHT["chat<br/>chat templates + tool calling (per family)"]
    SRV["cmd/serve<br/>OpenAI + Anthropic HTTP · chat/completions · responses · /v1/messages · embeddings"]
    DEC --> TKN
    CON -. "LogitProcessor" .-> DEC
    SRV --> DEC
    SRV --> CHT
    SRV --> CON
  end

  subgraph AIKIT["github.com/townsendmerino/aikit  (pure Go)"]
    direction TB
    EMBP["embed<br/>GGUF/safetensors parse · OpenGGUFBytes"]
    LIN["linalg<br/>SIMD dot/matmul (NEON · AVX2/FMA)"]
    ENC["encoder<br/>CodeRankEmbed embeddings (f32 / int8)"]
  end

  DEC --> EMBP
  DEC --> LIN
  TKN --> EMBP
  SRV --> ENC

  subgraph GPU["goinfer/gpu  (opt-in · -tags gpu · cgo)"]
    WG["WebGPU → cogentcore/webgpu (wgpu-native)<br/>(1) registers a matmul Backend<br/>(2) full-residency DecodeRunner (W8A8/W4A8 token forward)"]
  end
  GPU -. "registers into" .-> DEC
  GPU -. "and aikit/encoder" .-> AIKIT
```

The arrows into `gpu` are dashed because the dependency is *inverted*: `gpu`
imports `decoder`/`encoder` and registers into them on init, so `webgpu` never
enters the core module graph. The default `go build` pulls only `aikit` +
`golang.org/x/text` — pure Go, no cgo. (Native GPU is cgo via wgpu-native; a
future browser/wasm backend would reach `navigator.gpu` through `syscall/js`,
cgo-free — see `docs/roadmap.md`.)

## The contract

Numerics — the forward pass and quantization — are **parity-gated against
HuggingFace** and are the stable surface. The loader and `Architecture`
descriptor are pre-1.0 and move as new model families and quant formats land;
that's why the `.giw` format carries a version guard (a stale bundle triggers a
safe rebuild via the GGUF path, never a crash).

# goinfer architecture

A tour of how goinfer turns a prompt into tokens, and how the model gets from a
file (or the binary's own image) into RAM. Three diagrams: the **forward pass**,
the **load + memory paths**, and the **module map**.

> These diagrams are intentionally drawn at the *stage* level, not the
> struct-field level. goinfer is descriptor-driven — one generic decoder runs
> Gemma/Qwen/Llama/Mistral/Mixtral/GPT-2/Mellum by reading an `Architecture`
> descriptor — so the per-model specifics (which norm, GQA ratio, RoPE scaling,
> tied vs separate head) are *config*, not separate code paths. Drawing stages
> keeps these accurate across new model families; the numeric contract is the
> parity gate against HuggingFace, not this doc.

## 1. The forward pass (one decode step)

Each generated token runs the full stack once. Prefill runs the prompt tokens
through the same path (filling the KV cache) and only keeps logits for the last.

```mermaid
flowchart TB
  TOK["tokenizer · BPE / SentencePiece<br/>(id-exact vs HuggingFace)"] --> EMB["token embedding<br/>(+ optional embed scale, learned pos emb)"]
  EMB --> BLK

  subgraph BLK["decoder block × N  (Architecture descriptor)"]
    direction TB
    N1["norm · RMSNorm or LayerNorm"] --> ATT["causal attention<br/>RoPE (linear / llama3 / yarn)<br/>GQA · optional sliding window<br/>reads + appends KV cache"]
    ATT --> PA{"sandwich norm?"}
    PA -->|yes| N1b["post-attn norm"] --> R1(("+ residual"))
    PA -->|no| R1
    R1 --> N2["norm"] --> MLP["MLP · SwiGLU (SiLU gate · up → down)"]
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

  W --> DISP{"matmul dispatch<br/>(per weightMat precision)"}
  DISP -->|int4| K4["MatmulBTQ4 · CPU (nibble unpack)"]
  DISP -->|int8 W8A8| K8["MatmulBTW8A8 · CPU SDOT (fast default)"]
  DISP -->|int8| KQ["MatmulBTQ8 · CPU"]
  DISP -->|f32| BE["Backend.MatmulBT<br/>pure-Go SIMD · or webgpu (-tags gpu)"]
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
`docs/perf-campaign.md`. Pure runtime decode is higher: ~70 / ~36 tok/s on
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

Note the GPU path only substitutes for the **f32** kernel; quantized matmuls are
CPU-only, so the int8 demo is CPU by design (`-tags gpu` helps only unquantized
models).

## 3. Module map (and where cgo is quarantined)

goinfer is the LLM-runtime half; the tensor/embedding primitives live in
`aikit`. Everything in the default build is pure Go, no cgo. The one cgo
dependency (`cogentcore/webgpu`) is sealed inside the opt-in `goinfer/gpu`
submodule, built only under `-tags gpu`.

```mermaid
flowchart TB
  subgraph GOINFER["github.com/townsendmerino/goinfer  (pure Go)"]
    direction TB
    DEC["decoder<br/>forward pass · quant kernels · KV cache · samplers<br/>GGUF / .giw loaders"]
    TKN["tokenizer<br/>byte-level BPE · SentencePiece byte-fallback"]
    CON["constrain<br/>logit-mask grammars (JSON)"]
    DEC --> TKN
    CON -. "LogitProcessor" .-> DEC
  end

  subgraph AIKIT["github.com/townsendmerino/aikit  (pure Go)"]
    direction TB
    EMBP["embed<br/>GGUF/safetensors parse · OpenGGUFBytes"]
    LIN["linalg<br/>SIMD dot/matmul (NEON · AVX2/FMA)"]
  end

  DEC --> EMBP
  DEC --> LIN
  TKN --> EMBP

  subgraph GPU["goinfer/gpu  (opt-in · -tags gpu · cgo)"]
    WG["WebGPU backend → cogentcore/webgpu<br/>registers a matmul Backend"]
  end
  GPU -. "registers into" .-> DEC
  GPU -. "and aikit/encoder" .-> AIKIT
```

The arrows into `gpu` are dashed because the dependency is *inverted*: `gpu`
imports `decoder`/`encoder` and registers a backend on init, so `webgpu` never
enters the core module graph. The default `go build` pulls only `aikit` +
`golang.org/x/text`.

## The contract

Numerics — the forward pass and quantization — are **parity-gated against
HuggingFace** and are the stable surface. The loader and `Architecture`
descriptor are pre-1.0 and move as new model families and quant formats land;
that's why the `.giw` format carries a version guard (a stale bundle triggers a
safe rebuild via the GGUF path, never a crash).

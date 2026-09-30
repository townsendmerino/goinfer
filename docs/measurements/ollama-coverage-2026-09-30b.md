# Ollama library coverage snapshot — which of Ollama's 60 most-pulled models goinfer runs (2026-09-30, afternoon)

A snapshot, not a benchmark. It answers one question for someone arriving from Ollama: does goinfer run the
model I already use? It replaces [the morning's snapshot](ollama-coverage-2026-09-30.md) as the source for the
"Coming from Ollama?" section of the site's Models page (`docs/tasks/task-site-2026-09.md` S2), after Gemma 1,
CodeGemma and Gemma 2 support landed the same day. Every figure that section shows must appear here, per the
claims check (S8b).

## Method

- **Source.** `https://ollama.com/library?sort=popular`, read 2026-09-30 at 16:07 PDT. The raw page is kept
  beside this file (`ollama-coverage-2026-09-30b/library.html.gz`) with the script that reads it (`parse.py`):
  the first 60 entries, in page order, with the pull count and capability tags the page prints. They are the
  morning's 60; qwen3.8 and mixtral trade places at #50 and #51.
- **Pull counts are cumulative since each model was published.** They overstate older models (llama3.1,
  gemma2) against newer ones (qwen3.5, gemma4). A pull is a download, not a user and not current use. Treat
  the ordering as rough popularity, and the percentages below as a share of downloads, nothing more.
- **Pull counts are as printed.** The page prints one decimal below 100M and rounds at or above it, so
  llama3.1 reads 120.0 here where the morning read 119.9.
- **Capability tags.** The page now also prints a "cloud" chip (a hosted variant, not a model capability);
  it is left out.
- **Status is judged per family, against `docs/capability-matrix.json` at `83c45abd`.** An Ollama tag maps to
  the Hugging Face architecture its default weights use.
  - **S, supported**: goinfer runs the family, including every capability Ollama tags (vision included where
    tagged).
  - **T, text only**: goinfer runs the text model, but Ollama ships it with vision, and goinfer does not run
    the images.
  - **N, not supported**: goinfer cannot load the architecture. The `needs` column names the Hugging Face
    `model_type` it would take, read from each release's config.json (llava is Ollama's llava 1.6,
    llava_next), so the site's build can tell when the capability matrix gains it.
  - **U, unverified**: the family matches, but this checkpoint's variant has not been checked against
    goinfer's implementation.
- **What changed since the morning.** gemma2, gemma and codegemma move from N to S (Gemma 1 and CodeGemma are
  `model_type` gemma; Gemma 2 is gemma2). Every other row's status is the morning's.
- **Embedding models** run through aikit's encoder (BERT, nomic-bert, GTE, MiniLM, Arctic, BGE, XLM-RoBERTa),
  and `qwen3-embedding` / `embeddinggemma` through the decoder-as-embedder path.
- **Tags whose weights span architectures** (`deepseek-r1`: Qwen and Llama distills at small sizes, DeepSeek
  V3 at 671B) are S only because goinfer runs every architecture the tag's sizes use.
- **Loaders.** Ollama users hold GGUF files. Where goinfer loads a family from safetensors only, the row says
  so. The model still runs, but not from the file the visitor already has.

## Result

| status | tags | pulls (M) | share of the 60's pulls |
|---|---|---|---|
| S, supported | 46 | 854.7 | 92.4% |
| T, text only (Ollama ships vision) | 2 | 8.9 | 1.0% |
| N, not supported | 11 | 58.2 | 6.3% |
| U, unverified | 1 | 3.0 | 0.3% |
| **all 60** | **60** | **924.8** | **100%** |

**The sentence the site may use:** "Of the 60 most-pulled models in Ollama's library on 2026-09-30, goinfer
runs the families behind 92.4% of the pulls as Ollama ships them, and another 1.0% as text only."

## The 60, in page order

| # | Ollama tag | pulls (M) | Ollama tags | goinfer family | status | needs | note |
|---|---|---|---|---|---|---|---|
| 1 | llama3.1 | 120.0 | tools | llama | S |  |  |
| 2 | deepseek-r1 | 93.3 | tools, thinking | qwen2 / qwen3 / llama / deepseek_v3 | S |  | distills plus the 671B |
| 3 | nomic-embed-text | 87.6 | embedding | aikit encoder (nomic-bert) | S |  |  |
| 4 | llama3.2 | 84.7 | tools | llama | S |  |  |
| 5 | qwen2.5 | 41.4 | tools | qwen2 | S |  |  |
| 6 | gemma3 | 40.8 | vision | gemma3 | S |  |  |
| 7 | qwen3 | 38.5 | tools, thinking | qwen3 | S |  |  |
| 8 | mistral | 33.8 | tools | mistral | S |  |  |
| 9 | gemma2 | 33.7 |  | gemma2 | S |  |  |
| 10 | gemma4 | 26.1 | vision, tools, thinking, audio | gemma4 | S |  | images yes; audio input not yet |
| 11 | llama3 | 25.4 |  | llama | S |  |  |
| 12 | qwen2.5-coder | 21.9 | tools | qwen2 | S |  |  |
| 13 | qwen3.5 | 21.3 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S |  | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 14 | phi3 | 18.2 |  | phi3 | S |  |  |
| 15 | mxbai-embed-large | 15.2 | embedding | aikit encoder (BERT) | S |  |  |
| 16 | llava | 15.0 | vision | — | N | llava_next |  |
| 17 | gpt-oss | 13.4 | tools, thinking | gpt-oss | S |  |  |
| 18 | qwen3-coder | 9.6 | tools | qwen3_moe | S |  |  |
| 19 | gemma | 8.3 |  | gemma | S |  | Gemma 1 |
| 20 | qwen | 7.9 |  | — | N | qwen | Qwen 1 |
| 21 | phi4 | 7.7 |  | phi3 | S |  | Phi-4 uses the phi3 architecture |
| 22 | glm-ocr | 7.7 | vision, tools | — | N | glm_ocr |  |
| 23 | llama2 | 7.5 |  | llama | S |  |  |
| 24 | bge-m3 | 7.2 | embedding | aikit encoder (XLM-RoBERTa) | S |  |  |
| 25 | qwen3.6 | 6.9 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S |  | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 26 | codellama | 6.6 |  | llama | S |  |  |
| 27 | qwen3-vl | 6.4 | vision, tools, thinking | qwen3_vl | T |  | text side experimental; safetensors only |
| 28 | qwen2 | 6.2 | tools | qwen2 | S |  |  |
| 29 | tinyllama | 5.9 |  | llama | S |  |  |
| 30 | mistral-nemo | 5.7 | tools | mistral | S |  |  |
| 31 | minicpm-v | 5.5 | vision | — | N | minicpmv |  |
| 32 | llama3.2-vision | 5.4 | vision | — | N | mllama | cross-attention (mllama) |
| 33 | qwen2.5vl | 5.2 | vision | qwen2_5_vl | S |  | safetensors only |
| 34 | deepseek-coder | 4.7 |  | llama | S |  |  |
| 35 | llama3.3 | 4.2 | tools | llama | S |  |  |
| 36 | dolphin3 | 4.1 |  | llama | S |  |  |
| 37 | qwen3-embedding | 4.1 | embedding | qwen3 (decoder as embedder) | S |  |  |
| 38 | smollm2 | 4.1 | tools | llama | S |  |  |
| 39 | deepseek-v3 | 3.8 |  | deepseek_v3 | S |  |  |
| 40 | olmo2 | 3.8 |  | — | N | olmo2 | goinfer has olmo3, not olmo2 |
| 41 | all-minilm | 3.7 | embedding | aikit encoder (MiniLM) | S |  |  |
| 42 | codegemma | 3.2 |  | gemma | S |  | the Gemma 1 architecture |
| 43 | deepseek-coder-v2 | 3.2 |  | deepseek_v2 | S |  |  |
| 44 | mistral-small | 3.1 | tools | mistral | S |  |  |
| 45 | snowflake-arctic-embed | 3.1 | embedding | aikit encoder (Arctic) | S |  |  |
| 46 | orca-mini | 3.1 |  | llama | S |  |  |
| 47 | granite3.1-moe | 3.0 | tools | — | N | granitemoe | granitemoe; goinfer has granite and granitemoehybrid |
| 48 | starcoder2 | 3.0 |  | — | N | starcoder2 |  |
| 49 | nemotron-3-super | 3.0 | tools, thinking | nemotron_h | U |  | family matches; the Super MoE variant is unchecked |
| 50 | qwen3.8 | 2.9 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S |  | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 51 | mixtral | 2.9 | tools | mixtral | S |  | parity tier experimental |
| 52 | llama2-uncensored | 2.7 |  | llama | S |  |  |
| 53 | falcon3 | 2.6 |  | llama | S |  |  |
| 54 | translategemma | 2.5 | vision | gemma3 | S |  |  |
| 55 | mistral-small3.2 | 2.5 | vision, tools | mistral3 | T |  | vision tower ignored; safetensors only |
| 56 | minimax-m2.7 | 2.4 | tools, thinking | — | N | minimax_m2 | minimax_m2 |
| 57 | llava-llama3 | 2.3 | vision | — | N | llava |  |
| 58 | qwq | 2.3 | tools | qwen2 | S |  |  |
| 59 | embeddinggemma | 2.3 | embedding | gemma3 (decoder as embedder) | S |  |  |
| 60 | gemma3n | 2.2 |  | — | N | gemma3n |  |

## What this is not

- Not a speed comparison. It says whether a family loads, not how fast it runs.
- Not per-checkpoint validation. "Supported" means the family loads and is parity-gated at the tier the
  capability matrix shows. It does not mean every size under an Ollama tag was run. Gemma 1 and Gemma 2 are
  gated on seeded test models today, not yet on a released checkpoint.
- Not a census of what people use. Hosted-API rankings (OpenRouter) measure something different, and none of
  their top open models fits this page's audience's hardware, so they are left out.

## Refreshing it

Re-read the library page, write a new dated file beside this one, and point the site's claims at it. Do not
edit this one. A row's status changes when `capability-matrix.json` changes: a family gaining images, or a new
family an N row's `needs` names. The site derives each row's status from the matrix at build time and fails the
build when a derived status differs from this file's, or when an N row's `needs` becomes a family in the matrix,
so this file's status column is the reading at `83c45abd`, not the source of truth afterwards. The headline
percentages are claims, so when a status flips, they need a new snapshot rather than a recomputation.

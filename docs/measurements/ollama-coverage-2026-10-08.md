# Ollama library coverage snapshot — which of Ollama's 60 most-pulled models goinfer runs (2026-10-08, evening)

A snapshot, not a benchmark. It answers one question for someone arriving from Ollama: does goinfer run the
model I already use? It replaces [the 2026-10-02 snapshot](ollama-coverage-2026-10-02.md) as the source
for the "Coming from Ollama?" section of the site's Models page (`docs/tasks/task-site-2026-09.md` S2), after
Qwen3-VL's images were verified (`docs/tasks/task-multimodal-support-2026-10.md`, S10) and the capability matrix moved `qwen3_vl` to a full-oracle vision family. Every figure that section shows must
appear here, per the claims check (S8b).

## Method

- **Source.** `https://ollama.com/library?sort=popular`, read 2026-10-08 at 21:25 PDT. The raw page is kept
  beside this file (`ollama-coverage-2026-10-08/library.html.gz`) with the script that reads it (`parse.py`):
  the first 60 entries, in page order, with the pull count and capability tags the page prints. They are the
  same 60 tags as the 2026-10-02 reading; 14 of them trade places (the largest move is qwen3.8, 48 to 42) and 29 pull counts moved
  (by at most 0.8 M). No capability tag changed.
- **Pull counts are cumulative since each model was published.** They overstate older models (llama3.1,
  gemma2) against newer ones (qwen3.5, gemma4). A pull is a download, not a user and not current use. Treat
  the ordering as rough popularity, and the percentages below as a share of downloads, nothing more.
- **Pull counts are as printed.** The page prints one decimal below 100M and rounds at or above it, so
  llama3.1 reads 120.0 here.
- **Capability tags.** The page now also prints a "cloud" chip (a hosted variant, not a model capability);
  it is left out.
- **Status is judged per family, against `docs/capability-matrix.json` at `b379932b`.** An Ollama tag maps to
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
- **What changed since 2026-10-02.** qwen3-vl moves from T to S: goinfer now reads its images (`qwen3_vl` is `full-oracle 100.0%/1.00000` in the
  matrix, verified on Qwen3-VL-2B; safetensors only, no GGUF). Every other row's status is unchanged. Two things this row does not claim:
  the CUDA DeepStack prefill is off by default, and a 896-pixel image still has an open prefill gap against the transformers reference
  (`docs/tasks/task-multimodal-support-2026-10.md`, G-S10j), so "supported" here means the family loads and is parity-gated, as everywhere on this page.
- **Embedding models** run through aikit's encoder (BERT, nomic-bert, GTE, MiniLM, Arctic, BGE, XLM-RoBERTa),
  and `qwen3-embedding` / `embeddinggemma` through the decoder-as-embedder path.
- **Tags whose weights span architectures** (`deepseek-r1`: Qwen and Llama distills at small sizes, DeepSeek
  V3 at 671B) are S only because goinfer runs every architecture the tag's sizes use.
- **Loaders.** Ollama users hold GGUF files. Where goinfer loads a family from safetensors only, the row says
  so. The model still runs, but not from the file the visitor already has.

## Result

| status | tags | pulls (M) | share of the 60's pulls |
|---|---|---|---|
| S, supported | 48 | 877.5 | 94.0% |
| T, text only (Ollama ships vision) | 1 | 2.5 | 0.3% |
| N, not supported | 10 | 50.8 | 5.4% |
| U, unverified | 1 | 3.0 | 0.3% |
| **all 60** | **60** | **933.8** | **100%** |

**The sentence the site may use:** "Of the 60 most-pulled models in Ollama's library on 2026-10-08, goinfer runs the families behind 94.0% of the pulls as Ollama ships them, and another 0.3% as text only."

## The 60, in page order

| # | Ollama tag | pulls (M) | Ollama tags | goinfer family | status | needs | note |
|---|---|---|---|---|---|---|---|
| 1 | llama3.1 | 120.2 | tools | llama | S |  |  |
| 2 | deepseek-r1 | 93.5 | tools, thinking | qwen2 / qwen3 / llama / deepseek_v3 | S |  | distills plus the 671B |
| 3 | nomic-embed-text | 88.6 | embedding | aikit encoder (nomic-bert) | S |  |  |
| 4 | llama3.2 | 85.1 | tools | llama | S |  |  |
| 5 | qwen2.5 | 42.0 | tools | qwen2 | S |  |  |
| 6 | gemma3 | 41.0 | vision | gemma3 | S |  |  |
| 7 | qwen3 | 39.4 | tools, thinking | qwen3 | S |  |  |
| 8 | gemma2 | 34.4 |  | gemma2 | S |  |  |
| 9 | mistral | 33.8 | tools | mistral | S |  |  |
| 10 | gemma4 | 26.6 | vision, tools, thinking, audio | gemma4 | S |  | images yes; audio input not yet |
| 11 | llama3 | 25.5 |  | llama | S |  |  |
| 12 | qwen2.5-coder | 22.2 | tools | qwen2 | S |  |  |
| 13 | qwen3.5 | 21.9 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S |  | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 14 | phi3 | 18.2 |  | phi3 | S |  |  |
| 15 | mxbai-embed-large | 15.5 | embedding | aikit encoder (BERT) | S |  |  |
| 16 | llava | 15.1 | vision | — | N | llava_next |  |
| 17 | gpt-oss | 13.6 | tools, thinking | gpt-oss | S |  |  |
| 18 | qwen3-coder | 9.8 | tools | qwen3_moe | S |  |  |
| 19 | gemma | 8.3 |  | gemma | S |  | Gemma 1 |
| 20 | glm-ocr | 8.1 | vision, tools | glm_ocr | S |  | images from the safetensors checkpoint only (no GGUF); the vision tower runs on the CPU; tool calls are not rendered; parity tier experimental |
| 21 | qwen | 8.0 |  | — | N | qwen | Qwen 1 |
| 22 | phi4 | 7.7 |  | phi3 | S |  | Phi-4 uses the phi3 architecture |
| 23 | llama2 | 7.5 |  | llama | S |  |  |
| 24 | bge-m3 | 7.5 | embedding | aikit encoder (XLM-RoBERTa) | S |  |  |
| 25 | qwen3.6 | 7.0 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S |  | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 26 | codellama | 6.8 |  | llama | S |  |  |
| 27 | qwen3-vl | 6.5 | vision, tools, thinking | qwen3_vl | S |  | images from the safetensors checkpoint only (no GGUF); verified on Qwen3-VL-2B |
| 28 | qwen2 | 6.3 | tools | qwen2 | S |  |  |
| 29 | tinyllama | 6.0 |  | llama | S |  |  |
| 30 | mistral-nemo | 5.7 | tools | mistral | S |  |  |
| 31 | minicpm-v | 5.5 | vision | — | N | minicpmv |  |
| 32 | llama3.2-vision | 5.4 | vision | — | N | mllama | cross-attention (mllama) |
| 33 | qwen2.5vl | 5.3 | vision | qwen2_5_vl | S |  | safetensors only |
| 34 | deepseek-coder | 4.7 |  | llama | S |  |  |
| 35 | qwen3-embedding | 4.2 | embedding | qwen3 (decoder as embedder) | S |  |  |
| 36 | llama3.3 | 4.2 | tools | llama | S |  |  |
| 37 | dolphin3 | 4.2 |  | llama | S |  |  |
| 38 | smollm2 | 4.1 | tools | llama | S |  |  |
| 39 | deepseek-v3 | 3.8 |  | deepseek_v3 | S |  |  |
| 40 | olmo2 | 3.8 |  | — | N | olmo2 | goinfer has olmo3, not olmo2 |
| 41 | all-minilm | 3.7 | embedding | aikit encoder (MiniLM) | S |  |  |
| 42 | qwen3.8 | 3.4 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S |  | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 43 | codegemma | 3.2 |  | gemma | S |  | the Gemma 1 architecture |
| 44 | deepseek-coder-v2 | 3.2 |  | deepseek_v2 | S |  |  |
| 45 | mistral-small | 3.1 | tools | mistral | S |  |  |
| 46 | snowflake-arctic-embed | 3.1 | embedding | aikit encoder (Arctic) | S |  |  |
| 47 | orca-mini | 3.1 |  | llama | S |  |  |
| 48 | granite3.1-moe | 3.0 | tools | — | N | granitemoe | granitemoe; goinfer has granite and granitemoehybrid |
| 49 | starcoder2 | 3.0 |  | — | N | starcoder2 |  |
| 50 | nemotron-3-super | 3.0 | tools, thinking | nemotron_h | U |  | family matches; the Super MoE variant is unchecked |
| 51 | mixtral | 2.9 | tools | mixtral | S |  | parity tier experimental |
| 52 | llama2-uncensored | 2.7 |  | llama | S |  |  |
| 53 | falcon3 | 2.6 |  | llama | S |  |  |
| 54 | translategemma | 2.6 | vision | gemma3 | S |  |  |
| 55 | mistral-small3.2 | 2.5 | vision, tools | mistral3 | T |  | vision tower ignored; safetensors only |
| 56 | embeddinggemma | 2.4 | embedding | gemma3 (decoder as embedder) | S |  |  |
| 57 | minimax-m2.7 | 2.4 | tools, thinking | — | N | minimax_m2 | minimax_m2 |
| 58 | llava-llama3 | 2.3 | vision | — | N | llava |  |
| 59 | qwq | 2.3 | tools | qwen2 | S |  |  |
| 60 | gemma3n | 2.3 |  | — | N | gemma3n |  |

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
so this file's status column is the reading at `b379932b`, not the source of truth afterwards. The headline
percentages are claims, so when a status flips, they need a new snapshot rather than a recomputation.

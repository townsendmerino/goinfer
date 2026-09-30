# Ollama library coverage snapshot — which of Ollama's 60 most-pulled models goinfer runs (2026-09-30)

A snapshot, not a benchmark. It answers one question for someone arriving from Ollama: does goinfer run the
model I already use? It is the source for the "Coming from Ollama?" section of the site's Models page
(`docs/tasks/task-site-2026-09.md` S2). Every figure that section shows must appear here, per the claims check
(S8b).

## Method

- **Source.** `https://ollama.com/library?sort=popular`, read 2026-09-30 at about 05:50 PDT. The first 60
  entries, in page order, with the pull count and capability tags the page prints.
- **Pull counts are cumulative since each model was published.** They overstate older models (llama3.1,
  gemma2) against newer ones (qwen3.5, gemma4). A pull is a download, not a user and not current use. Treat
  the ordering as rough popularity, and the percentages below as a share of downloads, nothing more.
- **Status is judged per family, against `docs/capability-matrix.json` at `621036da`.** An Ollama tag maps to
  the Hugging Face architecture its default weights use.
  - **S, supported**: goinfer runs the family, including every capability Ollama tags (vision included where
    tagged).
  - **T, text only**: goinfer runs the text model, but Ollama ships it with vision, and goinfer does not run
    the images.
  - **N, not supported**: goinfer cannot load the architecture.
  - **U, unverified**: the family matches, but this checkpoint's variant has not been checked against
    goinfer's implementation.
- **Re-judged once, 2026-09-30, before this file's first commit.** The first reading (matrix at `ac1c530e`) had
  qwen3.5, qwen3.6 and qwen3.8 as T. P8a then gave `qwen3_5` images the same day (`c94337e2`), so they are S, noted
  as safetensors and dense sizes only, the way qwen2.5vl's row notes its loader. The Ollama page reading is
  unchanged. The Result table and the sentence below were recomputed from the rows.
- **Embedding models** run through aikit's encoder (BERT, nomic-bert, GTE, MiniLM, Arctic, BGE, XLM-RoBERTa),
  and `qwen3-embedding` / `embeddinggemma` through the decoder-as-embedder path.
- **Tags whose weights span architectures** (`deepseek-r1`: Qwen and Llama distills at small sizes, DeepSeek
  V3 at 671B) are S only because goinfer runs every architecture the tag's sizes use.
- **Loaders.** Ollama users hold GGUF files. Where goinfer loads a family from safetensors only, the row says
  so. The model still runs, but not from the file the visitor already has.

## Result

| status | tags | pulls (M) | share of the 60's pulls |
|---|---|---|---|
| S, supported | 43 | 808.3 | 87.5% |
| T, text only (Ollama ships vision) | 2 | 8.9 | 1.0% |
| N, not supported | 14 | 103.3 | 11.2% |
| U, unverified | 1 | 3.0 | 0.3% |
| **all 60** | **60** | **923.5** | **100%** |

**The sentence the site may use:** "Of the 60 most-pulled models in Ollama's library on 2026-09-30, goinfer
runs the families behind 87.5% of the pulls as Ollama ships them, and another 1.0% as text only."

## The 60, in page order

| # | Ollama tag | pulls (M) | Ollama tags | goinfer family | status | note |
|---|---|---|---|---|---|---|
| 1 | llama3.1 | 119.9 | tools | llama | S | |
| 2 | deepseek-r1 | 93.3 | tools, thinking | qwen2 / qwen3 / llama / deepseek_v3 | S | distills plus the 671B |
| 3 | nomic-embed-text | 87.5 | embedding | aikit encoder (nomic-bert) | S | |
| 4 | llama3.2 | 84.6 | tools | llama | S | |
| 5 | qwen2.5 | 41.2 | tools | qwen2 | S | |
| 6 | gemma3 | 40.8 | vision | gemma3 | S | |
| 7 | qwen3 | 38.3 | tools, thinking | qwen3 | S | |
| 8 | mistral | 33.7 | tools | mistral | S | |
| 9 | gemma2 | 33.6 | | — | N | |
| 10 | gemma4 | 26.0 | vision, tools, thinking, audio | gemma4 | S | images yes; audio input not yet |
| 11 | llama3 | 25.4 | | llama | S | |
| 12 | qwen2.5-coder | 21.9 | tools | qwen2 | S | |
| 13 | qwen3.5 | 21.2 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 14 | phi3 | 18.2 | | phi3 | S | |
| 15 | mxbai-embed-large | 15.2 | embedding | aikit encoder (BERT) | S | |
| 16 | llava | 15.0 | vision | — | N | |
| 17 | gpt-oss | 13.3 | tools, thinking | gpt-oss | S | |
| 18 | qwen3-coder | 9.6 | tools | qwen3_moe | S | |
| 19 | gemma | 8.3 | | — | N | Gemma 1 |
| 20 | qwen | 7.9 | | — | N | Qwen 1 |
| 21 | phi4 | 7.7 | | phi3 | S | Phi-4 uses the phi3 architecture |
| 22 | glm-ocr | 7.7 | vision, tools | — | N | |
| 23 | llama2 | 7.5 | | llama | S | |
| 24 | bge-m3 | 7.2 | embedding | aikit encoder (XLM-RoBERTa) | S | |
| 25 | qwen3.6 | 6.9 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 26 | codellama | 6.6 | | llama | S | |
| 27 | qwen3-vl | 6.4 | vision, tools, thinking | qwen3_vl | T | text side experimental; safetensors only |
| 28 | qwen2 | 6.2 | tools | qwen2 | S | |
| 29 | tinyllama | 5.9 | | llama | S | |
| 30 | mistral-nemo | 5.7 | tools | mistral | S | |
| 31 | minicpm-v | 5.5 | vision | — | N | |
| 32 | llama3.2-vision | 5.4 | vision | — | N | cross-attention (mllama) |
| 33 | qwen2.5vl | 5.2 | vision | qwen2_5_vl | S | safetensors only |
| 34 | deepseek-coder | 4.7 | | llama | S | |
| 35 | llama3.3 | 4.2 | tools | llama | S | |
| 36 | dolphin3 | 4.1 | | llama | S | |
| 37 | qwen3-embedding | 4.1 | embedding | qwen3 (decoder as embedder) | S | |
| 38 | smollm2 | 4.1 | tools | llama | S | |
| 39 | deepseek-v3 | 3.8 | | deepseek_v3 | S | |
| 40 | olmo2 | 3.8 | | — | N | goinfer has olmo3, not olmo2 |
| 41 | all-minilm | 3.7 | embedding | aikit encoder (MiniLM) | S | |
| 42 | codegemma | 3.2 | | — | N | Gemma 1 architecture |
| 43 | deepseek-coder-v2 | 3.2 | | deepseek_v2 | S | |
| 44 | mistral-small | 3.1 | tools | mistral | S | |
| 45 | snowflake-arctic-embed | 3.1 | embedding | aikit encoder (Arctic) | S | |
| 46 | orca-mini | 3.1 | | llama | S | |
| 47 | granite3.1-moe | 3.0 | tools | — | N | granitemoe; goinfer has granite and granitemoehybrid |
| 48 | starcoder2 | 3.0 | | — | N | |
| 49 | nemotron-3-super | 3.0 | tools, thinking | nemotron_h | U | family matches; the Super MoE variant is unchecked |
| 50 | mixtral | 2.9 | tools | mixtral | S | parity tier experimental |
| 51 | qwen3.8 | 2.8 | vision, tools, thinking | qwen3_5 / qwen3_5_moe | S | images from safetensors, dense sizes only; Ollama's GGUF and the MoE sizes run as text |
| 52 | llama2-uncensored | 2.7 | | llama | S | |
| 53 | falcon3 | 2.6 | | llama | S | |
| 54 | translategemma | 2.5 | vision | gemma3 | S | |
| 55 | mistral-small3.2 | 2.5 | vision, tools | mistral3 | T | vision tower ignored; safetensors only |
| 56 | minimax-m2.7 | 2.4 | tools, thinking | — | N | minimax_m2 |
| 57 | llava-llama3 | 2.3 | vision | — | N | |
| 58 | qwq | 2.3 | tools, thinking | qwen2 | S | |
| 59 | embeddinggemma | 2.3 | embedding | gemma3 (decoder as embedder) | S | |
| 60 | gemma3n | 2.2 | | — | N | |

## What this is not

- Not a speed comparison. It says whether a family loads, not how fast it runs.
- Not per-checkpoint validation. "Supported" means the family loads and is parity-gated at the tier the
  capability matrix shows. It does not mean every size under an Ollama tag was run.
- Not a census of what people use. Hosted-API rankings (OpenRouter) measure something different, and none of
  their top open models fits this page's audience's hardware, so they are left out.

## Refreshing it

Re-read the library page, write a new dated file beside this one, and point the site's claims at it. Do not
edit this one. A row's status changes when `capability-matrix.json` changes (Gemma 2 landing, or Qwen3.5+ images
from GGUF, `docs/multimodal.md` §P8b). The site derives each row's status from the matrix at build time, so this file's
status column is the reading on 2026-09-30, not the source of truth afterwards. The headline percentages are
different: they are claims, so when a status flips, they need a new snapshot rather than a recomputation.

# Tool-call coverage by family

Which families get **constrained** tool calls (a call cannot be malformed or name a tool you did not supply), which get tool calls
**parsed only**, and which get **none** — and on what evidence. Task T4 of
[`task-tool-grammar-union-2026-09.md`](tasks/task-tool-grammar-union-2026-09.md). The capability matrix's families are the rows;
"tools: yes" there (or anywhere) does not imply constrainability — this page is where that is decided.

## How a family gets its tool behaviour

goinfer does not assign a chat template per family. It fingerprints the **checkpoint's own template** (`chat.Detect`, `chat/chat.go`)
into one of eight renderers, and the renderer decides tools:

| renderer | tools | call form | `required` / named | `auto` |
|---|---|---|---|---|
| chatml, mellum2 | yes | `<tool_call>{…}</tool_call>` | constrained from token 1 | constrained from the opener |
| mistral (v0.3) | yes | `[TOOL_CALLS] [{…}]` | constrained from token 1 | constrained from the opener |
| llama3 | yes | bare `{…}` | constrained from token 1 | parsed only (no opener to arm on) |
| gemma4 | yes | its own call syntax | parsed only | parsed only |
| gemma3, harmony (gpt-oss), ministral | no | — | 400 on a named choice | — |
| none recognised | no | — | raw completion | — |

On chatml/mellum2 an unwrapped call that names a supplied tool is also accepted (Qwen2.5-Coder writes calls without the wrapper).
Under `auto` the constraint is off on a server running speculative decoding, which keeps its drafter.

## Census, 2026-09-24

**Method.** Every checkpoint on `nobara-pc` (`~/models` and the `/srv/models` archive; metadata read only) loaded through the same
tokenizer loaders `serve` uses, then `chat.Detect`, `SupportsTools` and `ToolCallWrapper` — the real code path, not a reading of names.
The driver was throwaway; its output is summarised here. A family's row describes **the checkpoints this box has**; where those are
pruned copies without their released template, the row says so instead of guessing.

| family | checkpoint(s) examined | resolves to | tool calls | caveat |
|---|---|---|---|---|
| Qwen2 / Qwen2.5 | qwen2.5-0.5b/7b-instruct, qwen2.5-coder-0.5b/1.5b (GGUF + safetensors) | chatml | **constrained** | measured (T0, T1–T3) |
| Qwen2-MoE | qwen15-moe-a27b | chatml | **constrained** | |
| Qwen2.5-VL | qwen25vl-3b-instruct | chatml | **constrained** | |
| Qwen3 | qwen3-1.7b (bf16, Q8_0), qwen3-4b, Qwen3-0.6B Q8_0 | chatml | **constrained** | |
| Qwen3-MoE | qwen3moe-30b-a3b-bf16 | chatml | **constrained** | |
| Qwen3-Next | qwen3next-80b-partial | chatml | **constrained** | |
| Qwen3.5-MoE | qwen3.6-35b-a3b (safetensors, Q8_0, Q4_K_M), Qwen3.5-35B-A3B Q4_K_M | chatml | **constrained** | |
| Qwen3.8 | qwen3.5-0.8b, qwen3.8-27b, Qwen3.8-27B GGUF | chatml | **constrained** | |
| Nemotron-H | Nemotron-3-Nano-30B (bf16, GGUF) | chatml | **constrained** | Nemotron-Nano-9B-v2 (HF and Q8_0): not recognised → none |
| Mellum2 | mellum2-unq | mellum2 | **constrained** | template is only in `chat_template.jinja`; resolved as generic chatml until finding 1 was fixed (same call form) |
| Llama (3.x) | llama-3.2-1b-instruct Q4_K_M | llama3 | constrained under `required`; **parsed only under `auto`** | T0: 5.7–6.5% of auto calls unusable, unchanged |
| Gemma 4 | E2B, 12B, 26B-A4B (HF and GGUF) | gemma4 | **parsed only** | bespoke call syntax, no JSON form |
| Gemma 3 | gemma-3-4b-it, gemma3-1b | gemma3 | none | template has no tool form in goinfer |
| gpt-oss | gpt-oss-20b (HF, MXFP4 GGUF), 120b partial | harmony | none | harmony has no tool support in goinfer |
| SmolLM3, Granite 4.2, LFM2.5, Olmo 3, Olmo Hybrid | smollm3-3b, granite-4.2-3b, lfm25-2.6b, olmo3-7b-think, olmo-hybrid-7b | chatml | *constrained, UNVERIFIED* | local copies carry **no template at all**; they resolve to chatml only by the vocab fallback. `chat.go` declines SmolLM3's and Olmo 3's released templates on purpose (M-36), and LFM2's native call syntax is not Hermes JSON — so the released checkpoints may resolve differently. Not a claim until checked against the released templates |
| Granite-4.0-H | granite-hf, granite-4.0-h-tiny Q8_0 | none | none | template (in `chat_template.jinja`, read since finding 1's fix) not recognised: Hermes-style `<tools>` inside `<|start_of_role|>` turns — needs its own renderer |
| Laguna | laguna-xs2, laguna-xs21 (HF, GGUF) | none | none | template not recognised (xs2's is in `chat_template.jinja`, read since finding 1's fix) |
| Command-R, Command-R7B | aya-expanse-8b, command-r7b | none | none | list-form `chat_template` (named templates) is ignored by design (finding 2) |
| DeepSeek-V2 | deepseek-v2-lite (HF, GGUF) | none | none | template not recognised |
| GLM-4.5/4.6 | GLM-4.5-Air Q2_K | none | none | template not recognised |
| Llama 4 | Llama-4-Scout GGUF | none | none | template not recognised |
| Phi-3 / Phi-4 | Phi-3-mini-4k Q4 GGUF | none | none | template not recognised |
| Ministral 3 | ministral3-3b-bf16 | none | none | local copy has no template; the released one routes to `ministral` (no tools) |
| Mistral | mistral-7b-instruct-v0.1 Q4_K_M | none | none | v0.1 has no tool form; v0.3 (`[TOOL_CALLS]`) would be constrained — no v0.3 checkpoint here |
| DeepSeek-V3, Kimi K2, Ling 3.0, GPT-2, InternLM2, Qwen3-VL, Spark-X2.5, Mixtral | — | not checked | — | no local checkpoint with a loadable tokenizer.json/GGUF template |

**What can be claimed.** A tool call cannot be malformed or name an unsupplied tool on the **Qwen families** (Qwen2 through Qwen3.8,
MoE and VL variants), **Nemotron-3-Nano** and **Mellum2**, under `required` and under `auto`; on **Llama 3.x** under `required`/named
choice only. That is the scope any README or recipe line should use, and no wider.

## Findings

1. **`chat_template.jinja` was never read — FIXED 2026-09-24.** Recent `transformers` saves the template as a separate
   `chat_template.jinja` beside `tokenizer_config.json`; the byte-level tokenizer loader read only the config's `chat_template` key, so
   such a checkpoint fell through to the vocab fallback or to raw completion, and a released SmolLM3/Olmo 3 shipped that way would have
   bypassed the M-36 fingerprints. `readTokenizerConfig` now falls back to the file (the key still wins when both exist; the no-sibling
   `.giw` mode still reads nothing — `tokenizer/chat_template_jinja_test.go`). Re-running this census after the fix changed exactly one
   resolution: mellum2-unq chatml → mellum2. Granite-4.0-H and Laguna XS.2 now reach `chat.Detect` with their template, and it is not
   recognised, so they stay on raw completion. SentencePiece-mode HF tokenizers (Gemma) never read a template from the config at all;
   they resolve through the vocab fallback, which is correct for them, and are unchanged.
2. **List-form `chat_template` is ignored** (Cohere's named `default` / `tool_use` / `rag` templates) — a documented choice in
   `tokenizer/bytelevel.go`, recorded here because it is why both Command-R families have no chat template at all.
3. Several GGUF templates are not recognised by any fingerprint (DeepSeek-V2, GLM-4.5, Llama 4, Phi-3, Granite-4.0-H, Nemotron-Nano
   9B): raw completion, no tools.

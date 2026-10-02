# D10 — Clef-flash: prior art, the encoder and head contract, the merge check, the reference fixture (2026-10-02)

Item D10 of [`task-constrained-confidence.md`](../tasks/task-constrained-confidence.md) (Route C). No goinfer code: reading, a merge check, a prior-art sweep, and the reference fixture. Everything below was read from the pinned release or measured on nobara-pc on 2026-10-02 (PDT).

## Pins

| | |
|---|---|
| model | `Cloudflare/clef-flash`, revision `17f0b0ad64efb65d273590632833508766b2aae6` (2026-10-01 15:23 UTC), Apache-2.0, not gated, 19.08 GB |
| the custom module | `joint_schema_model.py`, 576 lines, sha256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` |
| the head | `joint_head.safetensors`, 243,538,016 bytes, sha256 `19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0`, 122 tensors, **121,762,820 parameters**, all bf16 |
| backbone shards (verified against Hugging Face's LFS sha256) | `model-00001..4-of-00004.safetensors`: `8b45a8e9…`, `7590856c…`, `e6eac246…`, `9fcecc65…`; 760 tensors in the index |
| the larger release | `Cloudflare/clef`, revision `2f3de3dd85f379784083b0814d997ab627200f0c`, 54.99 GB in 12 shards, same module and head config family |
| where it lives here | `/srv/models/clef-flash` (the archive: `/home` has 9 GB free). Fixture generation reads it offline; any later TIMED run needs it on local disk, which needs space on `/home` first |

## The encoder contract (`encode_record`, read in full)

A request is `{state, questions, [images, videos, id]}`; each question is `{type, instructions?, criteria?}`, `type` one of `noul`, `choice`, `score`.

- **`render(value)`:** a string is returned as is; anything else is `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`. Go's `encoding/json` differs in key order, separators and escaping, so D12 needs a Python-compatible renderer.
- **Options (`question_options`):**
  - `noul`: always `[("true", d), ("false", d)]`, **true first**; the defaults are "The proposition is true or the answer is yes." / "...false or the answer is no."; a request's `criteria` can override those two descriptions only.
  - `choice`: `sorted((str(key), value) ...)`, **alphabetical by key, not in the order sent**. The head's logits come in that order; `systemone_answer` maps them back by option id.
  - `score`: `str(index)` for each entry of `criteria`, a list of descriptions.
- **Assembly.** `input_ids = prefix + [media] + state + schema + suffix`, and **each piece is tokenized separately** (`tokenizer(text, add_special_tokens=False).input_ids`), so there are no merges across a boundary and the token ids differ from tokenizing the whole string:
  - prefix: `<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n`
  - state: `render(state)`
  - schema: `\n\nSCHEMA FIELDS:\n`, then per question `\nFIELD {i}\nID: {id}\nTYPE: {type}\nINSTRUCTION: ` + `render(instructions)` (the question id if instructions is missing or empty) + `\nALLOWED OPTIONS:\n`, then per option `OPTION {k}: ` + `render({"option_id": id, "description": d})` (the description key only when not None; the JSON keys come out sorted, so `description` first) + `\n`, then `END FIELD\n`
  - suffix: `\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:`
- **Spans** are token indices into the whole sequence: the question's span is the tokens of `render(instructions)`; an option's span is the tokens of its `render(...)` JSON (not its `OPTION k: ` label). Spans are half-open and shifted by `len(prefix) + len(state)`.
- **Truncation:** `max_length` defaults to **16,384** (the blog claims a 64k window); the state is cut to `max_length - fixed_length` tokens, **keeping its first tokens** (JEV kept the first 60% and last 40%); a schema that alone exceeds `max_length` is a ValueError.
- **Media:** placeholders `<|vision_start|><|image_pad|><|vision_end|>` and the processor's output go in after the prefix. Out of scope until multimodal.md P8a.
- Verified by instrumenting the module's own tokenizer calls (`encoder.jsonl`): on the first fixture item the fragments are 237 schema tokens, a 36-token prefix, an 18-token suffix and a 298-token state, 589 in all, and they add up exactly.

## The head contract (`JointSchemaHead`, read in full)

Config (`joint_head_config.json`): `hidden_size 4096, width 1024, routing_layers 2, layers 4, heads 16, feedforward 4096`. Dropout is 0 and the model is in eval mode.

Input: the text model's `last_hidden_state` at **every position**, which is **after the backbone's final RMSNorm** (`ClefModel.forward` calls `base_model.model.language_model` for text-only input), plus `input_ids` and the lm_head weight (the backbone is untied: `tie_word_embeddings: false`). For one record, with `h = hidden_norm(last_hidden_state)` (an affine LayerNorm over 4096, eps 1e-5, with bias):

1. `memory = memory_projection(h)` (bias-free, 4096 to 1024), shape `[1, L, 1024]`; `g = h[-1]` (the last position, the `:` of `DECISIONS:`).
2. `qv[q] = mean(h[question_span])`; `ctx[q][o] = mean(h[option_span])`; `lex[q][o] = mean(lm_head_weight[input_ids[option_span]])` (the **raw** lm_head rows of the option's JSON tokens, not normalised).
3. `option_query = option_context_projection(ctx) + option_lexical_projection(lex) + option_question_projection(qv[q])`, all options of all questions concatenated, shape `[1, N, 1024]`.
4. Two `EvidenceRoutingLayer`s, each pre-norm: `x = x + MHA(query_norm(x), memory_norm(memory), memory_norm(memory))`; `x = x + FF(feedforward_norm(x))`, with FF = `Linear(1024,4096)`, exact GELU, `Linear(4096,1024)`. 16 heads, `torch.nn.MultiheadAttention` with bias (fused `in_proj_weight [3072,1024]`), no mask.
5. Split back per question: `routed[q]`. `base_field[q] = question_projection(qv[q])`; `w = softmax(routed[q] @ base_field[q] / sqrt(1024))` over the question's options; `summary[q] = sum(w * routed[q])`.
6. `fields = base_field + option_summary_norm(summary) + global_projection(g) + type_embedding(type_id)` (type ids noul 0, choice 1, score 2), shape `[1, Q, 1024]`.
7. Four `TransformerDecoderLayer(d_model 1024, nhead 16, ff 4096, activation "gelu", batch_first, norm_first=True)` called as `layer(fields, memory)`: `x = x + self_attn(norm1(x))` (over the questions, no mask); `x = x + multihead_attn(norm2(x), memory)` (**the memory is NOT normalised here**); `x = x + FF(norm3(x))`; LayerNorm eps 1e-5. Then `field = field_norm(fields)`.
8. Per question, per option: `anchor = normalize(qv[q] + g)` and `lexical_anchor = normalize(lex)` (both 4096-wide); `prior = exp(min(prior_logit_scale, ln 100)) * (lexical_anchor @ anchor)`; `o = option_norm(routed)`; `cosine = cos(field, o)`; `residual = residual_scorer(cat[field, o, field*o, |field-o|])` (`Linear(4096,1024)`, GELU, `Linear(1024,1)`); `joint = exp(min(joint_logit_scale, ln 100)) * cosine + residual`; **`logit = prior + sigmoid(residual_gate) * joint`**.
9. A softmax per question over its options, in the encoder's option order.

`systemone_answer` (the reference's own response shaping): `noul` is `round(P(true), 4)`; `choice` is the argmax over the REQUEST's criteria order (first maximum wins), **`confidence` is the top probability**; `score` is the expected level, `confidence` the top probability. goinfer's current `/v1/systemone` reports `confidence` as the margin over uniform, `(n*p - 1)/(n - 1)`, because TypeSafe's docs demo uses that form; **which one the `clef` route reports is a decision D13 must make explicitly** (the two differ for every kind except a certain answer).

## The merge check (HTTP range reads of Clef-flash tensors against local Qwen3.5-9B)

Clef-flash has 760 tensors against Qwen3.5-9B's 775: the 15 absent are the multi-token-prediction layer (`mtp.*`), dropped. Relative L2 difference of the tensors compared:

| tensor | rel L2 |
|---|---|
| GDN `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` (layer 0) | 0.057, 0.055, 0.059, **0.203**, 0.076 |
| full attention `q_proj`, `k_proj`, `v_proj`, `o_proj` (layer 3) | 0.071, 0.073, 0.083, 0.061 |
| MLP `gate_proj`, `up_proj`, `down_proj` (layers 0, 3, 31) | 0.081, 0.082, 0.057 / 0.051 |
| GDN `linear_attn.norm.weight` | 0.0014 (max 0.002: bf16 resolution; trained or merge rounding, not told apart) |
| `A_log`, `dt_bias`, `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, the final `norm`, the visual merger norm | bit-identical |
| `embed_tokens`, `lm_head` | bit-identical **over the first 2,000 of 248,320 rows only** (a full-tensor comparison was not made) |

So the adapters are merged into every linear projection including the GDN ones, nothing else moved, and **the checkpoint loads as a plain `qwen3_5` model: D3 is not needed.**

## Prior art (swept 2026-10-02)

- **Cloudflare:** hosted on Workers AI; the open weights and the reference module above; the model card's own results (a hosted-benchmark table; Clef-flash median request latency 38.8 ms, the table does not state the hardware for that row). Usage tested with torch 2.11 and transformers 5.10.2 on a single H200.
- **llama.cpp:** [ggml-org/llama.cpp#29831](https://github.com/ggml-org/llama.cpp/pull/29831), "model: add support for clef decision model (text-only)", by ngxson, **open and unmerged** (filed 2026-10-01 23:36 UTC, updated 2026-10-02 20:35 UTC; 24 files, +1323). It adds `src/models/clef.cpp` (the head in ggml), `conversion/clef.py`, `tools/server/server-decision.cpp` (`/v1/systemone`) and a batch API call marking each token's role (ordinary, question of kind noul/choice/score, option). Text-only (vision waits on #29622) and one sequence per batch. `ggml-org/Clef-GGUF` and `ggml-org/Clef-Flash-GGUF` say they require it. `bartowski/Cloudflare_clef-flash-GGUF` is backbone-only quantizations.
- **MLX:** `mlx-community/clef-{flash,}-{4,8}bit` and others ship a loader (`clef_mlx.py`) that runs backbone plus head. Its own quality check (a sampled Decision Index, 15,915 answers): the 4-bit model gives **the same top answer as bf16 on 96.4%**, mean max absolute probability difference 0.040, about one index point lost; latency on an M5 Max 0.31 s at 1k tokens and 7.0 s at 16k for the 9B 4-bit.
- **Ollama:** the library lists `clef` and `clef-flash`; four related issues (one closed: "add clef support via llama-server"; open: MLX System One support, Strands Decider, per-type temperatures for `/v1/systemone`).
- **vLLM / SGLang:** no hits for vLLM; SGLang not searched.
- **Others seen:** ONNX, EXL3, NVFP4, FP8 and further GGUF conversions on Hugging Face.

## The fixture

`scripts/pin_clef_d10.py` runs the release's own code unmodified (the module and head are sha256-checked first) on the 150 D0 items (`testdata/decisions/items.jsonl`) re-expressed as Clef records. **How an item becomes a record** (a D10 decision): one question with id `q`; `instructions` is the item's question text; the state is the item's state parsed as JSON when it parses and the raw string otherwise (some are plain text); noul has no criteria; choice has `criteria {label: None}`, the item's own labels as option ids with no invented descriptions; score has `["0".."5"]`.

- `testdata/decisions/clef/records.jsonl` (sha256 `5942ccb997012d69c37945ef8cf3a2dd1974f62009ef14d9a668f5d4c49f9b7c`) and `encoder.jsonl`: for every item the official encoder's token ids and spans and every fragment it tokenized. 150 prompts, 44,823 tokens (min 180, median 241, mean 299, max 683).
- Environment: torch 2.14.0+cpu and transformers 5.16.1 (the reference venv), against the release's tested 2.11 and 5.10.2; PIL is not installed there, so the fixture is text-only and uses `AutoTokenizer` (the release's `processor.tokenizer` is this tokenizer). A numerics drift between those transformers versions is possible and is the first suspect if D11 or D12 disagree with the goldens.
- **Probe (bf16, CPU, 2 items): 147 s for 589 tokens and 50 s for 238, about 0.23 s per prompt token.** Both gave valid distributions that agree with gold and match the encoder dump's token counts. bf16 is slow because this CPU has no bf16 hardware; the full 150 items at bf16 is about 2.9 hours, and f32 is expected to be roughly 5 times faster (D0 measured 45 ms per token at f32), about 35 to 50 minutes.
- **Queued for tonight** (`run-clef-fixture.sh`, resumable): f32 over all 150 items with `last_hidden_state` saved for the 3 shortest items (the goldens D11 and D12 need), then bf16 in two halves of 75 (each about 95 minutes, under the 3-hour job limit). The probabilities, the f32 hidden goldens and the environment records land in `~/goinfer-bench/decisions-d10/out/` and are committed under `testdata/decisions/clef/` after review, with files over 1 MB gzipped.

## What D10 leaves for D11 to D14

- **D11:** the all-positions hidden seam as written in the task doc is right (final-norm hidden at every position). The head applies its own LayerNorm to it, so the seam must return the post-final-norm values, as HF's `last_hidden_state` does.
- **D12:** the head is 121.8M parameters, about 487 MB at f32 (the task doc estimated 60M); it is still small next to the backbone, but six of its attentions (2 routing layers and the 4 decoder layers' cross-attention) run over the whole sequence (`memory` is L x 1024), the 4 decoder self-attentions run over the questions, and the reference's per-question loop is not batched. The encoder needs a Python-compatible `render`, per-fragment tokenization and the alphabetical option sort. The gate (head isolated, given the reference's own hidden states) needs the f32 hidden goldens queued above.
- **D13:** the two open choices are the `confidence` formula and the decision-model default quantization (already made for JEV: `int8int8`). The MLX figure (96.4% top-answer agreement at 4-bit) is the only external comparator for a quantization result.
- **Risks:** the module is version one of a custom Python file (pin the revision, refuse unknown keys and shapes); the context default (16,384) against the blog's 64k; llama.cpp's open PR is an independent port that could be read as a cross-check but is not an oracle.

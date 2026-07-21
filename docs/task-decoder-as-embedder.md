# Task — Decoder-as-embedder (`qwen3-embedding`, `embeddinggemma`)

**Origin:** aikit `docs/task-embedding-coverage.md`, **Bucket B**. Buckets A (encoder-side
loaders/pooling/tokenizers) and the multilingual stack are done and certified in aikit;
this is the remaining piece and it lives **here**, not in aikit — hence this doc.

**Shape:** these two models are *causal decoders used as embedders*. goinfer already runs
`qwen3` and `gemma3`, and already serves `/v1/embeddings`. The only genuinely new piece is a
**pooling head + instruction-prefix convention**.

> existing decoder forward → pool (last-token / mean) → L2-normalize → out the existing
> `/v1/embeddings`

---

## 1. What already exists (do not rebuild)

`cmd/serve/embeddings.go` already handles, for free, whatever backs the embedder:

- `input_type` → query/document asymmetry (`"query"|"search_query"` vs default document)
- `dimensions` → Matryoshka truncate **+ renormalize**
- `encoding_format` → `float` | `base64` (LE float32)
- L2-normalization (`postprocess`), OpenAI-shaped response, `usage` counting
- batch `input` (string | []string)

`cmd/serve/openai.go` holds the embedder as `server.embed`, plus `embedTok`, `embedID`,
`embedDim`.

## 2. The seam

The embedder is an **aikit interface**, `encoder.Encoder` (`aikit/encoder/model.go`):

```go
type Encoder interface {
    Encode(text string, isQuery bool) ([]float32, error)
    EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error)
    HiddenDim() int
}
```

Implement that over a loaded goinfer decoder and assign it to `server.embed`. Everything
downstream is unchanged. **This is the whole integration** — resist widening it.

## 3. Model specifics — verified from the real configs

aikit's task doc carries an explicit caveat that its buckets were formed "by name and
reputation, not by reading each `config.json`" — and that already bit us once
(`multilingual-e5-small` turned out to be `model_type: bert`, not XLM-R). So, read from HF:

### Qwen3-Embedding-0.6B — **verified**
- `architectures: ["Qwen3ForCausalLM"]`, `model_type: qwen3`, hidden **1024**, **28** layers
- `1_Pooling/config.json`: **`pooling_mode_lasttoken: true`**, `include_prompt: true`
- `modules.json`: `Transformer → Pooling → Normalize`
- `config_sentence_transformers.json` prompts:
  - **query**: `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"`
  - **document**: `""` (empty)

That maps *directly* onto the existing `input_type` asymmetry: `isQuery=true` prepends the
Instruct prefix; document prepends nothing. Note the query prefix is part of the pooled input
(`include_prompt: true`) — it is **not** stripped before pooling.

### embeddinggemma-300m — **NOT verified: gated**
Config fetches return HTTP errors (gated repo; needs accepted license / HF auth). **Do not
assume mean pooling.** Fetch `config.json` + `1_Pooling/config.json` +
`config_sentence_transformers.json` first and record what they actually say, exactly as above.

## 4. Three traps (each is a silent-wrong)

1. **Concurrency.** `embeddings.go` deliberately runs with **no mutex** — the comment is
   explicit: *"The encoder is goroutine-safe for concurrent Encode, so `/v1/embeddings` is
   served without a mutex (the per-model mutex guards only the shared decoder)."* A
   decoder-backed embedder is **not** goroutine-safe. It must serialize internally (own mutex)
   or you reintroduce concurrent mutation of shared decoder/KV state under load. Decide what
   `EncodeBatch`'s `concurrency` argument means for it (likely: clamp to 1).

2. **Token counting.** `countEmbedTokens` uses `s.embedTok` (an aikit `*embed.Tokenizer`) via
   `encoder.EncodeQuery`/`EncodeDoc`. A decoder uses its **own** tokenizer, so `embedTok` will
   be nil and `usage.prompt_tokens` silently reports **0**. Wire a decoder-side count or make
   the nil path explicit — don't let it quietly zero.

3. **Last-token pooling + padding.** Last-token pooling is position-sensitive: in a batch with
   right-padding, "last token" must be the last **non-pad** token, not the last slot. Wrong
   here returns plausible-looking vectors that are simply incorrect — exactly the silent-wrong
   class the parity gates exist to catch.

## 5. Parity discipline (non-negotiable)

Same bar the encoder work held; see `docs/parity-hunt-playbook.md` and
`docs/parity-coverage-policy.md`.

- **Vector gate** — cosine vs the HF `sentence-transformers` reference over a fixed sentence
  set. Bar **> 0.9999** (aikit's certified embedders all land at 1.000000).
- **Retrieval gate** — encode a small fixed corpus + queries, assert **top-k ordering** matches
  the reference. Cosine can be high while ranking is wrong.
- **Break-it-first** — perturb (a) pooling last-token → mean, and (b) drop the query Instruct
  prefix. Both gates must go **RED**, then revert. A gate that can't fail isn't a gate.
- **Pin the golden** with a script mirroring aikit's `scripts/pin_*.py`: dump `input_ids` +
  the reference embedding from the real model; commit the golden, gitignore the weights.

## 6. Definition of done

- [x] Decoder-backed `encoder.Encoder` implementation + wiring so a qwen3-embedding model can
      be served on `/v1/embeddings`.
- [ ] Qwen3-Embedding-0.6B: vector + retrieval gates green, with break-it-first.
      **Scaffolded, not certified** — needs the checkpoint + a pinned golden (see §7).
- [x] `usage.prompt_tokens` correct (trap 2) and concurrency safe under parallel requests (trap 1).
- [~] embeddinggemma: configs read and recorded; certify, or record why it's deferred.
      **Deferred, reason recorded** (§7) — still gated.
- [x] No regression to the existing encoder-backed `/v1/embeddings` path.

---

## 7. Status (2026-07-20, Mac session)

### Shipped

- **`decoder.HiddenLast(ids)`** (`decoder/embed.go`) — the missing seam. It runs the stack causally
  in a fresh KV cache and returns the last token's hidden state **after the final norm**, i.e.
  exactly HF's `last_hidden_state[:, -1, :]`, and stops before the LM head (an embedder never needs
  the logits, and for a tied 151k-row head that projection is the most expensive part of a token).
  Neither existing seam could serve this: `forward` consumes the hidden state in place on its way to
  logits, and `ForwardCapture` returns per-LAYER residuals, which are *pre*-final-norm.
  Guards: empty sequence, out-of-vocab ids, and the same own-`runLayers` arch decline
  (gemma4/qwen35/granite/nemotron/mla/llama4) `ForwardCapture` uses — an error, never a plausible
  wrong vector. **Padding trap (§4.3) is structurally avoided**: it pools one unpadded sequence, so
  "last token" is always the last real token; there is no batch slot to get wrong.
- **`decoderEmbedder`** (`cmd/serve/decoder_embedder.go`) — satisfies `encoder.Encoder`
  structurally, so everything downstream of `server.embed` is untouched.
  - *Trap 1 (concurrency):* own mutex around every call. `EncodeBatch`'s `concurrency` argument is
    deliberately ignored (clamped to 1) — one decoder, one KV cache; fanning out would corrupt it.
  - *Trap 2 (token counting):* new `embedTokenCounter` interface in `embeddings.go`, preferred over
    `s.embedTok` (which is nil for this embedder and silently reported `prompt_tokens: 0`).
  - Selected by `-embed-model` pointing at a **`.gguf` file**; an HF **directory** still takes the
    aikit encoder path. No new flag.

### Config facts verified this session (adds to §3)

Read from the live repo, same discipline as §3:

- `tokenizer_config.json`: `add_bos_token: false`, **`add_eos_token` absent**, `eos_token:
  "<|im_end|>"`, `pad_token: "<|endoftext|>"`, `model_max_length: 131072`.
- The model card appends **no EOS/EOD** in either the sentence-transformers or raw-transformers
  example, and `last_token_pool` selects the last *non-padding* token.
  → **the model eats exactly the `prompt+text` tokens, no special tokens on either end.** This
  materially affects last-token pooling and was not previously recorded.
- `config_sentence_transformers.json` re-confirmed verbatim (query Instruct preamble, document `""`).
- **No `sentence_bert_config.json` (404)** → no truncation shorter than `model_max_length` applies,
  so the embedder truncates nothing by default.

### Gates

Running now (mechanical), against **base Qwen3-0.6B**, which shares Qwen3-Embedding-0.6B's
architecture and exact geometry (hidden 1024 × 28 layers):

- `decoder/embed_test.go` — shape, bit-identical determinism, **last-token pooling proven** (changing
  the final token, or extending the sequence, must move the vector), context sensitivity, guards.
- `cmd/serve/decoder_embedder_test.go` — interface fit, query-prefix asymmetry (and that the prefix
  reaches the pooled vector), `EncodeBatch` == N× `Encode` (no KV leak between items), **concurrent
  `Encode` under `-race`** matches the serial baseline, honest token counts, empty-input rejection.

Scaffolded, skipping until certified:

- `scripts/pin_qwen3_embedding.py` → `testdata/qwen3_embedding_golden.json`.
- `cmd/serve/qwen3_embedding_parity_test.go` — vector gate (cosine > 0.9999), retrieval gate (top-1
  ordering), tokenizer agreement (kept separate so a tokenizer drift reports as itself), and
  **break-it-first** with two executable perturbations: pool the wrong position, and drop the query
  Instruct prefix — each must push cosine below the bar.
  `TestQwen3Embedding_gatesAreWired` fails loudly if the golden is pinned but the checkpoint is
  missing, so these can never silently no-op.

### Blocked on

1. **Qwen3-Embedding-0.6B checkpoint** — not present locally (only base Qwen3-0.6B). The base model
   cannot satisfy the vector gate: same architecture, different weights.
2. **HF reference env** — no `torch` / `sentence_transformers` installed, so the golden cannot be
   pinned here. No metrics were invented in their absence.
3. **embeddinggemma** — still **gated**: `1_Pooling/config.json` returns **HTTP 401** (verified
   2026-07-20). Its pooling mode therefore remains *unknown* and, per §3, must not be assumed to be
   mean. Deferred until the license is accepted / HF auth is available; the `pool` choice is the
   one design question its certification will settle.

To certify: place the checkpoint in `$HOME/models/`, `pip install sentence-transformers`, run
`scripts/pin_qwen3_embedding.py`, then `go test ./cmd/serve -run TestQwen3Embedding -v`.

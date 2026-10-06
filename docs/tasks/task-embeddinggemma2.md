# Task — add EmbeddingGemma 2 (`embeddinggemma2`)

**Status:** 2026-10-06: Gate 0 done, Gate 1 passed, Gate 3 built; Gate 2 pre-registered below and queued for the
night of 2026-10-06 (owner: "build it today, release tomorrow"). Work is on the branch `embeddinggemma2`, kept off
`main` while v0.21.0 is tagged.
**Venue:** mac for Gate 0–2 (a 270M text backbone fits anywhere); nobara-pc only if Phase 2's
audio tower is attempted.
**Scoped in:** this doc. Related, not superseded: `docs/multimodal.md` P7 (Gemma 4 vision/audio
towers) and `docs/tasks/task-embed-and-harness-ux.md`.

---

## What it is

Google's EmbeddingGemma 2, released and on HuggingFace as `google/embeddinggemma-2`. A natively
multimodal embedding model — text, code, images, video and audio into one shared space — built as
**modular encoders** so the modalities can be loaded selectively.

From the model card (`ai.google.dev/gemma/docs/embeddinggemma/model_card_2`, read 2026-10-06):

| | |
|---|---|
| total params | 740M |
| text backbone | **270M** (130M transformer + 140M embedder) — loadable alone |
| vision encoder | 170M, optional |
| audio encoder | 300M, optional |
| layers / d_model / d_ff | 24 / 512 / 2048 |
| attention | GQA |
| FFN activation | gated FFN with **GELU** |
| vocab | 262,144 |
| context | 8,192, shared across modalities |
| pooling | mean pool, then a 512→768 projection |
| output dims | 768 native, Matryoshka truncation to 512 / 256 / 128 |
| task prefixes | required for text quality; asymmetric for retrieval (separate query/document), symmetric for classification and clustering |

**Every number above is from the model card, not from the weights.** Before writing any code,
read the real `config.json` and modeling file from the HF repo and correct this table in place.
The Spark-X2.5 task found three of its own "we already have" claims wrong this way
(`docs/tasks/task-spark-x2-5.md`, Results) — the same check applies to the claims below.

## Why this one is worth the slot

The text backbone is Gemma-shaped, and the pieces it needs mostly landed for other reasons.
Verified in the tree at `53a241a`:

- **Gated FFN with exact GELU — already there, on both paths.** `decoder/mlp.go:379` `gegluExact`,
  reached via `ActGelu` in the decode-path switch (`decoder/mlp.go:524`) *and* in the batched-prefill
  switch (`decoder/forwardn.go:718`, calling `gegluExact` at `:720`). Both were checked: the second one is
  the copy Spark-X2.5 had to add after its own three gates missed that `Generate()` would crash on
  any multi-token prompt without it. EmbeddingGemma 2 is the second family to use this activation
  and inherits the fix.
- **Gemma's 262,144 vocab and tokenizer — already there** (`multimodal/gemma3_block.go:7`,
  `metal/softcap_test.go:24`).
- **A bidirectional attention mask over a span — already there, for a different reason.**
  `decoder/forwardn.go:265` and `:1693` set up a bidirectional mask over image blocks
  (`SetImageBlocks`, handled in `attendBatchedHeads`). A fully bidirectional encoder is the
  degenerate case of that machinery — the whole sequence as one block — which makes this a
  generalization rather than a new mask implementation. **Confirm this is the shape needed; see
  Gate 0.**
- **An embedder entry point and a parity pattern for one — already there.**
  `internal/serveapp/decoder_embedder.go` wires a causal decoder in as the embedder via
  `-embed-model` pointing at a file; `internal/serveapp/qwen3_embedding_parity_test.go` is the
  model for how an embedder gets gated.

What is **not** in the tree:

- **Mean pooling.** The decoder-embedder path pools with `HiddenLast` (`decoder/embed.go:37`),
  whose own doc comment says it "runs ids through the layer stack **causally**" (`:19`). A grep for
  mean pooling finds only a comment in `decoder/embed_test.go:70`. So goinfer's existing embedder
  is causal and last-token — both axes differ from what this model does, which is the substance of
  Gate 0. The aikit encoder path may already mean-pool; Gate 0 settles that.
- **The 512→768 output projection.** Small, but it is a real tensor to load and apply after
  pooling, not a reshape.
- **Matryoshka truncation.** Slice the vector and renormalize. Cheap, but it needs an API surface
  (Gate 3).
- **Any task-prefix concept.** `/v1/embeddings` has no task-type field. This is an API design
  decision, not a detail — see Gate 3.

Three projects use the result, which is why this earns a slot over another text-only embedder:
goinfer serves it at `/v1/embeddings`, aikit's RAG and code-search primitives consume it, and
skiff's "ship the index, not the server" story gets materially better from a 270M model with
128-dimension Matryoshka vectors.

## Results, 2026-10-06

### Gate 0: what the real checkpoint is (read at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`)

Read from the repo's `config.json`, `config_sentence_transformers.json`, `modules.json`, `1_Pooling/config.json`,
`tokenizer.json` and the safetensors header (1,376 tensors, 413 outside the vision and audio towers), and from
`transformers` 5.19.0's `models/embedding_gemma2/modular_embedding_gemma2.py` (the repo ships no modeling file; the
config says `transformers_version` 5.18.0.dev0, and 5.16.1, the version installed here, has no `embedding_gemma2`).

- **Bidirectional on every layer** (`is_causal = False`). The sliding layers attend over `|q − k| <= sliding_window`
  (512, an inclusive radius: a 1,025-wide window), the full layers over the whole input. Mean pooling follows. So the
  "re-scope down" rule does not fire.
- **The table above was wrong in four places.** The backbone is Gemma 4-shaped, not Gemma 3: Gemma 4's RMSNorm (the
  weight used as is, not `1 + w`), per-head q and k RMSNorm and a v RMSNorm with no weight, attention scaling 1.0; the
  FFN activation is `gelu_pytorch_tanh`, **not** the exact GELU the "already there" bullet assumed; every sixth layer
  is full attention with **head dim 512 and 1 KV head** (the others 256 and 2), RoPE theta 1e6 on full layers and 1e4
  on sliding ones; and the "140M embedder" is a projection-only per-layer-input (PLE) block: `RMSNorm(W · emb ·
  hidden^-0.5)` split per layer, mixed in after each MLP as `x + RMSNorm(proj(gelu(gate(x)) * ple_i))`, then
  `x *= layer_scalar` (a stored per-layer value, not 1). The 512 → 768 projection is the model's own last op
  (`embedding_projection`), before pooling.
- **The sentence-transformers pipeline:** Transformer → Pooling (mean, `include_prompt: true`) → Normalize. The
  tokenizer writes `<bos> … <eos>` around the input and the mean includes both. Twenty named prompts
  (`query` = `task: search result | query: `, `document` = `title: none | text: `, and STS, Classification, …);
  `default_prompt_name` is null, so sentence-transformers applies **no** prompt unless asked.
- **Where it lands: goinfer, as its own package (`embeddinggemma2/`), not aikit and not `decoder.Load`.** aikit's
  encoder is BERT-shaped (CLS, WordPiece); goinfer's decoder path is causal and built around a KV cache and decode
  steps an encoder has no use for, and putting a bidirectional encoder through it would mean edits to `forwardn.go`,
  `arch.go` and the other files frozen for v0.21.0. The package reads the text half of the checkpoint with aikit's
  safetensors reader and goinfer's `tokenizer.Load`, runs one CPU float32 forward, and implements aikit's
  `encoder.Encoder`, so serve's `/v1/embeddings` takes it with no change to how other encoders load. Not covered:
  quantization (f32 only) and GPU backends.
- **The census check does not apply.** `censusList` exercises `decoder.Load`'s serialization, which this encoder
  does not use.

### Gate 1: passed (tiny fixture, CPU float32)

`scripts/pin_embeddinggemma2_tiny.py` (transformers 5.19.0, torch 2.14.0) builds a random-weight model with the
real modeling code under the real checkpoint's tensor names and config layout: 6 layers (4 sliding, 2 full with a
wider head and 1 KV head), a sliding radius of 3 against inputs of 1, 2, 9 and 17 tokens, and every norm weight and
`layer_scalar` random rather than 1. `embeddinggemma2/model_test.go`:

- every layer's input, the projected last hidden state, the pooled vector and the normalised embedding: cosine
  1.000000000 in all four cases (bar 0.9999), each checked on its own;
- the sliding mask is the reference's cell for cell;
- **accuracy against float64:** the doc's "maxAbs in the 1e-7 band" was not met (embedding max |diff| to the float32
  reference 6e-8 to 2.4e-6), and measuring why moved the bar. The same weights run in float64 show that the float32
  *reference* is itself 6e-8 to 2.4e-6 from the true value, and goinfer is as close or closer (goinfer ÷ reference
  distance to float64: 1.02, 0.99, 0.36, 0.76). The error grows smoothly by about 4e-7 relative per layer, with no
  single lossy op. An absolute bar on the float32 output cannot tell goinfer's rounding from the reference's, so the
  gate is now: cosine >= 0.9999, and the embedding's max |diff| to float64 at most 1.5× the float32 reference's own.
  The golden carries the float64 embedding.
- **Twelve planted defects, all red:** layer_scalar ignored, Gemma 3's `1 + w` norm, a window one wider, v norm
  skipped, PLE skipped, a `1/sqrt(d)` scale, last-token pooling, RoPE thetas swapped, exact GELU, a causal mask, the
  embedding scale omitted, and the PLE scale omitted. The last is caught only by the float64 bar, at its margin
  (1.06e-7 against 0.97e-7): the RMSNorm right after that scale is scale-invariant, so omitting it acts only
  through `eps`.

An exploratory one-text smoke on the real checkpoint by day (not the gate): the 18 ids of a query-prompted input equal
sentence-transformers', and the embedding's cosine is 0.999999998, max |diff| 1.0e-7.

### Gate 3: built

- `-embed-model` takes an EmbeddingGemma 2 directory (or `hf:google/embeddinggemma-2:safetensors`;
  `pull.EncoderLoads` accepts `embedding_gemma2`). `-embed-quant` other than f32 is refused by name.
- **Task prompts: option 2, a non-standard `task` field, decided by the owner 2026-10-06** with **no prompt as the
  default**, because that is what sentence-transformers applies by default, so a client that knows nothing about
  prompts gets the reference's own output. `task` names any of the model's 20 prompts, or `"none"`; without it, the
  existing `input_type` maps `query` and `document` onto the model's own two; with neither, no prompt. The prompt
  applied is echoed in every response (`goinfer_task: {name, prompt}` and an `X-Goinfer-Embedding-Task` header), and
  an unknown task is a 400 naming the model's tasks.
- **`dimensions` takes exactly 768, 512, 256 and 128**, and any other value is a 400 naming them, pending Gate 2's
  truncation reading (rule below).
- Tests: `internal/serveapp/embeddinggemma2_embedder_test.go` (prompt choice, echo, refusals, widths, the quant
  refusal, the checkpoint dispatch), on a stub; a planted default-to-document fails it. A serve run on the real
  checkpoint answered all of these by day (exploratory).

### Gate 2: pre-registered 2026-10-06, before it runs

- **Instrument:** `docs/measurements/embeddinggemma2-2026-10-06/run-gate2.sh` on the Mac night queue, from a worktree
  pinned at the branch commit that carries this section. (1) `scripts/pin_embeddinggemma2_real.py` with
  sentence-transformers 6.1.0, transformers 5.19.0 and torch 2.14.0 in a pinned venv, loading the checkpoint in
  float32 on the CPU (revision `914f7f89`, sha256 `197a3296…`, from `~/models/embeddinggemma-2`), writes 48
  reference texts drawn from this repo's own docs at the pinned commit: 16 headings under `query`, 16 section bodies
  under `document`, 8 under no prompt, 2 under `STS`, 2 under `Classification`, and 4 documents of 1,200+ tokens, past
  the 1,025-wide window. (2) `go test -tags realckpt -run TestReal_parity ./embeddinggemma2/`. (3) The same script's
  truncation reading: up to 400 sections, heading (`query`) against body (`document`), recall@1 and recall@10 of a
  heading retrieving its own section at 768 and truncated to 512, 256 and 128.
- **Regime:** CPU, float32, not resident, on both sides; the prompt is stated per text.
- **Parity rule:** every text's ids equal the reference's, and every embedding has cosine >= 0.9999 with it. Any
  miss: the gate fails, nothing ships, and the miss is diagnosed by day (a tokenizer miss and a model miss are
  reported separately).
- **Truncation rule, per width below 768:** offered if its recall@10 is at least 0.95× the 768 figure and its
  recall@1 at least 0.90×; not offered if recall@10 is under 0.90× the 768 figure; in between, **ambiguous: parked
  for the owner**, and not offered until decided. A width that is not offered comes out of `MatryoshkaWidths`, and the
  docs say why.
- **Cost basis:** a 270M encoder on the CPU; the by-day smoke embedded 18 tokens in well under a second and loaded in
  2 s. 48 parity texts on each side plus about 800 encodes for the truncation reading: estimated 15 minutes, queued
  at 30.

## Gate 0 — desk work, no code, no GPU

Two unknowns decide the shape of everything after this. Resolve both from the real HF repo
(`config.json`, the modeling file, the ST/sentence-transformers config if present) and record
what you read, with the commit or revision.

1. **Is the text backbone's attention bidirectional or causal?** EmbeddingGemma's first
   generation used bidirectional attention over a Gemma base, and mean pooling implies it here,
   but confirm rather than infer. Bidirectional means the mask generalization above; causal means
   the existing decoder-embedder path takes it almost as-is.
2. **Does this land in aikit or in goinfer?** goinfer currently has two embedder paths: aikit's
   encoder (bidirectional, BERT-shaped, `-embed-model` pointing at an HF *directory*, with its own
   precision names and `LoadQ8`) and goinfer's decoder-embedder (causal `.gguf`, `-embed-model`
   pointing at a *file*). EmbeddingGemma 2 is a third shape: a Gemma-architecture encoder with
   mean pooling and a projection head. Decide which path it extends, on the basis of answer 1 and
   of where mean pooling already exists. State the reasoning in this doc before writing code —
   putting it in the wrong one is the expensive mistake here, not the kernels.

**If Gate 0 finds the backbone is a plain causal Gemma with last-token pooling after all**, this
collapses to a loader plus a projection and should be re-scoped down to a much smaller item rather
than run through the gates below.

## Gate 1 — tiny synthetic parity, text only

Standard shape for this repo: a pinning script (`scripts/pin_embeddinggemma2_tiny.py`) against the
real modeling code with `trust_remote_code`, a tiny random-weight fixture, and
`decoder/embeddinggemma2_test.go` asserting cosine against the reference. Text only; vision and
audio encoders disabled via the model's own `config_kwargs`, which is the documented way to reach
the 270M footprint.

Pass bar as elsewhere in the tree: cosine ≥ 0.9999 on the pooled vector and a maxAbs in the 1e-7
band. Pool and project before comparing — a match on the final hidden state but not the pooled
output is a pooling bug, and reporting the hidden-state cosine would hide it.

Add the fixture to `censusList` (`decoder/serialize_census_test.go`) rather than excluding it. The
projection head and the pooling state are per-model state nothing else censused covers, and that
check is what caught the Spark-X2.5 batched-prefill crash that all three of its own gates missed.

## Gate 2 — real oracle

`google/embeddinggemma-2` at 270M text-only, against the HF reference, on `mac`. Report the
regime: backend, precision, whether resident, and the task prefix used for the comparison — the
same text embeds differently under a retrieval-query prefix than a document prefix, so an
unlabeled number is not reproducible.

Also record what Matryoshka truncation costs, since the whole point of the smaller dimensions is
that someone will ship them: cosine of the 128-dimension truncation against the full 768 on a real
corpus slice, not on random text.

**Run budget.** Gate 1 is tiny-fixture work and fits the daytime rule. Gate 2 is a real-checkpoint
parity run, which `CLAUDE.md` § "Run budget" puts on the night queue by name — estimate it, then
`night.py add` it on the Mac rather than launching it by day. Nothing in this task needs a timed
throughput number, so no `bench_peer` cell and no timing lock is involved.

## Gate 3 — server wiring, and one API decision to make deliberately

Mechanical parts: `-embed-model` accepts the checkpoint; `/v1/embeddings` serves it;
`-embed-quant` reaches it (the M-17 class of bug in `decoder_embedder.go` was precisely this flag
being silently dropped, so assert it rather than assuming).

**Matryoshka dimensions.** OpenAI's embeddings API has a `dimensions` parameter and this model is
the first checkpoint here that can honor it meaningfully. Support 768/512/256/128, reject other
values with an error naming the supported set, and renormalize after truncating.

**Task prefixes — the real decision.** Text quality depends on them and the OpenAI shape has
nowhere to put them. Three options, and the doc should record which was chosen and why:

1. Apply the symmetric prefix always. Safe, loses retrieval quality, and silently — the worst
   property.
2. Add a non-standard field (`task` or similar) and document it. Honors the model, breaks drop-in
   compatibility for clients that don't know about it, though a default keeps them working.
3. Infer from request shape — a single input as a query, a batch as documents. Guesses, and guesses
   wrong on a batch of queries.

Option 2 with the symmetric prefix as the default looks right, but make it a decision with a
written reason rather than a default that happened. Whichever is chosen, the prefix actually
applied must appear in the response or the server logs, because a retrieval index built under one
prefix and queried under another degrades quietly and is very hard to diagnose afterwards.

## Phase 2 — the other four modalities, later and only on P7's back

Do not start this with Gate 1–3. The vision (170M) and audio (300M) encoders are separate towers,
and `docs/multimodal.md` P7 is already building exactly that kind of tower for Gemma 4 — a mel
front end, an encoder, and a `masked_scatter`-style splice whose seams the doc says already exist
and are inert-gated. Phase 2 should reuse whatever P7 lands, not race it.

The one capability here with no equivalent anywhere in the tree is a **shared embedding space
across five modalities** — text, code, image, video, audio into one 768-dimension space, which
makes cross-modal retrieval possible (search images with text, find audio by description). That is
the genuinely novel part of this model and the reason to keep Phase 2 on the roadmap rather than
dropping it. It is also worth nothing until the text half is measured.

## Decision rules, pre-registered

- Gate 0 says causal with last-token pooling → re-scope down, don't run Gates 1–3 as written.
- Gate 0 can't settle aikit-vs-goinfer from the config alone → read aikit's encoder path before
  choosing; do not start in goinfer by default because this doc lives here.
- Gate 1 cosine misses the bar on the pooled vector but matches on the final hidden state →
  pooling or projection bug, fix before proceeding; do not report the hidden-state number as the
  gate result.
- Gate 2's 128-dimension truncation loses more than it's worth on a real corpus → ship 768/512
  only and say so, rather than exposing a dimension that degrades retrieval.
- Phase 2 before P7's tower work lands → don't.

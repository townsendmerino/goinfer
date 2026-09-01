# Glossary

Every term the other eleven chapters use, in one plain line each, with a link to the chapter that
actually explains it. This page is for looking something up mid-chapter; it is not a summary, and
reading it end to end will not teach you inference.

**No measured numbers live here on purpose.** Ratios and throughputs belong in the chapter that
produced them, next to the machine and the date — a figure copied into a definition loses its
label and goes stale silently. Everything below should still be true several re-anchors from now.

---

## The shape of a model

**Token** — the unit a model actually reads: not a character and not a word, but a chunk of bytes
the tokenizer agreed on in advance. → [Ch. 1](./01-text-becomes-numbers.md)

**Vocabulary** — the fixed set of tokens a model knows. It is a hot-loop constant, not trivia: the
final matrix and every sampling pass are sized by it. → [Ch. 1](./01-text-becomes-numbers.md)

**Tokenizer** — the encoder from text to token IDs. Two schemes appear here (BPE and SentencePiece);
they disagree about whitespace in ways that matter. → [Ch. 1](./01-text-becomes-numbers.md)

**Embedding** — the lookup that turns a token ID into a vector. A row fetch from a big table, not a
computation. → [Ch. 2](./02-one-forward-pass.md)

**Hidden size** — the width of the vector flowing through the model. Almost every matrix in the
model is shaped by it. → [Ch. 2](./02-one-forward-pass.md)

**Forward pass** — one trip through the whole stack: embedding, every layer, then scores. The unit
of work everything else in this book is measured in. → [Ch. 2](./02-one-forward-pass.md)

**Layer** — one repeated block: attention, then a small feed-forward network, each added back onto
the running vector rather than replacing it. → [Ch. 2](./02-one-forward-pass.md)

**Residual connection** — the "added back onto" part: every block contributes a correction to a
running total instead of overwriting it. → [Ch. 2](./02-one-forward-pass.md)

**Attention** — the step where a position looks at earlier positions and mixes in what it finds.
The only place information moves *between* positions. → [Ch. 2](./02-one-forward-pass.md)

**Query, key, value** — attention's three projections of the same vector: what a position is
looking for, what each earlier position advertises, and what it hands over when matched.
→ [Ch. 2](./02-one-forward-pass.md)

**MLP / feed-forward** — the per-position half of a layer. No mixing between positions; most of the
parameters. → [Ch. 2](./02-one-forward-pass.md)

**LM head** — the final matrix, hidden vector to one score per vocabulary entry. Sized by the
vocabulary, which is why Chapter 1's number reaches this far. → [Ch. 2](./02-one-forward-pass.md)

**Logits** — those raw scores, before anything turns them into a choice.
→ [Ch. 2](./02-one-forward-pass.md)

## Choosing a token

**Softmax** — the conversion from arbitrary scores to something that sums to one, so it can be
sampled from. → [Ch. 3](./03-picking-a-token.md)

**Greedy decoding** — always take the highest-scoring token. Deterministic, and the baseline every
parity gate in this repo compares against. → [Ch. 3](./03-picking-a-token.md)

**Argmax** — the index of the largest score. Worth its own entry because it is a *discrete* read of
a continuous vector: an arbitrarily small numeric change can flip it, so "the argmax matched" is
much weaker evidence than it sounds. → [Ch. 3](./03-picking-a-token.md)

**Temperature** — a divisor on the logits before softmax. Lower is more deterministic; the
zero case is greedy decoding. → [Ch. 3](./03-picking-a-token.md)

**Top-k / top-p** — two ways to throw away the tail before sampling: keep a fixed count, or keep
enough of the probability mass. → [Ch. 3](./03-picking-a-token.md)

## Running it in a loop

**Autoregression** — generating one token, appending it, and running again. The loop that makes
everything else here a performance problem. → [Ch. 4](./04-the-loop-and-the-kv-cache.md)

**KV cache** — a memo table. Each position's keys and values are computed once and kept, so the
next step attends over stored work instead of recomputing the prefix.
→ [Ch. 4](./04-the-loop-and-the-kv-cache.md)

**Context window** — how many positions the cache is allowed to hold. Deep context is where decode
gets expensive. → [Ch. 4](./04-the-loop-and-the-kv-cache.md)

**Session** — a cache that outlives one request, so a follow-up turn reuses the prefix it shares
with the last one. → [Ch. 4](./04-the-loop-and-the-kv-cache.md)

**Prefill** — ingesting the prompt: many positions at once, compute-bound, parallel.
→ [Ch. 8](./08-prefill-versus-decode.md)

**Decode** — generating the reply: one position at a time, memory-bandwidth-bound, serial. The same
code as prefill, and a completely different performance problem.
→ [Ch. 8](./08-prefill-versus-decode.md)

**Memory-bandwidth-bound** — limited by how fast weights can be read, not by arithmetic. Decode's
condition, and the reason smaller weights make it faster.
→ [Ch. 8](./08-prefill-versus-decode.md)

**Time to first token** — how long the prompt takes before anything appears. Prefill's user-visible
cost. → [Ch. 8](./08-prefill-versus-decode.md)

## Making it fit

**Quantization** — storing weights in fewer bits than they were trained in.
→ [Ch. 5](./05-making-the-weights-small.md)

**int8 / int4** — the two widths used here. Fewer bits is less to read, which is why a *lossy*
change makes a bandwidth-bound loop faster. → [Ch. 5](./05-making-the-weights-small.md)

**Group scale** — the small floating-point multiplier shared by a run of quantized values, which is
what lets integers stand in for a real range. → [Ch. 5](./05-making-the-weights-small.md)

**GGUF / safetensors** — two on-disk checkpoint formats this repo reads. Different containers for
the same weights. → [Ch. 5](./05-making-the-weights-small.md)

**Residency** — keeping weights on the accelerator between tokens instead of shipping them across
per call. → [Ch. 6](./06-when-the-model-doesnt-fit.md)

**Paging** — running a model bigger than the memory you have by mapping it and letting only the
touched parts be resident. → [Ch. 6](./06-when-the-model-doesnt-fit.md)

**Expert pager** — an LRU over mapped spans. Routed experts are fetched into a fixed set of slots
and evicted as the routing moves. → [Ch. 7](./07-mixture-of-experts.md)

**Slot** — one reusable place an expert can be resident in. How many there are is the whole
paging trade. → [Ch. 6](./06-when-the-model-doesnt-fit.md)

**Major page fault** — the OS fetching a mapped page from disk because it was not resident. The
cost paging is trying to amortize. → [Ch. 6](./06-when-the-model-doesnt-fit.md)

## Sparsity

**Mixture of Experts (MoE)** — a layer holding many feed-forward networks and running only a few
per token. More parameters, similar work per token.
→ [Ch. 7](./07-mixture-of-experts.md)

**Expert** — one of those feed-forward networks. → [Ch. 7](./07-mixture-of-experts.md)

**Router** — the small layer that picks which experts a token goes to. It makes *content* determine
*cost*, which is what breaks a lot of comfortable assumptions.
→ [Ch. 7](./07-mixture-of-experts.md)

**Top-k routing** — keeping only the highest-scoring experts per token. A discrete choice, so a
small numeric shift can change which expert runs. → [Ch. 7](./07-mixture-of-experts.md)

**Dense** — the ordinary case: every token goes through the same weights. The opposite of MoE.
→ [Ch. 7](./07-mixture-of-experts.md)

## Guessing ahead

**Speculative decoding** — optimistic concurrency with rollback. Guess several tokens cheaply,
verify them in one pass, keep the prefix that survives.
→ [Ch. 9](./09-guessing-ahead.md)

**Drafter** — whatever produces the guess: a smaller model, a trained head, or a lookup over text
already seen. → [Ch. 9](./09-guessing-ahead.md)

**Verify pass** — the single forward pass that checks a whole draft at once. What makes the scheme
pay. → [Ch. 9](./09-guessing-ahead.md)

**Acceptance rate** — the fraction of guessed tokens that survive verification. The number the
whole technique lives or dies on. → [Ch. 9](./09-guessing-ahead.md)

**Rollback** — undoing the cache entries a rejected guess wrote. Some architectures cannot do it,
which is what excludes them. → [Ch. 9](./09-guessing-ahead.md)

**MTP** — a small extra head trained to predict more than one token ahead, used as a drafter.
→ [Ch. 9](./09-guessing-ahead.md)

## Where the work happens

**Kernel** — the small hot routine that does the arithmetic: a matrix multiply, a normalization, an
attention step. → [Ch. 10](./10-kernels-and-backends.md)

**Backend** — which machinery those kernels run on: CPU, CUDA, Metal, WebGPU.
→ [Ch. 10](./10-kernels-and-backends.md)

**cgo** — Go's bridge to C. Refusing it is this repo's defining constraint and the source of most
of its interesting problems. → [Ch. 10](./10-kernels-and-backends.md)

**SIMD** — one instruction operating on several values at once. Where the CPU speed comes from.
→ [Ch. 10](./10-kernels-and-backends.md)

**dp4a** — a GPU instruction that multiplies four 8-bit integers and accumulates, in one go. What
makes quantized matrix multiplies fast without tensor cores.
→ [Ch. 8](./08-prefill-versus-decode.md)

**Tensor cores** — dedicated matrix hardware on modern GPUs. Fast, and it accumulates in an order
the hardware chooses, which is why using it collides with bit-identity.
→ [Ch. 8](./08-prefill-versus-decode.md)

**Driver JIT** — handing the GPU driver portable assembly and letting it compile for the actual
card, instead of linking a vendor toolchain. → [Ch. 10](./10-kernels-and-backends.md)

## Being sure it works

**Parity** — agreement with a reference implementation, usually HuggingFace, on the same input.
→ [Ch. 11](./11-knowing-youre-right.md)

**Bit-identical** — not merely close but the same bits. Achievable only if the arithmetic happens in
the same order, which is why it constrains parallelism and hardware choice.
→ [Ch. 11](./11-knowing-youre-right.md)

**Bit-identity contract** — the promise that two paths agree exactly: verify with greedy, decode
with prefill. Cheap to state and easy to break silently.
→ [Ch. 11](./11-knowing-youre-right.md)

**Cosine similarity** — the usual way two logit vectors are compared when they cannot be identical.
Insensitive to scale, and forgiving enough that a high value is weak evidence on its own.
→ [Ch. 11](./11-knowing-youre-right.md)

**Golden** — a recorded output kept as the reference a later run must reproduce.
→ [Ch. 11](./11-knowing-youre-right.md)

**Gate** — a check that fails the build. A gate that cannot fail is not a gate, which is a
recurring theme rather than a slogan. → [Ch. 11](./11-knowing-youre-right.md)

**Regime** — the conditions a measurement was taken under: model, size, backend, depth. A number
without its regime is folklore. → [Ch. 11](./11-knowing-youre-right.md)

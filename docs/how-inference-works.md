# How inference works in goinfer (a from-scratch explanation)

> **Audience:** a smart reader with no machine-learning background. Builds the
> concepts up from zero, then ties each one to where it actually lives in this
> repo. Reads as both a general "how does LLM inference work" primer and a map of
> goinfer's specific architecture.
>
> **This is the ten-minute version.** `docs/book/` covers the same ground in
> eleven chapters at roughly seven times the length, for a reader who knows Go and
> not machine learning, and each chapter there ends in a measured number. This page
> is the one that ties concepts to specific SOURCE LINES; the book is the one that
> explains why inference works the way it does.
>
> **The drift between them is one-directional, and it is not this page.** This page
> carries no measured figures — the only ratios here (4× for int8 KV, 8× for 4-bit
> weights) are definitional arithmetic about bit widths and cannot go stale. So a
> retired benchmark number never needs correcting here; it needs correcting in the
> book and wherever else the repo quotes it. What this page does carry is line-anchored
> citations, and those are maintained by `scripts/queue_citation_lint.py`, which
> re-keys them by content when the code moves.

## The one-sentence version

A language model is a function that takes *some text so far* and produces *a
probability for every possible next word-piece*. "Inference" is just running that
function over and over: predict one piece, glue it onto the text, predict the
next, and so on, until you decide to stop. Everything below is detail about (a)
how that one prediction works and (b) the engineering tricks that make running it
thousands of times not unbearably slow.

That outer loop lives in [`decoder/model.go:935-1131`](../decoder/model.go#L893-L1131), a
function called `generateInto`.

---

## Step 1: Text becomes numbers (tokenization)

Computers don't operate on letters; they operate on numbers. So the very first
thing that happens is the text gets chopped into **tokens** — chunks that are
usually a word or part of a word ("running" might be `run` + `ning`) — and each
chunk is mapped to an integer ID from a fixed vocabulary of, say, 100,000
possible tokens.

Think of it like a phonebook: every possible word-piece has a number. "The cat"
might become `[464, 3797]`.

This repo supports two flavors of this chopping, because different model families
were trained with different schemes — SentencePiece (used by Google's Gemma) and
byte-level BPE (used by GPT-2/Llama). That lives in the [tokenizer/](../tokenizer/)
package ([sentencepiece.go](../tokenizer/sentencepiece.go),
[bytelevel.go](../tokenizer/bytelevel.go)). The important mental model: **a model
has a matched tokenizer, and the model only ever sees integers.** Text in →
integers in → the model does its thing → an integer out → that integer gets
turned back into text.

---

## Step 2: One prediction — "the forward pass"

This is the heart of it. Given the list of token IDs so far, how do we predict the
next one? The computation that does this is called the **forward pass** (data
flows *forward* through the network). In this repo, for a single new token, it's
[`Model.forward`](../decoder/model.go#L468-L474) — a small wrapper that runs the
layer stack ([`runLayers`](../decoder/model.go#L330)) and then projects to scores
([`logitsFromHidden`](../decoder/model.go#L492)). Here are its stages.

### 2a. Each token becomes a vector (the embedding)

An integer like `3797` is meaningless on its own. So the model keeps a giant
lookup table — one row per vocabulary word — where each row is a long list of
numbers (a **vector**, maybe 4,096 numbers long). That vector is the model's
learned "meaning" of that token. Words used in similar ways end up with similar
vectors.

The lookup is literally one line: [`m.w.Embed.Row(id, h)`](../decoder/model.go#L350)
— "go fetch row `id` from the embedding table and put it in `h`." From here on,
the token *is* that vector, called the **hidden state** (`h` in the code). The
entire job of the network is to repeatedly transform this vector so that, by the
end, it encodes "what comes next."

### 2b. The vector passes through a stack of identical layers

This is the "deep" in "deep learning." The model has dozens of **layers** stacked
on top of each other (could be 32, 80, more). The vector enters layer 1, gets
transformed, the result enters layer 2, gets transformed, and so on. Each layer
does the same two operations, and this loop is
[`runLayersFromEmbed`](../decoder/model.go#L386-L430):

1. **Attention** — "look at the other words and pull in relevant context."
2. **MLP** (also called the feed-forward network) — "think about what you just
   gathered."

Each of those is wrapped so its output is *added back* onto the input rather than
replacing it (a "residual connection" — line 397-428). The intuition: each layer
makes a small *edit* to the running vector rather than rewriting it from scratch.
This is why you can stack so many — each contributes a refinement.

### 2c. Attention — the part that makes it a *language* model

Here's the key problem attention solves. To predict the word after "The river
overflowed its ____", the model needs "bank" — but only because of "river"
earlier. The meaning of each word depends on the *other* words. Attention is the
mechanism for "each word looks around at the others and decides which ones matter
to it."

Mechanically, for each token, the model produces three vectors from its hidden
state:
- a **Query** ("what am I looking for?"),
- a **Key** ("what do I offer to others looking?"),
- a **Value** ("what content do I pass along if attended to?").

Each token's Query is compared against *every previous token's* Key. Strong
matches get high weight; the token then pulls in a weighted blend of those
tokens' Values. That blend gets added back into its hidden state. That's it —
that's how "river" reaches forward and disambiguates "bank."

In this repo that's [`causalAttention`](../decoder/attention.go#L49-L187).
"Causal" means each token may only look *backward*, never at future tokens (you
can't peek at the answer you're trying to predict). The actual
matching-and-blending kernel `causalAttention` calls is
[`attendBatchedHeads`](../decoder/forwardn.go#L292).

Two refinements you'll see in the code, worth knowing because they're everywhere
in modern models:
- **Multiple "heads"** — instead of one Query/Key/Value comparison, there are
  many running in parallel (one head might track grammar, another long-range
  topic). [decoder/attention.go:59](../decoder/attention.go#L59).
- **Position information (RoPE)** — raw attention has no sense of word *order*
  ("dog bites man" = "man bites dog"). So the model rotates the Query/Key vectors
  by an amount that depends on each token's position, encoding *where* each word
  is. [decoder/attention.go:117-139](../decoder/attention.go#L124-L129).

### 2d. The MLP — the "thinking" step

After attention gathers context, the **MLP** ([mlp.go](../decoder/mlp.go))
processes that enriched vector through a couple of big matrix multiplications with
a nonlinear squashing function in between. Loosely: attention is *gathering
information from neighbors*, the MLP is *computing on it*. This is where a lot of
the model's stored "knowledge" lives. The standard form here is
[`gatedMLP`](../decoder/mlp.go#L262).

(There's an important variant, **Mixture of Experts**, covered in the engineering
section — it's central to this repo's recent work.)

### 2e. From vector to a prediction (the LM head)

After the vector exits the last layer, it has been refined into a representation
of "what should come next." The final step,
[`logitsFromHidden`](../decoder/model.go#L492-L514), multiplies it against the
vocabulary table to produce one score for *every* token in the vocabulary —
100,000 numbers, where a high score means "this token is a likely next one."
These raw scores are called **logits**.

So the full forward pass is: **integer → embedding vector → (attention + MLP) ×
many layers → one score per possible next token.**

---

## Step 3: Picking the next token (sampling)

Now we have 100,000 scores. How do we choose one? That's
[sampler.go](../decoder/sampler.go), specifically
[`SampleWithInfo`](../decoder/sampler.go#L114-L133).

- The simplest choice: just take the highest-scoring token. That's **greedy /
  argmax** ([decoder/sampler.go:186](../decoder/sampler.go#L129)) — deterministic, the
  model's single best guess.
- More commonly we add controlled randomness so output isn't robotic.
  **Temperature** flattens or sharpens the scores (high temperature = more
  adventurous, low = more predictable). Then we usually restrict the random draw
  to the top few candidates — **top-k** (only the k best), **top-p / nucleus**
  (the smallest set covering p% of the probability) — to avoid picking something
  absurd ([decoder/sampler.go:188-190](../decoder/sampler.go#L131-L133)).

The scores are turned into actual probabilities via **softmax** (exponentiate and
normalize so they sum to 1), and one token is drawn. There are also **penalties**
to discourage the model from repeating itself
([decoder/sampler.go:179-181](../decoder/sampler.go#L122-L124)).

The output is a single integer — the next token.

---

## Step 4: The loop (autoregression)

Now zoom back out to [`generateInto`](../decoder/model.go#L893-L1131). We:

1. Run the forward pass on the prompt,
2. Sample one new token,
3. **Append it to the sequence,**
4. Run the forward pass again — now with that new token as input,
5. Sample the next one,
6. Repeat until we hit a stop token or a length limit
   ([decoder/model.go:1053-1130](../decoder/model.go#L1004-L1130)).

This is called **autoregression** — the model's own outputs become its next
inputs. The text you see "streaming" out of a chatbot is exactly this loop, one
token at a time.

That's the complete conceptual picture. Everything from here down is about making
it *fast and small enough to actually run*, which is where goinfer earns its keep.

---

## The engineering reality (this is where goinfer lives)

A naive version of the above would be unusably slow and memory-hungry. The hard,
interesting work of an "inference engine" is the optimizations.

### The KV cache — the single most important optimization

Notice the wasteful thing in the loop: to generate token #500, attention needs
the Keys and Values of all 499 earlier tokens. But we *already computed* those
when we processed them. Recomputing them every step would make generation
quadratically slow.

So we don't. We compute each token's Key and Value once and **stash them in a
cache**, then reuse them forever. That's the
[KVCache](../decoder/kvcache.go#L50-L105), and it's why generation stays roughly
linear instead of exploding. The cache is appended to on every step
([decoder/attention.go:157](../decoder/attention.go#L164)).

The catch: this cache *grows with context length* and becomes the dominant memory
consumer for long conversations. So a big chunk of this repo is clever ways to
shrink it:
- **int8 KV quantization** — store the cached Keys/Values as 8-bit integers
  instead of 32-bit floats, ~4× smaller, with a per-head scale factor to
  reconstruct them ([decoder/kvcache.go:20-25](../decoder/kvcache.go#L20-L25)).
- **Ring buffers for sliding-window layers** — some layers only ever need the
  last *W* tokens, so the cache for them is a fixed-size circular buffer that
  overwrites old entries instead of growing forever
  ([decoder/kvcache.go:132-141](../decoder/kvcache.go#L126-L141)).

### Quantization — making the *weights* small too

A model's "weights" (all those learned numbers in the tables and matrices) are
enormous — tens to hundreds of gigabytes at full precision. **Quantization**
stores them at lower precision: instead of 32 bits per number, use 8 bits or even
4 bits. A 4-bit model is ~8× smaller, at a small accuracy cost. This is what lets
a big model fit on a normal machine.

In this repo the abstraction is `linalg.WeightMat` (from the sibling `aikit`
library), which holds weights as full floats, int8, or int4 behind a uniform
interface; goinfer's policy for *which* precision a given model uses lives in
[weightmat.go](../decoder/weightmat.go#L16-L36). When the math needs them, they're
reconstructed ("dequantized") on the fly. int4 unpacking is currently the live performance lever because single-stream
generation is bottlenecked on memory bandwidth — the machine spends its time
*reading weights from memory*, so making the weights smaller directly makes it
faster.

### Mixture of Experts (MoE) — and why it complicates everything

Some modern models replace the single MLP with *many* parallel MLPs called
**experts**, plus a little **router** that, for each token, picks just the top few
experts to actually run ([`moeMLP`](../decoder/mlp.go#L67),
[`routeExperts`](../decoder/mlp.go#L133)). The payoff: the model can have
huge total knowledge (many experts) while only doing a little work per token (a
few experts). The cost: all those experts have to *exist in memory* even though
most go unused each step.

goinfer's answer is **expert demand-paging**
([moepaging.go](../decoder/moepaging.go)): keep experts on disk (memory-mapped)
and page them into RAM only as the router calls for them, under a memory budget.
This is the path GLM, DeepSeek, and the other MoE families travel.

### Pure Go, no C — the distinctive engineering bet

Almost every other inference engine (llama.cpp, vLLM, PyTorch) leans on
C/C++/CUDA for the heavy math. **goinfer's default build is written entirely in
Go with no CGO** (the optional WebGPU backend is the lone exception) — the
matrix-multiply kernels are hand-written Go using CPU SIMD instructions
(AVX2/FMA on x86-64, NEON on ARM) selected at runtime, via the sibling
`aikit/linalg` library. The payoff is a single self-contained static binary with
no dependency hell — "runs every modern attention variant in one pure-Go binary"
is precisely the positioning claim in the families roadmap.

### The architecture registry — supporting many model families

Different model families (Gemma, Llama, Qwen, GLM, Granite…) differ in small but
real ways — how they normalize, how they do positions, whether they're MoE, etc.
Rather than one tangled code path, this repo uses a **registry**
([decoder/registry.go:19-52](../decoder/registry.go#L19-L52)): each family registers an
"adapter" describing its quirks, and the engine resolves the right one at load
time. Most families share a generic forward pass; the genuinely different ones
(the Gated DeltaNet hybrid in
[forward_qwen35.go](../decoder/forward_qwen35.go), the Mamba-2 hybrid in
[forward_granite.go](../decoder/forward_granite.go)) get their own. This registry
is *the* extension point — adding a new model family is mostly "write a new
adapter."

### Two more worth a mention

- **Session prefix reuse** ([decoder/session.go:71-99](../decoder/session.go#L71-L99)) —
  in a chat, each new turn shares a long prefix with the last one (the whole
  conversation history). Instead of reprocessing it, the engine keeps the KV
  cache from before and only processes the *new* part. Huge win for chat and
  agent loops.
- **GPU full-residency path** ([residency.go](../decoder/residency.go)) — for
  supported model shapes, the engine can keep the whole model and cache resident
  on a GPU (via WebGPU) and run the per-token forward entirely on-device, falling
  back to CPU otherwise.

---

## The whole thing in one breath

Your text is chopped into integer **tokens**; each token becomes a **vector**;
that vector flows through a stack of layers that alternately **attend** (gather
context from earlier tokens, reusing a **KV cache** so it's not recomputed) and
**think** (the MLP, or a routed set of MoE experts); the final vector is scored
against the whole vocabulary to give **logits**; a **sampler** picks the next
token; and the whole thing **loops**, feeding each output back in, until it stops.
goinfer's particular contribution is doing all of this in **pure Go, with
aggressively compressed weights and KV cache**, across a growing **registry** of
model families.

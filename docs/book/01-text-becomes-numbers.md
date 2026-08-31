# Chapter 1 — Text becomes numbers

*The model never sees your text. This chapter is about what the model sees instead, and why
that choice reaches all the way through to a performance number in Chapter 3.*

---

## The one thing to hold onto

A language model is a function from integers to integers. Text goes in at one end and text
comes out the other, but the model itself only ever handles integers. Everything in this
chapter is about the layer that performs the conversion.

The conversion is not clever, and the conversion is not learned at run time. There is a fixed
table — called the **vocabulary** — mapping pieces of text to integer IDs. Tokenizing is
looking pieces of text up in the vocabulary. Detokenizing is looking integer IDs up in the
vocabulary in reverse.

```
  "The cat"                                                   " sat"
      |                                                          ^
      | tokenize                                       detokenize |
      v                                                           |
  [464, 3797]  →  [ the model ]  →  scores over the whole vocabulary
   token IDs       integers in,      one score per entry: 32,000 to
                   integers out      152,000 floats, every token
```

The pieces of text in the vocabulary are called **tokens**. A token is usually a word or part
of a word — "running" might be `run` + `ning` — and a typical vocabulary holds somewhere
between 32,000 and 152,000 tokens. Common words get their own vocabulary entry. Rare words
get assembled from several fragments.

If you want a Go analogy: the vocabulary is an interning table. Strings in, stable integer
handles out, and every downstream consumer works on the handles. The interning-table analogy
holds up well, with one important difference this chapter returns to at the end.

---

## Why not just use bytes?

You could. A byte is already an integer, a byte vocabulary would hold 256 entries, and no
lookup table would be needed at all.

The reason nobody uses raw bytes as the vocabulary is sequence length. Every token the model
processes costs compute, and — as Chapter 8 shows — the cost of attention grows faster than
linearly with the number of tokens. A 2,000-word document is roughly 2,600 tokens under a
normal vocabulary and roughly 12,000 bytes:

```
  same document, two vocabularies

  sub-word vocabulary:   2,600 tokens
  raw-byte vocabulary:  12,000 tokens     4.6x longer

  and because the cost of attention grows FASTER than linearly in
  sequence length, the penalty on the attention term is worse than 4.6x
```

So the vocabulary is a compression scheme. The vocabulary trades a bigger lookup table for
shorter sequences, and the trade is heavily worth it.

The opposite extreme — one vocabulary entry per whole word — fails for a different reason. You
cannot enumerate every word, and any word outside the vocabulary becomes unrepresentable.
Sub-word pieces are the compromise: common words are one token, and anything unusual is still
expressible by falling back to smaller pieces, down to individual bytes if necessary.

---

## The two schemes in this repo

Different model families were trained with different tokenizers, and a model is only correct
with the tokenizer the model was trained on. The match between model and tokenizer is a hard
constraint, not a preference. Feed a model another model's token IDs and you get fluent
nonsense, because ID 3797 means something different in each vocabulary.

goinfer implements two tokenizer schemes, in the `tokenizer/` package:

**Byte-level BPE** ([`bytelevel.go`](https://github.com/townsendmerino/goinfer/blob/main/tokenizer/bytelevel.go)) — used by the GPT-2 and Llama lineages. Byte-level BPE maps
every input byte to a starting symbol, then applies a learned list of merge rules repeatedly:
"these two adjacent symbols become this one symbol." The merge rules were learned during
training by repeatedly merging whichever pair of symbols was most frequent in the training
corpus. At run time the merge rules are a fixed, ordered list that byte-level BPE applies in
order:

```
  input                     l o w e s t
  merge rule 1:  e + s → es  l o w es t
  merge rule 2: es + t → est l o w est
  merge rule 3:  l + o → lo    lo w est
  merge rule 4: lo + w → low   low est
  result                       ["low", "est"]     2 tokens, not 6
```

The "byte-level" part matters: because byte-level BPE starts from raw bytes, nothing is
unrepresentable. Emoji, malformed UTF-8, a binary blob pasted into a prompt — all of it
tokenizes, possibly into many tokens, but never into an error.

**SentencePiece** ([`sentencepiece.go`](https://github.com/townsendmerino/goinfer/blob/main/tokenizer/sentencepiece.go)) — used by Google's Gemma family among others. Same
outcome, different construction: SentencePiece scores candidate segmentations of the input
against a learned unigram model and picks the best-scoring segmentation, rather than applying
merge rules in a fixed order.

The `tokenizer/` package also holds [`added.go`](https://github.com/townsendmerino/goinfer/blob/main/tokenizer/added.go), for tokens bolted on after training — the
special markers like `<|im_end|>` that chat templates use to delimit turns — and `gguf.go`,
which reads a vocabulary out of a GGUF checkpoint, since GGUF files carry their tokenizer
inside them.

You do not usually pick the tokenizer scheme. The loader reads which scheme the checkpoint
declares and constructs the matching tokenizer. Selecting the wrong tokenizer is not a subtle
bug; the output becomes obviously garbled.

---

## Where the Go analogy breaks

An interning table is a pure convenience — you could always use the strings directly. The
tokenizer is not a convenience, and the difference between an interning table and a tokenizer
shows up in three places.

**Token boundaries are not character boundaries.** A single Unicode character can be split
across several tokens under byte-level BPE, because the merge rules operate on bytes. Splitting
a character across several tokens means one token's text may be a fragment of a rune rather
than valid UTF-8 on its own.

That is not a hypothetical. goinfer's streaming server holds bytes back until a complete rune
is available, so a token that completes no new character emits nothing, and the next token
emits several tokens' worth of text at once. The number of streamed chunks is therefore always
less than or equal to the number of tokens generated, and anyone counting streamed chunks to
compute tokens per second undercounts — measured at up to **1.046 tokens per chunk, a 4.4%
under-read**, across 92 cells. Stop-sequence matching adds a second reason to hold bytes back, so
the undercount is not confined to multi-byte characters, and it only ever runs one way: a client
counting chunks sees the engine as slower than it is, silently and with no error to notice. The
fix is to read the token count the server reports rather than counting streamed chunks. Chapter 11 has more on why
plausible-looking measurements are the dangerous kind.

**The vocabulary size reaches into the hot loop.** After the model computes its answer, the
model produces one score per vocabulary entry — 152,000 floats for a large-vocabulary model,
32,000 for a small one — and the sampler has to work over every one of those scores to pick a
token. That work is not free, and Chapter 3 shows the sampler's cost landing between 5% and 18%
of a token's total cost depending on the model.

Two measurements from this tree separate the two things that cost depends on. Two models with
the same 151,936-entry vocabulary but very different sizes had nearly identical sampler costs:

```
  model   vocabulary   sampler cost   cost of one token   sampler's share
  1.5B      151,936       1.009 ms          4.535 ms           18.2%
  7B        151,936       1.102 ms         13.685 ms            7.5%

  sampler cost      depends on VOCABULARY SIZE   1.009 → 1.102 ms   (+9%)
  cost of one token depends on MODEL SIZE        4.535 → 13.685 ms  (+202%)
  sampler's share   = sampler cost / cost of one token
```

A 4.6x larger model changed the sampler's cost by 9% and the sampler's share of a token by a
factor of 2.4, in the opposite direction. So: **vocabulary size sets the sampler's absolute
cost; model size sets what fraction of a token that absolute cost represents.** That
distinction turns out to decide a shipped default, which is Chapter 3's story.

**The tokenizer and the model are a matched pair, not a utility.** You cannot swap tokenizers,
upgrade a tokenizer independently, or share one tokenizer across models. The tokenizer ships
with the checkpoint and the tokenizer is part of the model.

---

## What it costs

Tokenization itself is cheap. Tokenization is a table lookup and a merge loop over the input,
tokenization happens once per prompt rather than once per generated token, and tokenization
does not appear in any profile in this repo as a term worth optimizing. Chapter 8 shows where
prompt-processing time actually goes, and prompt-processing time does not go here.

The interesting cost is the cost tokenization *creates*: a vocabulary of *V* entries means every
single forward pass ends by producing *V* scores. Those *V* scores are a real cost, they are paid
on every generated token forever, and they are the direct consequence of the compression choice
made in this chapter.

Chapter 2 follows a single forward pass from those integer IDs to those *V* scores.

---

*Sources: `tokenizer/` (`bytelevel.go`, `sentencepiece.go`, `added.go`, `gguf.go`),
[`docs/how-inference-works.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/how-inference-works.md) §Step 1, [`internal/serveapp/openai.go`](https://github.com/townsendmerino/goinfer/blob/main/internal/serveapp/openai.go) and
[`internal/serveapp/openai_stream_usage_test.go`](https://github.com/townsendmerino/goinfer/blob/main/internal/serveapp/openai_stream_usage_test.go) (streaming hold-back), [`CHANGELOG.md`](https://github.com/townsendmerino/goinfer/blob/main/CHANGELOG.md)
(the 1.046 tokens-per-chunk measurement across 92 cells),
[`docs/QUEUE.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/QUEUE.md) and [`docs/spec/10-optfwd-gate.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/spec/10-optfwd-gate.md) (sampler-share measurements).*

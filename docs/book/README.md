# An inference primer for Go engineers

Eleven chapters on how a language model actually runs, written for someone who knows Go and
does not know ML.

The assumption throughout is that you understand goroutines, memory layout, cache behaviour
and profiling, and that you have never met a transformer. So the explanation budget goes to
the ML, and the systems half leans on intuitions you already have — a KV cache is a memo
table, an expert pager is an LRU over mapped spans, speculative decoding is optimistic
concurrency with rollback.

Each chapter ends with what the thing costs here, measured, with a pointer to the document
that produced the number.

| # | chapter | what it covers |
|---|---|---|
| 1 | [Text becomes numbers](./01-text-becomes-numbers.md) | Tokenization, and why vocabulary size reaches into the hot loop |
| 2 | [One forward pass](./02-one-forward-pass.md) | Embedding, the layer stack, attention, the MLP, and the LM head |
| 3 | [Picking a token](./03-picking-a-token.md) | Sampling and temperature; a shipped default that was wrong, and the input that hid it |
| 4 | [The loop and the KV cache](./04-the-loop-and-the-kv-cache.md) | Autoregression, the cache that makes it affordable, and the models that can't have one |
| 5 | [Making the weights small](./05-making-the-weights-small.md) | Quantization, why fewer bits means faster, and what parity means once the arithmetic moved |
| 6 | [When the model doesn't fit](./06-when-the-model-doesnt-fit.md) | Paging and residency; a slot sweep that broke two of its own guard rails |
| 7 | [Mixture of Experts](./07-mixture-of-experts.md) | Sparsity, routing, and why content becomes cost |
| 8 | [Prefill versus decode](./08-prefill-versus-decode.md) | Two different performance problems from one code path |
| 9 | [Guessing ahead](./09-guessing-ahead.md) | Speculative decoding, acceptance rate, and rollback |
| 10 | [Kernels and backends](./10-kernels-and-backends.md) | CPU, CUDA, Metal, and what refusing cgo costs |
| 11 | [Knowing you're right](./11-knowing-youre-right.md) | Parity gating, bit-exactness, and how a measurement lies |

## The shorter version, and the code map

`docs/how-inference-works.md` covers this same material in about 2,300 words instead of 15,000.
It is the better starting point if you want the whole arc in ten minutes, and it is the better
reference if your question is *"where does this live in the source?"* — it anchors concepts to
specific lines in `decoder/`, which these chapters mostly do not.

**The two overlap heavily and can drift.** A number corrected here belongs there too, and the
reverse. Neither page's tooling can see the other's prose.

## Reading order

One through eleven works. If you're here for the engineering rather than the concepts,
chapters 8, 6 and 11 stand alone reasonably well.

Chapter 11 is the one to read if you only read one. It is about how measurements in this
repo have gone wrong — instruments returning plausible numbers for the wrong reason, tests
vouching for behaviour the system cannot produce, retractions that failed to propagate — and
most of it generalizes past inference entirely.

## A note on the numbers

Every figure is traced to a repo document. Where a number is regime-specific — measured on
one backend, one model size, or a model slice rather than a whole model — the text says so,
because the alternative is how a non-transferable number loses its label and becomes folklore.

Some numbers in `docs/benchmarks.md` are currently marked stale pending re-anchor after a
driver and distro upgrade on the Linux box. Nothing in these chapters depends on those rows.

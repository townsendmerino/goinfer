# Chapter 8 — Prefill versus decode

*The same forward pass, run two ways, produces two completely different performance problems.
Confusing them is the most common mistake in reasoning about inference cost.*

---

## The split

**Prefill** processes the prompt. Every token of the prompt exists already, so the whole prompt
can go through the model at once. The matrix multiplies are large, the hardware is well used,
and the limit is arithmetic throughput.

**Decode** generates the reply. One token at a time, because Chapter 4's loop cannot be
unrolled — token N+1 needs token N. The matrices are skinny: a single position against the full
weight set. The hardware is mostly idle, waiting on memory.

Same code path. Opposite bottlenecks.

| | prefill | decode |
|---|---|---|
| positions per pass | many | one |
| limited by | compute | memory bandwidth |
| parallelism | plenty | almost none |
| user experience | wait before first word | speed of words appearing |

Most of this book so far has been about decode, because decode is where quantization
(Chapter 5), paging (Chapter 6) and the KV cache (Chapter 4) do their work. This chapter is
about prefill, and about why prefill matters more than prefill used to.

---

## Why prefill got important

For a chat with a short question, prefill is a rounding error. You type twenty tokens and
generate five hundred.

Agentic workloads invert that ratio. An agent sends a system prompt, tool definitions, file
contents, and a conversation history — thousands of tokens — and gets back a short tool call.
Then the agent does it again, with the history one turn longer. The ratio flips: mostly prompt,
little generation, repeated every turn.

That is why prefill is where this repo's widest gap against peers sits, and it is why
Chapter 4's prefix reuse matters so much for this workload.

---

## What prefill costs here

Measured on CPU, with the number of workers as the variable:

| prompt tokens | 1 worker | 6 workers (shipped default) | speedup |
|---|---|---|---|
| 1,520 | 89.7 s (16.9 tok/s) | **33.8 s (44.9 tok/s)** | 2.65× |
| 3,020 | 333.3 s (9.1 tok/s) | **101.6 s (29.7 tok/s)** | 3.28× |

Two things stand out.

The parallel speedup is good but sublinear — 3.28× from six workers. Prefill has real
parallelism available, and 3.28× is a respectable fraction of six.

More importantly, look at the scaling with prompt length:

```
  prompt tokens    1,520  →  3,020        2.0× more tokens
  time (6 workers)  33.8s → 101.6s        3.0× more time

  if prefill were linear in prompt length, 2× the tokens would cost 2×
  the time. It costs 3×, because each position compares against every
  earlier position — exactly what Chapter 2's attention diagram showed.
```

Against peers, this is the weakest lane. [`docs/benchmarks.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/benchmarks.md) records Ollama at roughly 4–5×
faster per prefill token. Decode is competitive; prefill is not.

---

## Where the time actually goes

Profiling long-context prefill shows the answer is not spread around.

On a 4-layer slice of a Mixture-of-Experts model, attention accounted for **97.1%** of an
8,192-token prefill, and attention's share climbs with context length:

```
  prompt tokens    attention    everything else
       1,024          83.2%          16.8%
       2,048          89.7%          10.3%
       4,096          94.9%           5.1%
       8,192          97.1%           2.9%

  the same quadratic story as the stopwatch above, showing up in a profile
```

Two kernels dominate: the query-key comparison and the attention-value mixing. On an 8k
profile they were 51.1% and 18.7% respectively, in the higher-precision path.

Those figures come with a warning attached, and the warning is a Chapter 11 warning. **97.1%
is a 4-layer-slice number.** The slice was used because the full model did not fit in available
memory, and the slice is representative for that model specifically because that model's layer
pattern repeats every four layers. But a slice is not a model, and quoting a slice figure as a
model figure is exactly the kind of regime error this repo has a convention against.

The consequence showed up immediately. A fast-attention optimization measured **3.11×** on
the slice and **1.52×** on the full model. Both are real numbers; only one answers the
question anyone was asking.

---

## The flag

`--cpu-fast-attention` runs attention in lower precision. On dense models it measured 2.28×
at 8k context, behind a cosine similarity of 0.9976 — a small, stated, opted-into divergence
rather than a bit-identical result.

For a long time `--cpu-fast-attention` refused Mixture-of-Experts architectures. The reasoning
was Chapter 7's: routing is discontinuous, so a small numerical shift can flip an expert
selection, which is a much bigger change than a slightly different number. The reasoning was
plausible, and the reasoning was never measured — behind a test whose comment said the
exclusion was pinned and whose body tested only a dense model.

When the exclusion was finally measured, the mechanism turned out to be real in kind but not in
magnitude, and `--cpu-fast-attention` now covers Mixture-of-Experts models. The full-model win
is **1.52×**.

That left a live product question rather than a technical one: a third off prefill on the most
common agentic shape is a large saving, and the path is not bit-identical, so should the
bit-exactness contract be opt-in or opt-out?

It was settled in favour of speed. `--cpu-fast-attention` is now **on by default**, and
`--cpu-exact-prefill` is the way back to the bit-identical kernel — it wins if both are passed,
because between a speed request and a correctness request the correctness one is the safe way to
read a contradiction the user did not know they had expressed.

Two things made that defensible rather than merely faster.

The first is a floor. Attention cost grows with the square of the prompt, so the saving grows with
length — but the divergence does not. A short prompt would have paid the full change in output for
almost none of the speed: an eight-token prompt was measured diverging at the third generated token
and never re-converging, at a length where the win is nil. So the fast path is floored at 512 prompt
tokens; below that the exact kernel runs regardless of the flag. Every existing golden in the repo
uses a shorter prompt than that, which is why flipping the default left all of them passing
untouched — and also why a new golden had to be written above the floor, since otherwise nothing
tested the path that had just become the default.

The second is what the default costs, stated rather than implied. Prefill and decode no longer agree
bit-for-bit above the floor. Nor do two machines: the same prompt on the same checkout produced a
different first token on Apple Silicon than on x86, because the compiler fuses multiply-add on one
and not the other and there is no wider accumulator to absorb the difference. That is a real
property to know about before diffing outputs across machines, and it is the reason
`--cpu-exact-prefill` exists rather than being a courtesy.

---

## The tensor-core wall

Everything above is CPU. On CUDA the story gets a third character: **tensor cores**, purpose-built
matrix hardware that Ollama's prefill runs on and this repo's does not — and the reason is not that
nobody got to it.

goinfer's int4 weights carry a 16-bit floating-point scale per 32-element group, which forces a
floating-point accumulation every 8 values. Floating-point addition is not associative — `(a + b) +
c` and `a + (b + c)` can round to different bits — so a tensor core, which accumulates in whatever
order its hardware tiling dictates, rounds differently than the scalar path that respects the
scale-group boundaries. Move to one scale per row instead, and there is no mid-accumulation rescale
to do: the whole dot product accumulates in exact int32, and the single float multiply happens once,
at the end — which is why a tensor-core GEMM under per-row scales would be bit-identical **by
construction**. Group-scale granularity, not bit-identity itself, is what tensor cores can't get
past.

So the fork is: keep the format goinfer ships, or pay a parity refresh to move to per-row scales and
open the door to tensor cores. It was scoped, and it was measured before it was decided.

**Phase 0** quantized real weights both ways and compared each to the unquantized original. Naive
per-row scales were 1.73× worse than shipped; the best per-row scale search this repo could run was
still 1.24× worse.

**Phase 0b** ran that 1.24×-worse format through 28 layers of an actual forward pass and checked
what came out the other end:

```
  build                       perplexity     top-1 agreement
  shipped (per-group)             28.5             76.7%
  per-row, best scale found      108.0             68.6%

  a 1.24× error in the weights becomes a roughly 4× error in the output.
  small errors don't stay small once they cross a discrete line — the
  same lesson Chapter 7 drew from routing, applied here to argmax over
  the vocabulary.
```

---

## Why the fork stays closed

There was a third option: keep the format, but relax the *contract* instead — stop requiring
bit-identical output and require only that prefill and decode land on the same tokens, gated on the
first 64. It sounds like a reasonable compromise, and it fails for the same reason Phase 0b just
demonstrated. Token selection is `argmax` over a vocabulary — a discrete decision — and this repo
has now measured, twice, that discrete decisions here flip on margins near zero. A tolerance gate
built on that assumption would pass on most prompts and diverge on whichever ones happened to land
near a tie: a gate that's red 5% of the time at random is harder to trust than one that's always red
or never red, and it is exactly the kind of thing that would surface months later on a model family
nobody happened to be testing against.

So tensor cores are not pursued, **as a decision, not an omission** — recorded 2026-08-04. The
format stays group-scaled int4.

That leaves the 4–5× prefill gap this chapter measured as two problems wearing one number. Part of
it is this fork: dp4a, the integer path GEMV actually runs on, tops out around a third of what
tensor cores could reach on this hardware, full stop, without opening the fork above. But goinfer's
GEMV is only at 54% of the dp4a ceiling it's already allowed to reach today — which costs nothing in
bit-identity, isn't blocked by anything in this section, and is still on the table.

---

## What it costs

The concrete version, for a user: a 3,020-token prompt takes 101.6 seconds to process at the
shipped CPU default. With fast attention at 1.52×, roughly 67 seconds. In an agent loop
resending a growing context every turn without prefix reuse, that difference compounds over
every turn.

Set against decode on the same class of machine — around 39–41 tok/s on a 1.5B model — the
asymmetry is stark:

```
  generating a 500-token reply     500 / 40 tok/s   ≈   12 seconds
  reading the 3,020-token prompt   measured         =  101.6 seconds

  the prompt costs roughly 8× the reply, and the user waits for the
  prompt before seeing a single word
```

That is the shape of the problem, and it's why prefill is where the work is.

Chapter 9 turns to the one technique that attacks decode's fundamental sequential
constraint.

---

*Sources: [`docs/queue-performance.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/queue-performance.md) (G16/G20 prefill baselines), `docs/benchmarks.md`
(peer prefill ratio), [`docs/measurements/mellum2-moe-prefill-split-RESULT.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/mellum2-moe-prefill-split-RESULT.md) (attention share,
the slice-versus-model correction, and the 3.11×/1.52× figures), [`docs/ollama-chase.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/ollama-chase.md)
(§7, the bit-identity fork — Phase 0/0b and the tensor-core decision), and [`CLAUDE.md`](https://github.com/townsendmerino/goinfer/blob/main/CLAUDE.md) (measurement
discipline).*

# Chapter 7 — Mixture of Experts

*Chapter 2's MLP used all its weights on every token. This chapter replaces it with something
that uses a small fraction, which is how a 35B model runs on a laptop — and which breaks
assumptions running through the rest of the engine.*

---

## The idea

In a dense model, every parameter participates in every token. Reading every parameter is
where most of the cost goes, and most of that reading is arguably wasted: the weights that
matter for parsing Go syntax are probably not the weights that matter for French poetry, but
both sets of weights get read either way.

Mixture of Experts replaces the single MLP in each layer with many smaller ones — the
**experts** — plus a small **router** that picks which ones to use for each token.

```
dense:  h → MLP → out
MoE:    h → router → pick top-k of N experts → run only those → out
```

A typical configuration has 64 or 128 experts per layer and routes each token to 8 experts. So
the model holds a very large number of parameters and uses a small fraction of those
parameters per token.

The naming convention tells you both numbers:

```
  35B-A3B   =   35 billion total parameters
                 3 billion ACTIVE per token      →  3/35 ≈ 8.6% of the model
                                                     touched per token

  per layer:  64 experts stored,  8 selected  →  1/8 of the expert weights

  capacity comes from the 35B; cost comes from the 3B
```

You get the capacity of a large model at close to the compute of a small one.

---

## Why this makes paging work

Chapter 6 described streaming weights from disk. That only has a chance of keeping up if you
don't need all of them.

Mixture of Experts is what makes streaming viable. A dense 35B model needs all 35 billion
weights for every token, so streaming a dense 35B model means moving the entire model per
token, and the bandwidth required is hopeless. A Mixture-of-Experts 35B model needs about 3
billion weights per token — and, crucially, *which* 3 billion is decided by the router before
the experts are needed.

So the access pattern is sparse and predictable, which is exactly the shape a prefetching cache
can exploit. The expert pager keeps recently-used experts resident, and the router's choice
tells the expert pager what to fetch next.

That is why every model discussed in Chapter 6 was a Mixture-of-Experts model. Paging does not
rescue dense models.

---

## Where the cost goes

On the CUDA side, running a 35B-A3B model on an 8 GB card, roughly **48%** of each token is
expert data movement. Not arithmetic — moving expert weights across PCIe to the device. Nearly
half of every token is a bus transfer.

That figure is the whole optimization story for this model class. The measured throughput:
10.74 tok/s at the default settings, rising to 15.69 tok/s with CUDA graphs enabled, which cut
launch overhead. Work on overlapping the transfer with computation is bounded at about 1.18×,
because the transfer is already close to the bus's ceiling.

When almost half your token is a bus transfer, the levers are: move less, move it earlier, or
move it once and keep it. Chapter 6's slot tuning is the third. There isn't a fourth.

---

## What MoE breaks

Sparsity buys a lot and costs something in every subsystem that assumed density.

### Content becomes cost

This is the one that catches people, including in this repo.

On a dense model, the cost of a forward pass does not depend on what the text says. Same
number of tokens, same weights, same work. That property is so reliable that benchmark
prompts here use repeated filler — four unique words per entry — because it gives exact token
counts and the content genuinely doesn't matter.

On an MoE, **content is routing**. Identical rows select identical experts. Those experts stay
hot in cache, the pager never has to stage anything, and the measured cost is a best case that
no real prompt will produce.

That happened. A profile run with the standard prompt file showed no expert-paging activity at
all, because every row routed identically. The prompt file was correct — for dense models.
Chapter 11 has the general form of this failure.

The recovery was to keep the degenerate case as a *control arm* rather than discard it: a
batch whose rows all select the same experts is precisely the ceiling that expert-batching
optimizations could aspire to, so it went from confound to bound.

### Routing is discontinuous

A dense MLP is a smooth function of its input. Nudge the input slightly and the output moves
slightly.

A router is not smooth. A router takes a top-k, and near a tie, a tiny numerical difference
flips which expert gets selected — which changes the output by however much two different
experts differ, which is a lot:

```
  router scores, top-2 selected      the same scores after a
                                     1e-5 arithmetic perturbation

    expert A   0.4001   ← selected     expert A   0.4001   ← selected
    expert B   0.4000   ← selected     expert B   0.39999
    expert C   0.3998                  expert C   0.3998   ← selected NOW

  a 1e-5 change in the score produced a DIFFERENT EXPERT, and expert C's
  output is not 1e-5 away from expert B's — it is a different function
```

This has a direct consequence for Chapter 5's accuracy trades and Chapter 11's parity work.
An optimization that perturbs the arithmetic slightly is low-risk on a dense model and
potentially high-risk on an MoE, because the perturbation can flip a routing decision rather
than merely shifting a number.

That reasoning was sound enough that a fast-attention flag in this repo refused MoE
architectures entirely on those grounds — for months, behind a test whose doc comment claimed
to pin the exclusion and whose body asserted nothing about MoE at all. When the exclusion was
finally measured, the mechanism was real in kind but did not hold in magnitude, and the flag
was extended to cover MoE.

Chapter 11 tells that story properly. The short version: a plausible mechanism is not a
measurement, and a comment claiming coverage is not coverage.

### Batching gets harder

Prefill (Chapter 8) processes many tokens at once. On a dense model those tokens all use the
same weights, so it's one large matrix multiply.

On a Mixture-of-Experts model, different tokens in the same batch route to different experts.
You either run many small matrix multiplies or reorder the batch by expert, and reordering the
batch by expert is a real implementation with real complexity. Whether that reordering pays was
measured here, and the answer was bounded at under 5% — because on the workload in question,
attention rather than the expert feed-forward network was consuming the time. Chapter 8
explains why attention was consuming the time.

---

## What it costs

Mixture of Experts is the reason large models run on small machines at all. A 35B-A3B model
runs on an 8 GB CUDA card at 10.74–15.69 tok/s, and on a 16 GB Mac at about 2.2 tok/s. Neither
of those would be possible with a dense model of the same total size.

The costs are concentrated in the plumbing rather than the model: about half of a token is
expert transfer on the constrained GPU, benchmark inputs need rethinking because content now
determines cost, and any optimization that touches numerical precision has to be evaluated
against routing stability rather than just output similarity.

Chapter 8 turns to the split this book has deferred twice: prompt processing and token
generation are two different performance problems.

---

*Sources: [`docs/deltanet-residency-plan.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/deltanet-residency-plan.md) (35B-A3B CUDA figures, expert DMA share),
[`docs/task-moe-streaming.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/task-moe-streaming.md), [`decoder/moepaging.go`](https://github.com/townsendmerino/goinfer/blob/main/decoder/moepaging.go), [`CLAUDE.md`](https://github.com/townsendmerino/goinfer/blob/main/CLAUDE.md) (tests, measurement
discipline), [`docs/task-metal-expert-streaming-at-scale.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/task-metal-expert-streaming-at-scale.md),
[`docs/measurements/mellum2-moe-prefill-split-RESULT.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/measurements/mellum2-moe-prefill-split-RESULT.md) (the under-5% batching bound).*

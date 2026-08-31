# Chapter 6 — When the model doesn't fit

*Chapter 5 halved the weights and the model still doesn't fit. This chapter is about running
it anyway, and about a slot sweep that broke two of its own guard rails.*

---

## The situation

A 27B model quantized to 4 bits is over 15 GB. This repo's development machines are an 8 GB
CUDA card and a 16 GB Mac. Neither holds it.

There are three responses, in increasing order of interest.

**Refuse.** Defensible, and what many runtimes do.

**Offload.** Put some layers on the GPU and run the remaining layers on CPU. Offload works,
and the CPU portion sets the pace, so you get GPU hardware running at CPU speed for part of
every token.

**Page.** Keep the weights on disk or in host memory and stream the pieces you need, when you
need them. This is what makes the difference for the models in this repo, and it's what the
rest of the chapter is about.

---

## Paging

The forward pass touches weights in a predictable order: layer 0, then layer 1, and so on.
That predictability is what makes streaming viable. You know which weights are needed next
before you need those weights, so you can be fetching layer N+1's weights while computing
layer N:

```
  time →

  compute:   [ layer N ][ layer N+1 ][ layer N+2 ]
  fetch:     [ N+1     ][ N+2       ][ N+3       ]
              ^^^^^^^^^^ overlapped with compute, so the fetch is free
                         — as long as the fetch finishes first

  when the fetch does NOT finish first, compute stalls, and the stall is
  what every number in this chapter is measuring
```

The Go instinct is right here — paging is a prefetching problem over a known access sequence,
with a bounded buffer of resident slots and an eviction policy. goinfer's implementation is an
LRU over memory-mapped spans, and prior work in the repo settled that plain LRU beats the
cleverer policies at every realistic budget, so the remaining question is how many slots to
keep, not which eviction policy to run.

Paging works best on Mixture-of-Experts models, where each token only uses a fraction of the
weights. Chapter 7 explains why. For now: it means the streaming has a real chance of
keeping up, because you're not asking for all the weights on every token.

---

## How the pages get read

The dull-sounding part turned out to be worth a 3.23× speedup.

The original implementation copied bytes out of a memory-mapped region. Copying out of a
memory-mapped region looks free — a copy is just a copy — but touching an unmapped page
triggers a **major page fault**: the operating system suspends the thread, reads from disk,
and resumes the thread. Measured on a 35B model, the byte-copy approach took 98.5 major faults
per staging operation.

Using `pread` to read the span explicitly instead drops it to essentially zero: 1,330,594
faults became 147. The thread asks for bytes and gets them, rather than being trapped on
access.

Result on a 16 GB M1 Pro, running a 35B-A3B model through the Metal pager:

| approach | tok/s |
|---|---|
| byte-copy from mmap | 0.595 / 0.641 |
| `pread` | **1.967 / 2.022** |

A 3.23× improvement, measured with the arms interleaved in both orders so that warm-up
effects can't account for it.

Two things worth noting about that number. It's larger than the same change bought on a
different model family (1.26×), for a structural reason: this model stages more, smaller
spans, and `pread`'s advantage scales with the number of staging operations rather than the
number of bytes. And the alternative it beat isn't the real competition — the same model on
the CPU pager runs 1.52–1.73 tok/s, so the Metal path's actual advantage over the shippable
alternative is about 1.23×, not 3.23×.

That distinction — beating a strawman versus beating the real alternative — is the kind of
thing Chapter 11's "include the do-nothing arm" rule exists to force.

---

## How many slots?

With faults eliminated, the remaining cost is *how many* staging operations happen, which is
set by how many resident slots you keep. More slots means more cache hits means less staging.

That sounds monotone. It isn't.

| slots | tok/s | spread | hit rate | staging share |
|---|---|---|---|---|
| 8 | 1.858 | 1.9% | 29.5% | 67% |
| 16 | 1.934 | 0.6% | 50.0% | 65% |
| 32 | 1.909 | 1.2% | 64.8% | 64% |
| **64** | **2.191** | 3.0% | 76.9% | 58% |
| 128 | 1.583 | 20.8% | 85.5% | 32% |

N=64 is the optimum, at +14.8% over the default. N=128 has the **best** staging metrics on
the whole ladder — highest hit rate, staging share nearly halved, faults still zero — and
throughput collapses by 27.7%.

The cause is arithmetic the staging metrics cannot see:

```
  128 slots × 40 layers × ~1.5 MB per expert  ≈  7.7 GB of resident buffers

  on a 16 GB machine, of which the model is also streaming through the
  page cache → the buffers crowd out the cache they depend on
```

The 20.8% spread at N=128, an order of magnitude worse than any other point on the ladder, is
the thrashing signature.

The N=128 row is the clearest available argument for gating on the total rather than on
component metrics. Every intermediate measure improved and the quantity you care about got
worse.

---

## Two guard rails that failed

Both failures are recorded in `CLAUDE.md` and both generalize past this feature.

**The stopping rule stopped too early.** The sweep was pre-registered to stop when a doubling
of slots bought less than 5%. From 8 to 16 bought 4.1% — under the bar. Applied as written,
the sweep halts at N=16 and never sees the +14.8% at N=64, two doublings later.

The stopping rule silently assumed the curve was monotone-diminishing. Plateau-then-step is
the normal shape wherever a resource threshold gets crossed, and crossing a resource threshold
is exactly what happened at N=64. What saved the sweep was that the full ladder had *also* been
pre-registered, and the stopping rule and the full ladder disagreed. The lesson recorded was
not "write a better rule" but "pre-register two things that can disagree."

**The memory guard inverted.** A ceiling keyed on resident set size, written specifically to
catch this cliff, reported 263–426 MB at N=128 against 1154 MB at N=8 — *less* memory at the
failure point than at the baseline. Darwin's unified buffer cache reclaims under pressure, so
RSS reports what survived, not what was requested.

A guard that inverts under the guard's own condition is worse than no guard, because such a
guard reassures. The fix is to key budget checks on a quantity you compute yourself —
allocated bytes are known at build time, as the 7.7 GB arithmetic above shows — rather than on
the operating system's account of what remains.

---

## What it costs

Paging turns "cannot run" into "runs slowly," which is a large qualitative change and a modest
quantitative change. A 35B model on a 16 GB laptop runs at about 2.2 tok/s with the tuned
slot count. That is usable for some things and not for interactive chat.

The ceiling is known: with staging driven entirely to zero the model would reach about 5.2
tok/s on that machine, so slots can move it from 2.0 toward at most 5.2, and the slot lever
closes at 64. Getting further means making each staging operation cheaper, which is a
different problem.

There's a better case on the CUDA side, where a 26B-A4B model runs **fully GPU-resident** on
an 8 GB card — every expert on the device, no offload. That is one of the distinctive things
this engine does.

Chapter 7 explains why Mixture-of-Experts models are the ones where all this works.

---

*Sources: `docs/task-metal-expert-streaming-at-scale.md`, `docs/task-moe-streaming.md`,
`CLAUDE.md` (measurement discipline), `docs/benchmarks.md` §B4.*

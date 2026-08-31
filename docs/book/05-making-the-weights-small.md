# Chapter 5 — Making the weights small

*A model's weights are the bulk of what it is. This chapter is about storing them in fewer
bits, why that makes things faster for a reason people usually get wrong, and what "still
correct" means once you've changed the arithmetic.*

---

## The problem

A model with 7 billion parameters, stored as 16-bit floats, is 14 GB. That doesn't fit on a
consumer GPU and strains a laptop. A 27B model at 16 bits is over 50 GB.

More importantly, every single one of those numbers has to be read from memory for every token
you generate. Chapter 4 removed redundant computation; Chapter 4 did nothing about the fact
that decode reads the entire weight set once per token.

That is the real constraint. Decode is **memory-bandwidth-bound**: the arithmetic units spend
most of their time waiting for weights to arrive. If you halve the bytes, you nearly halve the
time — not because there is less arithmetic, but because there is less waiting.

```
  a 7-billion-parameter model, read once per generated token

    16-bit floats   14.0 GB per token
     8-bit ints      7.0 GB per token
     4-bit ints      3.5 GB per token

  the arithmetic is identical in all three rows; only the traffic changes,
  and on a memory-bound workload the traffic is what sets the time
```

This is a familiar shape if you've optimized Go: the win comes from cache behaviour and
memory traffic, not from instruction count.

---

## Quantization

Store each weight in fewer bits. Instead of a 16-bit float per number, use 8 bits, or 4.

The mechanism is scaling. Take a block of weights — say 32 or 64 weights — find the largest
magnitude in the block, and store a single scale factor for the block plus a small integer per
weight. To use a weight, multiply that weight's integer by the block's scale.

```
  storing:    scale = max|w| / 7            (7 = largest 4-bit signed int)
              int[i] = round(w[i] / scale)

  using:      w[i] ≈ int[i] × scale

  worked, with a block whose largest magnitude is 0.021:

    scale = 0.021 / 7 = 0.003

    w        w / scale      stored int    reconstructed
    +0.021     +7.00            +7           +0.021      exact
    −0.019     −6.33            −6           −0.018      off by 0.001
    +0.008     +2.67            +3           +0.009      off by 0.001

  the rounding error is the accuracy cost, and it is bounded by half a
  scale — which is why the block's largest magnitude sets the precision
  for every weight in that block
```

![One group of 32 weights as it sits in memory: 32 four-bit integers (16 bytes) plus one f32
scale (4 bytes), so 20 bytes for 32 weights — 5.0 bits per weight rather than 4. The scale is
amortized over the group, so a smaller group costs more per weight: group 16 is 6.0 bits, 32 is
5.0, 64 is 4.5, against 32 for f32 and about 8 for int8.](./05-fig-block-layout.svg)

Block size is the tuning knob. Smaller blocks track local variation better and cost more scale
factors; larger blocks are more compact and lose more precision where one block spans very
different magnitudes. Note that the scale factors are themselves stored, so a "4-bit" format
costs slightly more than 4 bits per weight in practice.

The naming you'll see: **int8** is 8 bits per weight, **int4** is 4. A suffix like `int8int8`
means both the weights and the activations flowing through them are 8-bit. `q4_k_m` is
llama.cpp's naming for one particular 4-bit scheme, and goinfer reads those files.

---

## The counter-intuitive result

The obvious expectation is that int4 is faster than int8 — half the bytes — but less accurate,
and that you would pick between int4 and int8 based on how much quality you can spare.

On Apple Silicon CPU decode in this repo, that expectation was wrong for a while, and then the
expectation became right for an interesting reason.

Measured after a fix to how the LM head is quantized:

| model | int4 | int8int8 |
|---|---|---|
| 0.5B | 81.9–83.75 tok/s | 85.25 tok/s |
| 1.5B | 39.1–40.7 tok/s | 37.56 tok/s |

**int4 now matches or beats int8int8 at both sizes, at half the weight RAM.** The current
guidance in [`docs/benchmarks.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/benchmarks.md) is that int4 is the right default on Apple Silicon CPU
decode.

What is instructive is what `docs/benchmarks.md` does with the *old* advice, which said the
opposite. The old advice is not deleted. The old advice is kept, marked superseded, with the
reason recorded: the old advice was a correctly-diagnosed reading of the machine at the time,
and the thing the old advice was measuring around — the LM head's drag on the int4 path — no
longer exists. The advice changed because the code changed, not because the earlier
measurement was wrong.

That is the repo's convention and Chapter 11 explains why it matters. A superseded number
that quietly disappears leaves the next reader unable to tell whether it was wrong or whether
the world moved.

---

## What "still correct" means

You changed the arithmetic. The model now computes with different numbers than the reference
implementation does. So in what sense is it the same model?

Two answers, and this repo uses both.

**Numerical parity against HuggingFace.** The reference is the same checkpoint running in
Python. goinfer's forward pass is gated against the Python reference, and there is a `parity`
runner in `cmd/gate` for exactly that comparison. Quantization is expected to move the numbers
slightly; the parity gate establishes how much the numbers move, and establishes that the
deviation stays within a stated tolerance rather than drifting.

**Bit-identical decode.** For a fixed quantization and a fixed seed, goinfer produces exactly
the same tokens every time. Bit-identical decode is a determinism claim, not an accuracy claim
— bit-identical decode says the engine is reproducible, which is what makes every other test in
the repo meaningful.

The distinction matters for what you may and may not do. Quantizing weights is a stated,
measured accuracy trade the user opts into by choosing a checkpoint format. Quantizing the KV
cache, as Chapter 4 mentioned, interacts with the determinism contract differently — which is
why it is treated as a separate decision rather than an obvious extension.

---

## Formats, briefly

Quantization schemes ship inside checkpoint files, and goinfer reads several:

- **safetensors** — HuggingFace's format, usually 16-bit, sometimes pre-quantized
- **GGUF** — llama.cpp's format, carrying its own quantization schemes and its tokenizer
- **GPTQ / AWQ** — quantization methods that use calibration data to decide where precision
  matters most, rather than treating every block identically
- **.giw** — goinfer's own pre-quantized bundle format, so the quantization work happens once
  rather than at every load

Reading GGUF matters practically: GGUF is what most locally-available quantized models are
distributed as. Reading safetensors matters because safetensors is what models are *published*
as, so goinfer can run a checkpoint on release day without waiting for someone to convert that
checkpoint.

---

## What it costs

Quantization is close to free in the sense that matters: it makes decode faster and models
smaller, at an accuracy cost small enough that 4-bit is a reasonable default.

The number worth carrying: int4 halves the weight memory against int8 and, on Apple Silicon
CPU decode, gives equal or better throughput. That combination is why it's the default there.

What quantization does *not* fix is a model that does not fit at all. A 27B model at 4 bits is
still over 15 GB, which fits neither of this repo's development GPUs. Chapter 6 is about what
you do when the model does not fit.

---

*Sources: `docs/benchmarks.md` §int4/int8int8 comparison, [`docs/task-w4a8-neon-bandwidth.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/task-w4a8-neon-bandwidth.md),
`cmd/gate` (parity runner), [`docs/api-tiers.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/api-tiers.md) (`.giw`).*

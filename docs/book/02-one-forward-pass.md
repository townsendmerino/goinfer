# Chapter 2 — One forward pass

*Chapter 1 ended with a list of integers. This chapter turns that list of integers into a
score for every possible next token. Everything after this chapter is about making this one
operation cheaper.*

---

## The shape of the thing

Give the model the token IDs so far. The model returns one number per vocabulary entry — a
score saying how good that vocabulary entry would be as the next token. That computation is
the **forward pass**, so called because data flows forward through the network with no loops.

In this repo the forward pass is `Model.forward` in `decoder/model.go`, a thin wrapper that
runs the layer stack (`runLayers`) and then projects the layer stack's result to scores
(`logitsFromHidden`). Under those two calls is essentially all of the work this book is about.

The forward pass has four stages. Ordinary function composition, no recursion, no branching on
content:

```
token IDs → embedding → [layer] × N → final vector → LM head → scores
```

---

## Stage 1: IDs become vectors

Each token ID indexes into a large table — the **embedding matrix** — and pulls out a vector
of a few thousand floats. Token 3797 always returns the same vector.

The embedding matrix is a lookup, not a computation. The embedding matrix was learned during
training, and the embedding matrix is a substantial fraction of the model's weights, because
the embedding matrix has one row per vocabulary entry:

```
  embedding matrix size = vocabulary size × hidden size

                        = 152,000 × 4,096
                        = 622,592,000 numbers, for the embedding matrix alone
```

Chapter 1's compression choice shows up here as memory: a larger vocabulary buys shorter
sequences and pays for the shorter sequences in embedding rows.

The vector pulled out of the embedding matrix is the model's working representation of that
token. Everything downstream operates on vectors of this width — the **hidden size** — and the
hidden size never changes as the vector passes through the layers. That fixed width is what
lets layers be stacked identically.

---

## Stage 2: The layer stack

The model applies the same *kind* of computation N times, with different weights each time. A
small model might have 24 layers; a large model 64 or more. `runLayers` is the loop over the
layers.

Each layer does two things in sequence, and each of the two things adds its result back into
the running vector rather than replacing the running vector:

```
h = h + attention(norm(h))
h = h + mlp(norm(h))
```

Those `h +` terms are **residual connections**. Residual connections matter more than residual
connections look: a residual connection means each layer proposes an adjustment to the
representation rather than rewriting the representation, so information from early layers
survives to the end. Without residual connections, deep stacks do not train.

The `norm()` calls rescale the vector to keep the vector's magnitude in a workable range. The
`norm()` calls are cheap and unglamorous and the arithmetic inside them is fussy —
normalization is one of the places where f32 versus f64 changes the answer, which Chapter 11
cares about.

### Attention — the part that makes it a language model

The MLP looks at one position. Attention is the only stage where positions see each other, and
attention is what makes this a *language* model rather than a very elaborate lookup table.

For each position, attention produces three vectors from the current representation: a
**query** (what am I looking for?), a **key** (what do I offer?), and a **value** (what do I
contribute if selected?). The query at the current position is compared against the keys at
every earlier position; the comparisons become weights; the weights are used to mix the values
together.

```
  generating position 4, which attends over positions 1..4

  position     1        2        3        4
  key         k₁       k₂       k₃       k₄
  value       v₁       v₂       v₃       v₄
                \        \        \        |
   q₄ · kᵢ →   0.1      0.6      0.1      0.2      ← compare, then normalize
                                                     so the weights sum to 1
  out₄ = 0.1·v₁ + 0.6·v₂ + 0.1·v₃ + 0.2·v₄         ← mix the values

  position 4 compares against 4 keys; position 8,000 compares against 8,000.
  That is the whole reason attention cost grows faster than sequence length.
```

The plain-Go reading: for each position, do a weighted average over all previous positions,
where the weights come from a dot product. Chapter 8 is about what attention costs when the
context is long, and Chapter 4 is about the cache that stops attention being far worse.

The comparison step is a matrix multiply (`MatmulQK` in the profiles), and the mixing step is
another matrix multiply (`MatmulAV`). On long-context prefill those two matrix multiplies
account for the overwhelming majority of the time — Chapter 8 has the figure.

### The MLP — where most of the weights live

After attention, each position independently goes through a small feed-forward network:
project up to a larger width, apply a nonlinearity, project back down. No cross-position
interaction at all.

The MLP is where most of a dense model's parameters sit. The MLP is also the stage that
Mixture of Experts replaces, and Chapter 7 is entirely about what happens when the MLP is
replaced.

---

## Stage 3: Vector to scores

After the last layer, one final normalization and one final matrix multiply against a matrix
of shape hidden size × vocabulary size. That final matrix multiply is `logitsFromHidden`, and
that final matrix is the **LM head**.

Out comes one score per vocabulary entry. Those scores are called **logits** — unnormalized
scores, not probabilities. Turning the logits into a choice is Chapter 3.

The LM head is a single matrix multiply but a large one, and the LM head's cost scales with
vocabulary size. The LM head has been the subject of real optimization work here: a fix to how
the LM head is quantized is what made int4 competitive with int8 on Apple Silicon, which
Chapter 5 covers.

---

## What the model does not have

Two absences that trip up people arriving from ordinary systems work.

**No memory between calls.** The forward pass is a pure function of the tokens you hand the
forward pass. Nothing persists between calls. If you want the model to know what was said
three turns ago, those tokens must be in the input. The KV cache in Chapter 4 is an
optimization that avoids recomputing work; the KV cache is not a memory of the conversation.

**No loop.** One forward pass produces one set of scores, which is one token. Generating a
sentence means running the whole forward pass once per token, feeding each output token back
in as input. That loop is Chapter 4's subject, and that loop is the reason a language model is
expensive to *run* rather than expensive to build.

---

## What it costs

Two regimes, and the distinction between the two regimes organizes most of this book.

Processing a *prompt* means running the forward pass over many positions at once. Parallelism
is available, because all the positions exist already, so the matrix multiplies are large and
the hardware is used well. Processing a prompt is **prefill**.

Generating a *reply* means running the forward pass once per token, on a single position,
because you cannot compute token N+1 until you have token N. The matrices are skinny, the
hardware is underused, and you are usually waiting on memory rather than on arithmetic.
Generating a reply is **decode**.

The shape of the matrix multiply is the whole difference:

```
  prefill, 2,048 positions at once      decode, 1 position at a time

    [ 2048 × hidden ] × [ weights ]       [ 1 × hidden ] × [ weights ]
     ^^^^                                  ^
     lots of rows to amortize each         one row per weight read:
     weight read over → compute-bound      → memory-bound
```

The same computation, two completely different performance problems. Chapter 8 takes the two
regimes apart properly.

For now, one measured anchor, on the same class of hardware — an M1 Pro CPU backend, int4:
generating tokens runs around 39–41 per second, while processing a 3,020-token prompt takes
101.6 seconds at the shipped default. Same forward pass, wildly different economics.

---

*Sources: `decoder/model.go` (`forward`, `runLayers`, `logitsFromHidden`),
`docs/how-inference-works.md` §Step 2, `docs/benchmarks.md` (decode-only, greedy, depth 128,
quiet box), `docs/queue-performance.md` (G16/G20 prefill).*

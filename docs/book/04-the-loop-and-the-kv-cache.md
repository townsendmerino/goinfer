# Chapter 4 — The loop and the KV cache

*One forward pass gives one token. Getting a paragraph means running it hundreds of times,
and doing that naively is quadratically wasteful. This chapter is about the memo table that
fixes it, and the costs the fix creates.*

---

## Autoregression

The loop is embarrassingly simple:

```
tokens := tokenize(prompt)
for !done {
    scores := forward(tokens)
    next   := sample(scores)
    tokens = append(tokens, next)
}
```

Each generated token becomes input for the next round. That is **autoregression**, and
autoregression is why generation is inherently sequential — you cannot start token N+1's
forward pass before token N exists, because token N is part of token N+1's input.

That single fact shapes everything. Autoregression is why decode underuses hardware, why
speculative decoding (Chapter 9) is interesting at all, and why generating 500 tokens takes
roughly 500 times as long as generating one token.

---

## The waste

Look at what the loop recomputes. On round 100, `forward` receives 100 tokens and runs the
entire stack over all of them. On round 101 it receives 101 tokens and runs the entire stack
over all of them — including the 100 it just did, producing identical results.

Attention is the stage where the recomputation bites. Chapter 2's description: for each
position, compare that position's query against the keys of every earlier position and mix the
corresponding values. The key and value vectors for position 5 depend only on the token at
position 5 and on what came before position 5. The key and value vectors for position 5 do not
change when you append token 101.

So the key and value vectors are being recomputed hundreds of times for nothing.

---

## The cache

Keep the key and value vectors. Keeping them is the **KV cache**, and the KV cache is the
single most important optimization in inference.

After computing the key and value vectors for a position, store the key and value vectors. On
the next round, compute key and value vectors only for the one new token, append the new
vectors to the store, and run attention against the whole stored set:

```
  WITHOUT the cache, round N          WITH the cache, round N

  recompute K,V for all N positions   compute K,V for 1 new position
  → work per round grows with N       → work per round is constant
  → total work over N rounds ~ N²     → total work over N rounds ~ N

  the comparison against N stored keys still happens either way —
  that term is what Chapter 8 is about
```

Per-token cost stops growing with the number of tokens generated so far and starts growing
only with the *comparison* against those tokens.

The Go reading is a memo table on a pure function, which is exactly what it is: keys and
values are a deterministic function of the prefix, and the prefix only ever grows at the end.
Append-only memoization on a function whose inputs are a prefix of a slice.

This turns generation from quadratic into something much closer to linear, and there is no
serious inference implementation without it.

---

## What the cache costs

You have traded compute for memory, and the memory is not small.

The KV cache holds, for every layer and every position, a key vector and a value vector. The
KV cache grows with every token generated, forever, for the life of the request:

```
  cache bytes = layers × positions × 2 × kv-width × bytes-per-number
                                     ^
                                     one key vector + one value vector

  qwen2.5-coder-1.5B, f32 cache: 28 layers, kv-width 256

    per position   28 × 2 × 256 × 4 B  =  57,344 B  =  56 KiB
    at 32,000 positions                =  1.84 GB

  the same model's weights on disk at q4_k_m are 1.12 GB — so at 32k
  context the cache is LARGER than the model it serves
```

Three consequences follow, and each is a real engineering thread in this repo.

**Long context is a memory problem before long context is a compute problem.** As the
arithmetic above shows, at 32,000 tokens the KV cache can exceed the model's weights in size.
On a machine that barely fits the weights, the KV cache is what actually runs you out of
memory.

**The KV cache can be quantized too.** Store keys and values in 8 bits instead of 16 and the
KV cache halves. Quantizing the KV cache is an accuracy trade like any other, and — as
`docs/ollama-chase.md` notes — KV quantization is not compatible with this repo's bit-identical
determinism contract in the way some other optimizations are. Chapter 11 explains why that
determinism contract is worth protecting.

**The KV cache can be reused across calls.** If two requests share a prefix — the same system
prompt, or the next turn of the same conversation — the cached keys and values for that prefix
are still valid. goinfer implements prefix reuse as prompt-prefix reuse, and goinfer also
persists warm caches to disk in a `.giw-kv` format so the caches survive a restart.

Prefix reuse is why agentic workloads care about the KV cache. An agent loop resends a growing
conversation every turn. Without prefix reuse, every turn pays full prompt-processing cost from
scratch — and Chapter 8 shows that prompt-processing cost is large.

---

## Where the cache doesn't work

Some newer architectures make the cache impossible, and the reason is instructive.

Standard attention keeps a **list**: one key/value entry per position. Truncating the list to
an earlier point is dropping entries off the end of the list. Reusing a prefix is keeping the
first N entries of the list.

Gated DeltaNet layers — used in several recent Qwen models — keep a **running total** instead.
Each token folds into a single fixed-size matrix that is updated in place. `deltaState` in
`decoder/deltanet.go` is that running total, and `deltaState`'s comment says plainly that the
running total is fixed size, independent of sequence length, and *not position-truncatable*.

```
  STANDARD ATTENTION — a list          GATED DELTANET — a running total

  [k₁v₁][k₂v₂][k₃v₃][k₄v₄]             S = fold(fold(fold(fold(S₀,t₁),t₂),t₃),t₄)
     ↑ keep the first 2 to                    ↑ one fixed-size matrix, updated in place
       reuse a 2-token prefix
                                       to get back to "after t₂" you would have to
  truncate = drop entries              UNDO t₃ and t₄ — and the gate discarded what
  reuse    = keep a prefix             you would need to undo them. Not recoverable.
```

Fixed size sounds like a win, and for memory the running total is a win: the running total does
not grow with context at all. But you cannot truncate a running total to an earlier position,
and you cannot invert the update because the gating deliberately discards information. So for
Gated DeltaNet layers:

- prefix reuse falls back to full recomputation
- speculative rollback (Chapter 9) is not possible at all

The loss of speculative rollback blocks an entire optimization family for those models.
Chapter 9 picks that up.

---

## What it costs

The KV cache's value is hard to state as a single number, because without the KV cache the
workload is a different shape entirely. The useful framing is what the KV cache's *absence*
does: on the models above where prefix reuse cannot apply, each turn of a multi-turn
conversation reprocesses the whole prompt from scratch.

At the shipped 6-worker default on CPU, a 3,020-token prompt takes 101.6 seconds to process.
An agent loop that pays that every turn, twenty turns deep, spends over half an hour on
prompt processing alone. With prefix reuse it pays it once.

Chapter 5 turns to the other memory problem: the weights themselves.

---

*Sources: `decoder/deltanet.go:145-153`, `docs/qwen3_5_moe.md`, `docs/ollama-chase.md`,
`docs/api-tiers.md` (`.giw-kv`), `docs/queue-performance.md` (prefill baselines).*

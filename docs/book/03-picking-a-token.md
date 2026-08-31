# Chapter 3 — Picking a token

*Chapter 2 produced one score per vocabulary entry. Choosing among them looks trivial and
isn't — it's where two of this repo's more instructive measurements live.*

---

## The naive version

You have 152,000 scores. Take the largest. Done.

Taking the largest score is **greedy decoding**, and greedy decoding is genuinely what happens
when temperature is zero. Greedy decoding is also deterministic: same prompt, same tokens,
every time. Determinism is worth more than determinism sounds, because determinism is what
makes the parity testing in Chapter 11 possible. If output were always random you could not
assert that two implementations agree.

But greedy output is noticeably flat. Taking the most likely token at every step produces text
that loops, repeats stock phrases, and commits early to whatever the text started saying. So
in practice you sample instead of taking the largest score.

---

## Sampling, and what temperature does

Sampling turns the scores into a probability distribution and draws from the probability
distribution. The conversion is **softmax**: exponentiate every score, then divide by the
total so the results sum to 1. **Temperature** is a divisor applied to the scores before the
exponentiation:

```
                exp(score[i] / T)
  p[i]  =  ---------------------------
            sum over j of exp(score[j] / T)

  T small  → dividing by a small number SPREADS the scores apart
             → exp() magnifies the gaps → the top token dominates → nearly greedy
  T large  → dividing by a large number SQUASHES the scores together
             → the gaps shrink → the distribution flattens → unlikely tokens get a chance
```

![The same three scores at three temperatures, drawn as probability bars. At T=0.2 the top token
takes 99.3% and the others are effectively unreachable. At T=0.7 the split is 79.8%, 19.1% and
1.1%. At T=1.5 it is 60.7%, 31.1% and 8.2%, and the third token becomes a live
option.](./03-fig-temperature.svg)

The divisor is applied to the scores, not to the probabilities, which is why a small change in
temperature can move behaviour a lot: the exponential amplifies whatever the division did.

- `T → 0` — greedy. Deterministic, repetitive.
- `T ≈ 0.2` — nearly greedy, slight variation. Code, structured output.
- `T ≈ 0.7–1.0` — the normal chat range.
- `T > 1.2` — increasingly incoherent.

Two filters usually run alongside temperature. **Top-k** keeps only the k highest-scoring
tokens before sampling. **Top-p** (nucleus sampling) keeps the smallest set of tokens whose
probabilities sum to p. Top-k and top-p both exist to cut off the long tail of nonsense that
flattening the distribution would otherwise let through.

---

## Why this chapter has performance content

Here is the part that surprises people. The sampler is not free, and the sampler's share of a
token's cost varies enormously by model.

Measured in this tree, on the same box:

| model | vocabulary | sampler's share of a token |
|---|---|---|
| phi3-mini | 32,064 | 5.4% |
| qwen2.5-coder-1.5B | 151,936 | 18.2% |
| qwen2.5-coder-7B | 151,936 | 7.5% |

Look at the last two rows. Same vocabulary, and the sampler's share differs by 2.4×. Yet the
sampler's *absolute* cost in those two cases was nearly identical — 1.009 ms and 1.102 ms.

That comparison gives the mechanism cleanly. **Vocabulary size sets the sampler's absolute
cost. The size of the decode step sets what fraction of the token that absolute cost
represents.** A big model does so much work per token that the sampler disappears into the
token; a small model with a large vocabulary spends nearly a fifth of every token choosing.

The relationship looks like a rule you could compute at load time. The relationship nearly is
such a rule. What stopped the relationship being a rule is the rest of this chapter.

---

## A shipped default that was wrong

goinfer has an optimization called **optimistic forward**. The idea: start the next token's
work speculatively before the current token's sampling has finished, on the guess that
sampling will pick the token optimistic forward assumed. When the guess is right, two stages
have been overlapped. When the guess is wrong, the speculative work is thrown away.

Whether optimistic forward pays depends on how often the guess is right, which depends on how
peaked the distribution is, which depends on temperature. At low temperature the top token is
nearly certain and the guess almost always lands. At chat temperatures the guess often does
not land.

Optimistic forward shipped enabled for all sampled decode. Measured, optimistic forward only
pays below about T = 0.26. Typical chat sampling sits at T = 0.7–1.0, squarely in the losing
regime, costing **5.5–6.8%**.

The payoff is badly asymmetric: the win at T = 0.2 is 1.1%, and the losses run to 6.8%. That
asymmetry, rather than the exact crossover location, is what settles the question — you would
want optimistic forward off at chat temperatures even if the crossover sat anywhere in a wide
band. Optimistic forward now gates at T ≤ 0.2.

---

## The measurement that nearly went wrong

The crossover was first measured with the repo's standard benchmark prompt file, whose
entries have four unique words each. That file is correct for measuring throughput: decode
cost doesn't depend on content, and repeated filler gives exact token counts.

The same prompt file is exactly wrong for measuring optimistic forward, because optimistic
forward's entire value is *how predictable the generated text is*, and four-unique-word filler
is maximally predictable.

On that filler, the 1.5B's crossover measured at T ≈ 0.95. On real prose it moved to
T ≈ 0.37. The largest swing landed exactly where a proposed adaptive-gating feature had its
whole case: what looked like a 5.1% win at T = 0.6 was a 4.3% loss.

The adaptive-gating feature was never built. The measurement that would have justified the
feature was an artifact of the input, and the input was correct — for a different question.

There is a broader point here that Chapter 11 develops. A benchmark input calibrated for one
class of measurement is not usable for another, and nothing warns you. The prompt file was
right. The instrument was right. The pairing was wrong.

---

## What it costs

Sampling is a few percent of a token on a large model and up to a fifth of a token on a small
model with a large vocabulary. That is enough to matter and not enough to be the main event.

The more useful takeaway is the shape of the result. A single constant — cap optimistic forward at T ≤ 0.2
— is approximately optimal across all three models measured, once they're measured on
realistic input. An earlier reading suggested three models needed three different answers,
which would have justified a per-model adaptive mechanism. Realistic input collapsed the
spread to something one constant covers.

Chapter 4 moves to the optimization that actually dominates: the cache that stops every token
from redoing all the work of every token before it.

---

*Sources: [`decoder/spec_optfwd.go`](https://github.com/townsendmerino/goinfer/blob/main/decoder/spec_optfwd.go), [`docs/QUEUE.md`](https://github.com/townsendmerino/goinfer/blob/main/docs/QUEUE.md) (G26 ladder, sampler-share and crossover
measurements, the prompt-confound retraction), [`scripts/prompts.json`](https://github.com/townsendmerino/goinfer/blob/main/scripts/prompts.json).*

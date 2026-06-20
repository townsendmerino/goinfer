# 02 — Cache / n-gram drafting

> Status: **proposal**. Depends on [00-core](./00-core.md). Low effort; high ROI on
> code-edit, RAG, summarization, and agent loops. Build alongside [01](./01-grammar-fused.md).

## Idea

A large fraction of real output is **copied verbatim from the input or from earlier
output**: code edits echo the surrounding file, RAG answers quote retrieved
passages, summaries reuse phrasing, and agent loops repeat a fixed system prompt,
tool schemas, and prior tool results. At those positions the target distribution
puts most of its mass exactly where a copied n-gram would — low `TV(p, q)`, high
acceptance ([00-core](./00-core.md) §1) — for essentially zero compute.

This is the cheapest drafter that works on free text. Recent results
("Cacheback", REST, prompt-lookup) show cache-only drafting is competitive on these
workloads with no draft model at all.

## Mechanism

Maintain a **suffix automaton** (or a rolling n-gram index) over the concatenation
of: the current prompt + everything generated so far in this session. On each step,
take the last `m` generated tokens as a key, look up where that suffix occurred
before, and propose the token(s) that followed — as a draft chain, or as several
branches when the suffix had multiple continuations (a natural tree for
[03](./03-router-tree.md)).

- **`QProb` in the rejection step must be `q(x)=1` (a point mass), not the empirical
  frequency.** An n-gram proposal is *deterministic* — we commit to one token, we do
  not sample from a distribution — so the lossless coupling treats it as the point
  mass `δ_x`: accept with probability `min(1, p(x)/q(x)) = min(1, p(x)) = p(x)`, and on
  rejection sample the correction from `p` with `x` removed and renormalized
  (`(p − δ_x)+`). Using a fractional `q` like `0.93` here would make the accept
  probability `min(1, p(x)/0.93) ≥ p(x)` — it **over-accepts** and silently biases the
  output away from `p`. The empirical continuation frequency is still valuable, but as
  a **feature for `α̂`** ([00-core](./00-core.md) §4), never as the `q` in the
  rejection arithmetic. (If we ever proposed *stochastically* — sampling a branch from
  the index's frequencies — then `q` would legitimately equal those frequencies; we do
  not, and shouldn't, since the deterministic top continuation is both cheaper and
  higher-acceptance.)
- Match length is the key acceptance signal: longer / more specific suffix matches
  predict higher acceptance ([00-core](./00-core.md) §4). Feed match length into
  `α̂` and let [04](./04-adaptive-depth.md) extend the copy run while `α̂` stays high.

## Why it suits goinfer

- The server **already keeps warm KV sessions** and the token history they imply, so
  the corpus to index is already resident — the automaton is a cheap side-structure,
  not new state to persist.
- It is purely a token-stream structure: no model weights, no GPU, pure Go, trivially
  within the no-cgo default build.
- It shines on `cmd/serve`'s agent traffic (fixed system prompt + tool specs repeated
  every turn) and on the RAG coding agent in `demo/agent`.

## Expected payoff

Large committed-per-verify on `codeedit` / `rag` / `agent` suites; little to nothing
on novel `reasoning` text (where 05's head earns its keep). Because cost is ~zero,
it is almost always worth running as one source in the [03 router](./03-router-tree.md)
tree even when its hit rate is low — a miss costs nothing, a hit is free tokens.

## Variants to evaluate

- **Suffix automaton** (online, `O(1)` amortized append, exact longest-suffix match)
  vs a simpler fixed-order **n-gram hash** (cheaper, approximate). Start with the
  hash for a baseline, measure the gap.
- **Scope of the corpus:** prompt-only vs prompt+session vs prompt+session+a static
  domain corpus (e.g. the user's codebase). More corpus = more hits but more index
  cost and staleness risk; measure the curve.
- **Session-local online learning:** the index improves within a session as the
  model establishes repeated structure — quantify the within-session acceptance lift.

## Risks / open questions

- Index memory growth on very long sessions — bound it (sliding window / cap) and
  measure the acceptance cost of the bound.
- Tokenizer alignment: index and propose at the token level to avoid byte/҂token
  mismatches with the verifier.
- Staleness when a static corpus drifts from the live distribution — a miss is cheap,
  but a confidently-wrong long branch wastes verify width; cap branch length by `α̂`.

## Validation plan

- Correctness: lossless by construction (verifier owns `p`); assert output ≡ baseline.
- Speed/acceptance: `codeedit`, `rag`, `agent` suites in the [00-core](./00-core.md)
  harness; report committed/verify and the acceptance-vs-match-length curve.

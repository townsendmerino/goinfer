# 00 — Core: the acceptance model, instrumentation, and the verify/rollback substrate

> Status: **proposal**. This is the foundation; every spoke depends on it. Build it first.

This doc answers the question "*is there a mathematical relationship behind successful
speculation, and can we instrument and analyze it?*" — yes, and the relationship is
exact. It then specifies the shared machinery the spokes plug into: the draft-tree
representation, the single-pass verifier, per-family state rollback, and the
benchmark/instrumentation harness.

---

## 1. The governing identity

Let `p(· | x_<t)` be the **target** next-token distribution and `q(· | x_<t)` the
**draft** distribution at the same context. Lossless speculative decoding
(Leviathan et al. 2023; Chen et al. 2023) accepts a draft token `x ~ q` with
probability `min(1, p(x)/q(x))`. The marginal probability that a drafted token is
accepted is therefore

```
α(x_<t)  =  Σ_x q(x)·min(1, p(x)/q(x))  =  Σ_x min(p(x), q(x))  =  1 − TV(p, q)
```

where `TV(p, q) = ½ Σ_x |p(x) − q(x)|` is total-variation distance.

**This is the whole game.** Per-position acceptance is *exactly* one minus the
total-variation distance between the draft and target distributions. Successful
speculation ⇔ the drafter matches the target in TV at the contexts you actually
visit. Every spoke (grammar, n-gram, head, …) is just a different mechanism for
making `q` close to `p` cheaply, in a different region of context space:

- **Grammar-forced positions** (01): the legal set is a singleton, so `p` collapses
  onto one token and any `q` that respects the grammar has `TV → 0`, `α → 1`.
- **Verbatim/copy positions** (02): when the continuation echoes the prompt, an
  n-gram `q` puts mass exactly where `p` does → low TV.
- **Novel-text positions** (05): a feature head learns `q ≈ p` directly.

Note what does *not* drive acceptance: target **entropy** by itself. A *perfect*
drafter (`q = p`) accepts with probability 1 even at maximum entropy, because
`TV(p, p) = 0`. Acceptance loss comes entirely from **mismatch**, not uncertainty.
Entropy matters only indirectly — high-entropy contexts are *empirically* where a
cheap `q` diverges most from `p`. That distinction is measurable (§4) and tells you
whether a rejection was "genuine uncertainty" or "a fixable draft miss."

## 2. How far to speculate (length & speedup)

For a single chain of depth `γ` with per-position acceptance `α` (treated i.i.d. as
a first approximation), the expected number of tokens committed per verification
step — accepted draft tokens before the first rejection, plus the one bonus token
the target always produces — is

```
E[committed]  =  (1 − α^(γ+1)) / (1 − α)
```

With `c` = cost of one draft step relative to one target step, the wall-clock
improvement factor over plain decoding is

```
S(γ)  =  (1 − α^(γ+1)) / [ (1 − α) · (γ·c + 1) ]
```

The `+1` in `(γ·c + 1)` is the target verify pass, charged as **one** target step —
which holds only while that batched forward stays **memory-bound** (the weight stream,
read once, dominates; the extra `γ` positions ride along ~free). This is exactly
goinfer's single-stream on-device regime, but it is an *assumption with a ceiling*:
once a tree (03) or depth (04) grows wide/deep enough to turn the verify
compute-bound, the verify cost stops being `1` and this formula over-promises. That
ceiling is the budget `B` the harness measures per backend (§7); treat `S(γ)` as valid
for `γ`-and-tree-width within `B`.

(Leviathan et al. 2023). Maximizing `S` over `γ` gives the optimal draft depth
`γ*(α, c)`: it grows as `α → 1` and as `c → 0`. The marginal reading — *extend the
draft while the expected marginal accepted token is worth more than the marginal
draft+verify cost it adds* — is the basis of [04 adaptive depth](./04-adaptive-depth.md).
The i.i.d. assumption is only a starting point; the real `α` is position- and
context-dependent, which is exactly what §3–§4 measure.

## 3. Why goinfer can measure this exactly

The expensive quantity in §1 is `p` (the target distribution). But **at verify time
the target forward pass computes `p` for every speculated position anyway** —
accepted or rejected — and the drafter already produced `q`. goinfer owns both, in
process, in pure Go. So for each speculated position we can record the *exact*
per-position acceptance `α = 1 − TV(p, q)` (a low-variance, continuous signal),
not merely the realized accept/reject Bernoulli outcome.

Proposed per-position trace record (illustrative):

```go
type SpecTrace struct {
    // identity / context
    Pos        int       // depth within the draft block / tree
    Source     DrafterID // grammar | ngram | model | head
    Forced     bool      // grammar pinned this token (expect α≈1)
    Token      int

    // cheap, available at DRAFT time (usable to decide depth/routing)
    QTop1      float64   // draft top-1 probability
    QEntropy   float64
    Streak     int       // consecutive accepts so far in this block
    StateNorm  float64   // optional: hidden-state / residual norm

    // available at VERIFY time (target side)
    PTop1      float64   // target top-1 probability
    PEntropy   float64
    TV         float64   // total-variation(p, q)
    AcceptProb float64   // = 1 − TV  (exact per-position acceptance)
    Accepted   bool      // realized outcome of rejection sampling
}
```

Logging `SpecTrace` behind a build flag / sampling rate turns every run into a
labeled dataset: `(features) → AcceptProb`. Dump to JSONL; analyze offline.

## 4. The analysis: from traces to a predictive model

The scientific core. Two questions, both answerable from the trace dataset:

**(a) What predicts acceptance?** Regress `AcceptProb` (or the binary `Accepted`)
on the **draft-time-only** features — `QTop1`, `QEntropy`, `Pos`, `Source`,
`Forced`, `Streak`, `StateNorm`. A tiny model suffices (logistic regression →
gradient-boosted trees → small MLP, in that order of complexity). Call the fitted
function `α̂(features)`. Its calibration and AUC/Brier score are a *direct, numeric
answer* to "is there a mathematical relationship and how strong is it." High AUC ⇒
success is predictable from cheap signals ⇒ adaptive depth and routing will work.
Low AUC for a source ⇒ that source's acceptance is near-random given what you can
see cheaply ⇒ don't try to gate it, just measure its base rate.

Expected findings to validate (hypotheses, to be confirmed empirically):

- `Forced = true` → `α̂ ≈ 1` (grammar). Should be near-deterministic.
- High `QTop1` (confident draft) → high `α`. The cheapest single predictor.
- `α` decays with `Pos` (errors compound down a chain) → motivates trees over long
  linear drafts (03) and bounded depth (04).
- For n-gram hits, `α` jumps with match length / context specificity.

**(b) Where does the rejection mass come from?** Bucket rejected positions by
target entropy `PEntropy`. Split total lost acceptance into:

- *Irreducible* — high `PEntropy`, target genuinely uncertain; even `q = p` only
  helps so much. Not worth more drafter effort.
- *Fixable* — low `PEntropy` (target nearly certain) yet `TV` large; the drafter
  simply guessed wrong. This is the addressable budget, and it tells you *which
  spoke* to invest in (e.g., lots of fixable mass on structural tokens ⇒ ship 01).

`α̂` then feeds back into the live system:

- **[04 adaptive depth]:** keep extending while `α̂^depth · (target step time)` >
  marginal cost; stop otherwise.
- **[03 router]:** choose source `s* = argmin_s [ draftcost(s) + (1 − α̂_s)·penalty ]`
  per position.

> The full experimental runbook — JSONL schema, the model-fitting ladder, calibration
> checks (reliability diagram / ECE), and the rejection-mass decomposition — is
> [06 acceptance analysis](./06-acceptance-analysis.md).

## 5. The block-optimality gap (an upper bound worth knowing)

Token-by-token rejection sampling is **not** optimal for *block* acceptance: the
optimal coupling between the draft-sequence distribution and the target-sequence
distribution (a maximal-coupling / optimal-transport problem) can commit strictly
more tokens than greedy per-token verification. "Block verification" results
formalize this. We don't have to implement optimal coupling, but the harness should
estimate the **gap** between our realized accepted-length and the coupling upper
bound on a sample of contexts. If the gap is small, per-token verify is fine and
effort belongs in the drafter; if it's large, better *verification* (not drafting)
is the lever. This keeps us from over-investing on the wrong side of the loop.

## 6. The verify + rollback substrate

The greedy/linear case already exists: `GenerateSpeculative` runs one target
`ForwardN` over `[cur, draft…]`, accepts the matching prefix, replaces the first
mismatch with the target's own argmax, and rolls back with `TruncateTo`. The work
here is to (a) generalize the accept test from greedy argmax-equality to **sampled
rejection sampling** (the `min(1, p/q)` rule, lossless under a sampler), (b) generalize
the linear draft to a **tree**, and (c) extend rollback past softmax/GQA. The
substrate below subsumes the existing path as its greedy, single-source, linear,
softmax-only special case.

All spokes share one verifier. Given a `DraftTree`, it:

1. Runs the target model over all tree nodes in **one batched forward pass** with a
   tree/causal attention mask so each node attends only to its ancestors.
2. Walks root→leaf applying per-token rejection sampling, selecting the **longest
   accepted path**. The `q` used in the accept test is the drafter's *proposal*
   distribution at that node: for a **deterministic** drafter (grammar 01, n-gram 02)
   that is the point mass `q(x)=1`, so accept reduces to `min(1, p(x)) = p(x)`; for a
   **distributional** drafter (model/head 05) it is the head's softmax. Getting this
   `q` wrong is a silent losslessness bug, not a crash — see [02](./02-cache-ngram.md).
   - **Linear chain:** on the first rejection, sample the corrected token from the
     residual `(p − q)+` (normalized) and stop.
   - **Tree (multi-child):** rejection is *not* terminal. When a child is rejected,
     subtract its mass and try the next sibling against the renormalized residual
     (SpecInfer / token-tree verification); only fall back to a fresh `(p − Σq)+`
     sample once every sibling at that node is exhausted. The naive "stop on first
     rejection" rule is lossless but throws away the other branches' work — and a
     *wrong* sibling-retry is a distribution bug. This is the hardest correctness
     surface in [03 router](./03-router-tree.md); gate it per family.
3. **Rolls back** model state to the accepted prefix and emits the bonus token.

Step 3 is where attention family matters — and where naive implementations are
silently wrong:

| Family (goinfer) | KV/state shape | Rollback on partial accept |
|---|---|---|
| softmax / GQA | per-token K,V appended | truncate appended entries — trivial **(shipped: `TruncateTo`)** |
| MLA (DeepSeek/Kimi) | per-token latent KV | truncate latent entries — trivial |
| **Mamba-2 (SSM)** | single rolling recurrent state | **checkpoint state at block start; restore on partial accept.** Cannot "subtract" tokens. |
| **Gated DeltaNet (linear attn)** | single rolling state matrix | **same: checkpoint + restore.** |

For SSM / linear-attention families, the verify pass must either (a) snapshot the
recurrent state before the draft block and restore it, then re-advance over the
accepted prefix, or (b) use a chunked scan that retains per-step states within the
block. Either way the cost of a snapshot is bounded by the state size (small,
fixed) — not the sequence length. **This must be correctness-gated against
non-speculative decode for every family**, because a wrong rollback is a silent
distribution bug, not a crash. This subtlety is also a differentiator: goinfer is
one of the few runtimes that even runs these families.

```go
// Illustrative — not an API commitment.
type StateCheckpoint interface {
    Snapshot(ctx *DecodeState) Restorer // cheap for SSM/linear (fixed-size state)
}
type Restorer interface{ RestoreTo(prefixLen int) }
```

## 7. The benchmark / instrumentation harness

One harness, shared by every spoke, so numbers are comparable.

**Metrics (per workload):**
- `α̅` — mean per-position acceptance (`1 − TV`), and its distribution.
- `committed/verify` — mean tokens committed per target forward pass.
- end-to-end **tok/s** and **ms/token**, vs the non-speculative baseline.
- `c_eff` — measured draft:target cost ratio on the actual backend (CPU SIMD vs
  WebGPU differ; record both).
- correctness flag — greedy bit-exactness vs baseline; sampled in-distribution test.

**Workload suites** (acceptance is workload-dependent, so always report per-suite):
- `structured` — JSON/tool-call/schema-constrained (exercises 01).
- `codeedit` / `rag` — high prompt↔output overlap (exercises 02).
- `chat` / `reasoning` — novel prose (exercises 05).
- `agent` — long loops with fixed system prompt + tool specs (exercises 02 + warm KV).

**Outputs:** a comparison table (this suite × each spoke), plus the raw `SpecTrace`
JSONL for the §4 analysis. The harness is the arbiter for the status table in the
[README](./README.md): a spoke is "benched" only when it has numbers here.

## References

> ⚠️ Verify every arXiv ID below against the live listing before treating this page as
> a citable disclosure — the two 2024/2025 IDs are from memory and unconfirmed; a wrong
> number in a defensive publication is worse than no number.

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative
  Decoding*, 2023 — the TV identity, expected length, speedup formula.
- Chen et al. *Accelerating LLM Decoding with Speculative Sampling*, 2023 —
  correctness of the rejection step.
- *A Theoretical Perspective for Speculative Decoding Algorithm*, arXiv:2411.00841 —
  optimality analysis, distribution-matching view.
- *Token-Driven GammaTune*, arXiv:2504.00030 — adaptive `γ` calibration (relevant to 04).
- Block-verification / maximal-coupling literature — the §5 upper bound.

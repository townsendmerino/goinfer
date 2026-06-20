# 01 — Grammar-fused speculation

> Status: **proposal**. Depends on [00-core](./00-core.md). Lowest effort, highest ROI
> on goinfer's headline workloads (structured output / tool calls). Build first among the spokes.

## Idea

goinfer's `constrain` package already enforces a byte-level grammar by masking
illegal tokens to −∞ at every step (JSON Schema / Go-struct → grammar). At many
positions the grammar's automaton has **exactly one legal continuation** — the
required key after `{`, the `:` after a key, the closing `}` / `]` / `"`, the comma
between elements, structural whitespace, an `enum`/`const` literal, the fixed
scaffold of a tool call. At those positions the target distribution `p`, *after the
mask is applied*, is a point mass on a single token regardless of the model's raw
logits.

By the [00-core](./00-core.md) identity `α = 1 − TV(p, q)`: if the grammar pins the
token, any `q` that respects the grammar has `TV = 0`, so **acceptance is 1 at zero
model cost**. The grammar *is* a perfect drafter for the structural skeleton. We
just have to read it out instead of stepping the model.

## Mechanism

Extend the grammar automaton to expose, at the current state, not only the legal
*set* but a **forced run**: the maximal sequence of tokens that are determined
because each successive automaton state has a single legal byte-path (modulo
tokenizer segmentation). Concretely:

1. At each decode position, query the automaton for the forced run `f_1..f_k`
   (`k ≥ 0`). These become draft nodes with `Forced = true`, `QProb = 1`.
2. Hand them to the shared verifier as a draft chain. The target still runs **one**
   forward pass over the run (lossless verification — we never skip the target's
   right to disagree where it legally *could*), but commits all `k` at once.
3. Where the grammar leaves genuine freedom (a string body, a number, a free-text
   field), emit no forced nodes; let another drafter (02 n-gram, 05 head) or plain
   decode handle that slot. This is the natural hand-off point for [03 router](./03-router-tree.md).

A stronger variant worth measuring: when a forced run is **both** grammar-pinned
*and* tokenization-unique (only one token sequence encodes those bytes), the target
cannot change it without violating a constraint that is *already enforced* — so the
forced bytes can be committed and the target pass over them skipped entirely. This
is a genuine compute skip, not just a guaranteed accept. Gate it carefully (see
risks) and prove it lossless before enabling.

## Why it suits goinfer

- The DFA already exists and already runs on the hot path — marginal cost is reading
  an extra field from the automaton.
- Structured output, `response_format`, and `tool_choice: any|tool` already ride
  this grammar in `cmd/serve`. Those are exactly the workloads with the highest
  structural-token fraction, i.e. the most free acceptance.
- It composes: grammar fills the skeleton, a content drafter fills the slots, the
  verifier commits the merged tree once.

## Expected payoff

On `structured` workloads the committed-per-verify rate should track the structural
fraction of the output. For dense JSON / tool-call traffic that fraction is large.
The [00-core](./00-core.md) harness reports it directly; the §4 analysis predicts
`α̂ ≈ 1` on `Forced` rows as a sanity check.

## Risks / open questions

- **Tokenizer segmentation.** "Grammatically forced bytes" ≠ "forced tokens": the
  same bytes can tokenize multiple ways, and the model's choice among equal-byte
  tokenizations is a real (if usually trivial) degree of freedom. The forced-run
  extractor must operate at the byte level and only emit token-forced nodes when the
  segmentation is unique, or fall back to verifying.
- **Losslessness of the compute-skip variant** must be proven, not assumed —
  property test: every grammar-skipped generation equals the fully-verified one.
- **Sampling temperature.** Under the mask, forced positions are temperature-
  invariant (one legal token), so this is safe across sampler settings — confirm in
  the harness.
- Interaction with `StopWhenComplete()` and EOS handling at the end of a constrained
  object.

## Validation plan

- Correctness: property test asserting grammar-fused output ≡ non-speculative
  constrained output, greedy and sampled, across the existing schema test corpus.
- Speed/acceptance: `structured` suite in the [00-core](./00-core.md) harness;
  report committed/verify and tok/s vs constrained-but-non-speculative baseline.

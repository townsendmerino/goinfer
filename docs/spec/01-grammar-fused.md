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

A stronger variant worth measuring — but the saving is narrower than it first looks.
When a forced run is **both** grammar-pinned *and* tokenization-unique (only one token
sequence encodes those bytes), the target cannot change it without violating a
constraint that is *already enforced*. So those positions need no acceptance test
(`α≡1`) and we can **skip the lm_head projection + sampling** over them — the win is
that work, plus collapsing what would have been `k` sequential decode steps into one
batched verify pass.

What we **cannot** skip in general is the **transformer forward** over the forced run:
its KV / hidden states are inputs to every later token (the first *free* slot after
the run attends to them), so the attention/MLP layers must still run to populate the
cache. The forward over a forced run is skippable *entirely* only when the run is
**terminal** — nothing downstream consumes its KV (e.g. the closing `"}` immediately
before EOS). Treat "skip lm_head/sampling on forced positions" (always sound) and
"skip the whole forward on a terminal forced run" (sound only at end-of-generation) as
two separate, separately-gated optimizations. Prove each lossless before enabling.

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

## Increments

- **Inc 1 (done):** `constrain.Masker.ForcedRun(max)` + `Grammar.Clone()` (all three
  grammars). Non-mutating forced-run extractor: probes on a cloned grammar, returns
  the run of positions where exactly one surface token is legal and the doc can't yet
  end. `TestForcedRun` (model-free, byte vocab) gates it.
  - **FINDING that reshapes 01:** goinfer's grammars (json / schema / tool) permit
    **optional whitespace at every structural boundary** (`isWS` throughout). So a
    whitespace token is *also* legal at `{`, after a key, before `:`/values, etc. —
    strict single-token forcing fires only **inside fixed literals** (object keys,
    enum/const values), NOT at the scaffolding the doc's intro assumed. The win is
    therefore narrower than "the whole structural skeleton" until/unless a
    whitespace-free grammar mode exists.
- **Inc 2 (done — GO, with a design correction):** measured on real DeepSeek/Qwen BPE
  (`TestGrammarForcedFraction`, qwen2.5-coder vocab) over 3 tool/schema outputs:
  - **Strict token-forcing (inc-1 `ForcedRun`) = 0%** everywhere. With a real BPE
    vocab, even when the *bytes* are forced (inside `"approved"`), MANY tokens are
    legal byte-prefixes of the continuation (`o`, `ov`, `oved`, …) — so "exactly one
    legal token" almost never holds. The inc-1 token-level criterion is the WRONG
    granularity.
  - **Byte-level forcing ceiling = 44–74%** (weather 54%, record 44%, enum-heavy 74%).
    Structural forcing is real and material — inside keys (`ocation"`) and enum/const
    values (`approved"`, `elsius"`, `rue`) the byte continuation is fully determined;
    the non-forced bytes are the optional-ws boundaries, the first char of each
    key/enum, and free values (`Paris`, `30`).
  - **Verdict: pursue 01, but the drafter must be BYTE-LEVEL.** Extract the forced
    byte-run, retokenize it canonically (the tokenizer's own encoding — what the model
    most likely emits), and propose those tokens. Acceptance is <1 (the model's
    tokenization of the forced bytes is a real degree of freedom — exactly the §risks
    "tokenizer segmentation" point), but the verify keeps it lossless and the masked
    target distribution over forced literals is concentrated, so acceptance should be
    high. The 44–74% byte ceiling is the headroom.
- **Inc 3 (done — works end-to-end, lossless):**
  - **3a:** `constrain.Masker.ForcedBytesRun(max)` — byte-level forced-run primitive
    (`TestForcedBytesRun`).
  - **3b:** `decoder.GrammarDrafter` (forced byte-run → canonical retokenize) +
    `GenerateGrammarSpeculative` (CPU, greedy). The verify masks every position with a
    grammar clone rolled forward over the accepted prefix (`Masker.GrammarClone` /
    `MaskAt` / `Commit` / `TokenBytes`), and the grammar advances only over emitted
    tokens — so no rewind is needed. **Gate `TestGrammarSpecParity`: token-identical to
    plain constrained `Generate`** (greedy, same schema), output
    `{"location": "Paris", "unit": "celsius"}`, with realized **acceptance 0.40 /
    1.13 tok per verify round** — the forced runs fire (keys/enum), losslessly.
  - The win is modest on a tiny JSON object and bounded by tokenization agreement
    (the model's split of forced bytes vs canonical); it scales with the structural-
    token fraction / output length (inc-2 ceiling 44–74% bytes).
- **Inc 4 (done — density characterization, `TestGrammarSpecHarness`):** tok-per-verify
  across schema densities (qwen2.5-coder-0.5b, greedy, CPU):

  | schema | tok/round | accept | |
  |---|---|---|---|
  | free-string (1 string field) | 1.10 | 0.50 | only the key forces |
  | mixed (string + 2-enum) | 1.11 | 0.22 | |
  | enum-heavy (3× 2-choice enums) | 1.05 | 0.14 | the choice breaks the run at each disambiguation point |
  | **fixed-keys (3× single-value enums)** | **1.45** | 0.53 | keys AND values fully determined |

  **Verdict:** a real, lossless, zero-draft-cost win, but **modest (~1.05–1.45 tok/
  round)** — best only when keys *and* values are fully fixed (const / single enum).
  Multi-choice enums force less than intuition suggests (the choice point breaks the
  forced run). On the batched-CPU path the wall-clock win is therefore ~1.1–1.3×
  on structured output, not the "free structural skeleton" the intro imagined.
  Worth wiring into `cmd/serve` *if the integration is cheap* (a miss costs ~nothing,
  so it's safe to always run on constrained requests); not worth heavy investment.
  Remaining follow-ups (unfunded unless the win justifies): sampled + resident
  grammar-spec paths, the `cmd/serve` constrained/tool routing.

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

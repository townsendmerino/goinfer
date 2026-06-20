# Speculative decoding in goinfer — design index

> Status: **proposal / RFC**. Cut 2026-06-20. These are design docs for work not yet
> built — **but not greenfield**: `decoder/speculative.go` already ships a
> parity-gated *greedy, single-draft-model, fixed-K* speculator (`GenerateSpeculative`)
> with batched `ForwardN` verify, softmax/GQA rollback via `TruncateTo`, GPU-resident
> verify, and `SpecStats` telemetry. That is the substrate [00-core](./00-core.md)
> extends; the new work is sampled-lossless rejection, the richer drafters (01/02/05),
> trees (03), adaptive depth (04), instrumentation (`SpecTrace`), and per-family
> rollback beyond softmax/GQA. Treat every page here as a dated public disclosure
> (defensive publication): publish before discussing the approaches elsewhere.

## Thesis

Speculative decoding speed is governed by one quantity:

```
speedup  ≈  (tokens accepted per verify)  /  (draft cost per verify)
```

So "smarter speculation" means: push **acceptance toward 1.0** while spending
**near-zero draft compute**. goinfer is the ideal place to chase this — it is
single-stream, latency/bandwidth-bound, on-device (no continuous batching, one
decode worker per model). That is precisely the regime where speculative decoding
wins biggest, and where a high-batch server's throughput tradeoffs do *not* apply.

The design rejects the usual "one draft model, fixed window K" shape. Instead:

> A **speculation controller** picks, per position, the cheapest draft source
> likely to be accepted, merges candidates into one **draft tree**, and verifies
> the whole tree in a single target forward pass — with correct state rollback
> across every attention family goinfer runs.

Why this fits goinfer specifically:

- It already owns the **full forward pass in pure Go**, so it can expose hidden
  states (for a feature head) and dump draft/target distributions (for
  instrumentation) essentially for free.
- It already has a **byte-level grammar engine** (`constrain`), which can act as a
  zero-cost drafter for structured output — the workloads goinfer is sold on.
- It already keeps **warm KV sessions**, so the token history needed for
  cache/n-gram drafting is already resident.
- It runs **softmax/GQA, MLA, Mamba-2 (SSM), and Gated DeltaNet (linear attn)** —
  families whose verify/rollback semantics differ, which most spec-dec
  implementations get wrong. Getting it right is a differentiator.

## The map

| # | Doc | What it adds | Effort | Expected ROI | Status |
|---|-----|--------------|--------|--------------|--------|
| 00 | [Core: math, instrumentation, verify/rollback](./00-core.md) | The shared substrate every spoke builds on. Acceptance theory, the measurement layer, the draft-tree/verify loop, per-family rollback, the benchmark harness. | High | Prerequisite | proposed |
| 01 | [Grammar-fused speculation](./01-grammar-fused.md) | Grammar-forced tokens from the `constrain` DFA as a ~zero-cost drafter. | Low | High on structured/tool-call output | proposed |
| 02 | [Cache / n-gram drafting](./02-cache-ngram.md) | Suffix-automaton over prompt + session history; copy verbatim runs for free. | Low | High on code-edit / RAG / agent loops | proposed |
| 03 | [Router + draft trees](./03-router-tree.md) | Fuse grammar + cache + model/head into one verified tree, chosen by predicted acceptance. | Med | Compounds the others | proposed |
| 04 | [Adaptive draft depth](./04-adaptive-depth.md) | Replace fixed K with optimal-γ / entropy-gated depth from the acceptance predictor. | Low–Med | Broad, cheap | proposed |
| 05 | [EAGLE-3 feature head](./05-eagle3-head.md) | Feature-level autoregressive draft head over target hidden states; pure-Go. | High | Matches general-purpose SOTA | proposed |
| 06 | [Acceptance analysis playbook](./06-acceptance-analysis.md) | The experimenter's runbook: trace schema → calibrated `α̂` → where to invest. Operationalizes 00-core §4. | Med | Powers 03/04, orders the backlog | proposed |
| 07 | [Stage B: M=K GEMM verify](./07-stageb-gemm-verify.md) | Wire the existing (gated) tiled W8A8 GEMM into the resident verify so projection weights stream once across K rows — the only thing blocking a GPU speculative win. | High (runner surgery) | Unlocks GPU; CPU already ships | design |

Build order: **00 first** (nothing is testable until verify + rollback +
instrumentation exist), then 01 and 02 (cheapest wins, and they generate the
acceptance data that powers everything else), then 04, then 03, then 05.
[06 (the analysis playbook)](./06-acceptance-analysis.md) runs continuously from the
moment 01/02 start emitting traces — it is how 04 and 03 obtain their predictor.

## Shared vocabulary

These types anchor the design and are referenced by every spoke; the full set
(plus `SpecTrace`, `StateCheckpoint`, and the verify/rollback detail) lives in
[00-core](./00-core.md). Illustrative sketches — not an API commitment.

```go
// A Drafter proposes continuation tokens for the current context as a tree of
// candidates. Each node carries the draft probability q used for that token,
// which the verifier needs for lossless rejection sampling.
type Drafter interface {
    Draft(ctx *DecodeState, budget int) *DraftTree // budget caps tree width
    Cost() float64                                 // relative draft cost, for the router
}

// The Verifier runs the target model over the whole tree in ONE forward pass,
// applies per-token rejection sampling (lossless), returns the longest accepted
// path + one bonus token, and rolls back KV/SSM state to the accepted prefix.
type Verifier interface {
    Verify(ctx *DecodeState, tree *DraftTree, s Sampler) (accepted []int, bonus int)
}
```

## Non-goals / invariants

- **Lossless only.** Every scheme here must reproduce the target model's sampling
  distribution exactly (rejection sampling / the grammar mask). No "approximate"
  speedups that change outputs. This is testable and must be gated.
- **Parity-gated, like the rest of goinfer.** Acceptance and speed are reported
  on the shared harness (00-core); correctness is asserted against non-speculative
  decode, bit-for-bit on greedy and in-distribution on sampled.
- **Pure-Go, no new cgo** in the default build. The GPU verify path may use the
  existing WebGPU backend under `-tags gpu`.

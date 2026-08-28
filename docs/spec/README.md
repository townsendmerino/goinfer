# Speculative decoding in goinfer — design index

> Status: **largely BUILT** (was an RFC cut 2026-06-20; implemented over 2026-06-21 on
> branch `spec-decode-ngram`). Every spoke below is implemented and lossless-gated except
> where marked kill-gated/parked. See **[Where this landed](#where-this-landed-2026-06-21)**
> for the honest scorecard and [experiments.md](./experiments.md) for the dated run log.
> Treat every page here as a dated public disclosure (defensive publication).

## Where this landed (2026-06-21)

The whole program was built and measured. The lossless invariant held throughout — every
scheme is parity-gated against non-speculative decode (greedy bit-exact). The headline
findings, honest:

- **The CPU winner is n-gram (02), shipped and serve-wired (`--spec ngram`).** It wins on
  copy-heavy traffic (code edits / RAG / agent loops) because its draft is *free*. This is
  the one unambiguous production win.
- **A model drafter cannot win on CPU — and the reason is acceptance, not the verify.**
  EAGLE-3 (05) is built end-to-end and lossless (greedy + tree drafting), but it's a
  wall-clock *loss* on CPU (~0.4×). The batched verify *is* amortized — `forwardN` streams
  each weight once across the K rows, and a CPU verify node costs ~0.5 of a target step
  (`AdaptiveDepth.Theta`); that amortization is exactly why n-gram wins. EAGLE loses because
  acceptance is low (~1.6 tok/verify) and, unlike n-gram, its draft is **not free**: a head
  forward per drafted token plus a per-round capture/correction forward — overhead that ~1.6
  accepted tokens can't pay back. tok/verify gains are real (2.1 with trees), but the lever
  is **α**, not verify batching (see 05's kill-gate); they translate only when the target
  step is expensive enough (big model / GPU) to dwarf the draft cost.
- **The GPU verify (07, "Stage B") is parked, not dead.** On small models the batched verify
  is ~break-even (compute-bound at M=8). A large-dim microbench found it *does* amortize on
  big models (70B-layer dims: 1.37× at M=8, 1.58× at M=4) — but only for **short linear**
  drafts (trees are the wrong shape: wide M kills amortization), and it needs a >8 GB GPU to
  host a real large model. Unpark trigger: big GPU + a large model worth serving fast.
- **Acceptance is predictable and now calibrated (06).** α̂_ngram(match_len) is a tiny
  monotone table (AUC 0.82, held-out mean ECE 0.14); α̂_grammar trace-fit to ~0.20. Read that
  0.20 precisely: it indicts *this* drafter, not grammar speculation in principle. Free
  grammar drafting must *guess* the tokenization of the forced bytes, and the canonical
  retokenization diverges from how the model splits them under the mask (the mismatch
  compounds with depth). The fix — a tokenization-aware forced drafter — is described but
  unbuilt (01/06); accurate grammar drafting fundamentally needs the model. So grammar ranks
  weakest *as currently built*, and the router treats it as the floor. The 03 router ranks
  sources by these calibrated α̂, shrunk online toward each source's running accept rate to
  absorb cross-workload drift.
  **Cross-model check (added qwen2.5-coder-0.5b + llama-3.2-1b, different tokenizers):**
  α̂_grammar is *stable* — 0.20 on both, the fragility above is not a qwen quirk. α̂_ngram does
  *not* cross-calibrate (llama AUC 0.64, held-out ECE 0.29, and the qwen "sql inverts" finding
  did not reproduce) — but it still *ranks* n-gram ≻ grammar correctly on both models, so the
  table is a routing prior and the online correction (not per-family static tables) is the
  lever for the drift. Every number here is {qwen2.5-coder-0.5b, llama-3.2-1b} on an RTX
  2070S — two points, not a law.
- **Grammar-fused (01) is shipped + serve-wired** for constrained/tool requests, but modest
  (the tokenization fragility above caps it).
- **Trees + the per-family rollback substrate are built and gated** (tree attention in
  `forwardN` is bit-identical to causal; recurrent families guarded out of rollback).

Net: the production win (n-gram) shipped; the SOTA drafter (EAGLE) is built, characterized,
and correctly gated as GPU-blocked; the analysis layer (06) is calibrated and validated.

> Original RFC context (still accurate as substrate): `decoder/speculative.go` shipped a
> parity-gated greedy, single-draft-model, fixed-K speculator with batched `ForwardN`
> verify, softmax/GQA rollback via `TruncateTo`, GPU-resident verify, and `SpecStats`
> telemetry — the base [00-core](./00-core.md) extended.

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
| 00 | [Core: math, instrumentation, verify/rollback](./00-core.md) | The shared substrate every spoke builds on. Acceptance theory, the measurement layer, the draft-tree/verify loop, per-family rollback, the benchmark harness. | High | Prerequisite | ✅ built |
| 01 | [Grammar-fused speculation](./01-grammar-fused.md) | Grammar-forced tokens from the `constrain` DFA as a ~zero-cost drafter. | Low | High on structured/tool-call output | ✅ shipped + serve-wired (modest — tokenization-fragile) |
| 02 | [Cache / n-gram drafting](./02-cache-ngram.md) | Suffix-automaton over prompt + session history; copy verbatim runs for free. | Low | High on code-edit / RAG / agent loops | ✅ **shipped, `--spec ngram` (the CPU win)** |
| 03 | [Router + draft trees](./03-router-tree.md) | Fuse grammar + cache + model/head into one verified tree, chosen by predicted acceptance. | Med | Compounds the others | ✅ router shipped (calibrated α̂ + online correction); tree-verify built; full trees kill-gated for n-gram |
| 04 | [Adaptive draft depth](./04-adaptive-depth.md) | Replace fixed K with optimal-γ / entropy-gated depth from the acceptance predictor. | Low–Med | Broad, cheap | ✅ shipped |
| 05 | [EAGLE-3 feature head](./05-eagle3-head.md) | Feature-level autoregressive draft head over target hidden states; pure-Go. | High | Matches general-purpose SOTA | ✅ built end-to-end + lossless (greedy+trees); ⛔ CPU wall-clock loss → needs GPU (07) |
| 06 | [Acceptance analysis playbook](./06-acceptance-analysis.md) | The experimenter's runbook: trace schema → calibrated `α̂` → where to invest. Operationalizes 00-core §4. | Med | Powers 03/04, orders the backlog | ✅ α̂_ngram + α̂_grammar trace-fit + online correction + held-out validation |
| 07 | [Stage B: M=K GEMM verify](./07-stageb-gemm-verify.md) | Wire the existing (gated) tiled W8A8 GEMM into the resident verify so projection weights stream once across K rows — the only thing blocking a GPU speculative win. | High (runner surgery) | Unlocks GPU; CPU already ships | ⏸ NO-GO small models; **parked** conditional-GO for ~70B + short drafts (needs >8 GB GPU) |
| 08 | [DSpark / DFlash block drafters](./08-dspark-dflash.md) | Pretrained block drafters (DeepSeek DSpark, z-lab DFlash): per-block draft cost + published tok/verify far above EAGLE's — the α lever 05/06 named, and 07's short-linear-draft shape. | Med–High | Model-quality α on novel text + constrained traffic, near-free draft | 📋 proposal (cut 2026-08-15), queued as P10 |
| 09 | [MTP / NextN heads](./09-mtp-heads.md) | Checkpoint-shipped multi-token-prediction heads (DeepSeek-V3 layout) on families already on disk: a head trained jointly with its own target, against 05's imported general head. | Med (loader + one block) | Higher α than an imported head, if joint training is the lever | ✅ Gate 0 + **Gate 1 PASSED** (0.8B: code 2.02 / math 2.91 / chat 2.48 tok/verify vs 05's cross-target 1.60) — **MECHANISM only**; Gate 2 not evaluable at that scale; returns for a separate decision |
| 10 | [Gating optimistic forward](./10-optfwd-gate.md) | Not a new drafter: when the SHIPPED optimistic-forward overlap should run. Measured unconditional-on, it wins up to 7.4% on large-vocab models and loses up to 6.8% on small ones, and no temperature constant fits both. | Low–Med (decision logic only) | Recovers ~6% on small-vocab sampled decode without giving up the large-vocab win | 📋 design (cut 2026-08-27), **awaiting sign-off — no code**; kill-gated on window-level variance |

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

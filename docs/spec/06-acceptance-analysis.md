# 06 — Acceptance analysis playbook

> Status: **proposal**. The experimenter's runbook. Depends on [00-core](./00-core.md)
> (which defines the `SpecTrace` record and the acceptance identity). This doc turns
> trace dumps into three artifacts: a calibrated predictor `α̂`, a per-workload
> diagnostic report, and a go/no-go signal for each spoke. Hand this to whoever runs
> the experiments.

## Status (2026-06-21): first α̂ artifact shipped — calibrated α̂_ngram, wired into the router

The trace foundation and the first calibrated predictor are built and gated, following
the doc's rule to *measure predictability before shipping anything heavier*:

- **Emitter (§2):** `decoder.SpecTrace` (schema-matched) + the `specTracer` hook in the
  n-gram path + `TraceCollector` (JSONL sink, `MeanAccept`). The harness dumps the §06
  dataset with `GINFER_SPECTRACE_OUT`.
- **Predictability verdict (§6):** `TestNgramAlphaPredictor` runs the copy-heavy
  workloads, buckets traces by suffix `match_len`, and reports the calibration curve +
  AUC. Result (qwen2.5-coder-0.5b): **match_len → accept AUC = 0.818**, monotone curve
  (len2 → 0.70, len3 → 0.86, len11 → 0.92, len16-capped → 0.97). So acceptance IS
  predictable from the one cheap draft-time feature — the minimal 1-D predictor suffices;
  no tree ensemble, no Python needed (§4 ladder stops at step 1).
- **α̂_grammar — trace-fit (the surprise):** `TestGrammarAlphaPredictor` runs a pure
  `GrammarDrafter` over JSON-schema workloads (every drafted token grammar-FORCED) and
  measures their acceptance. Result: **~0.20, not the ≈1 the "forced ⇒ certain" intuition
  predicts** — the drafter's *canonical* retokenization of the forced bytes usually
  differs from how the model tokenizes those same bytes under the mask, and the mismatch
  compounds within a run (depth-0 ~0.24 → depth-1 ~0.08 → depth-2 ~0). So **grammar is the
  WEAKEST source**, not the strongest: the original `0.90` guess was backwards, and only
  measuring caught it.
- **Artifact + wiring (§9):** `ngramAlpha(match_len)` — a tiny monotone interpolation
  table fit from the n-gram curve (the "small coefficient table" the §0 boundary permits
  in the pure-Go runtime). `NgramDrafter.Confidence` returns this **calibrated accept
  probability** (not raw match-length) and `grammarConf` is the trace-fit **0.20** on the
  same scale, so the [03 router](./03-router-tree.md) compares sources principally: any
  n-gram copy (≥0.70) outranks tokenization-fragile grammar, which drafts only when n-gram
  has no copy (a 20% free-token shot still beats nothing, and a miss costs ~nothing).
  Routing only affects speed — verify is lossless either way; `TestGrammarRouterSpec` holds
  at 5.67 tok/round on the agent loop, `TestNgramAlphaTable` gates monotonicity + range +
  the grammar<n-gram ordering.

- **Online correction (§9) — built.** `RouterDrafter` now blends each source's static α̂
  with its **running realized accept rate**: `effective = (w·rate + prior·α̂)/(w + prior)`,
  an EMA-tracked `rate` (λ=0.85, ~7-round memory) and a capped observation weight `w` that
  ramps trust as outcomes accumulate. The verify loop feeds it via the new
  `OutcomeRecorder` interface (`RecordOutcome(accepted, drafted)`). Cold-start (no outcomes)
  returns pure α̂, so it's byte-identical to the prior router until evidence arrives; then a
  source whose static α̂ is wrong for the live workload is demoted within ~10 rounds
  (`TestRouterOnlineCorrection`: a 0.70-static / 0-accept source falls below a 0.20-static /
  always-accept one). This is the fix for the cross-workload mis-rank the α̂_grammar work
  surfaced (a spurious short n-gram match in prose).
  **Scope:** the router is created per-request in `cmd/serve`, so the correction adapts
  *within* a long generation but resets each request — it does not yet persist across
  requests. True cross-session workload-drift adaptation needs shared, thread-safe per-model
  stats (the documented next step); the within-request form is the tested foundation.

- **Cross-workload held-out validation (§3) — done.** `TestNgramAlphaCrossWorkload` scores
  the shipped `ngramAlpha` table (fit on the copy-heavy `specWorkloads`) against five
  workloads it was NOT fit on (csv/config/markdown/sql/prose-rep), via per-workload AUC +
  ECE = Σ_bin (n/N)·|empirical − ngramAlpha(match_len)|. Result (qwen2.5-coder-0.5b): **mean
  held-out ECE = 0.14** — it generalizes acceptably on average. But the per-workload spread
  is real and instructive: config 0.04 / prose-rep 0.05 (tight) vs markdown 0.24 / sql 0.23
  (drift), and **sql AUC = 0.32 — the match_len→accept relationship INVERTS there** (longer
  copies accept less). So the static table is a good *prior* but genuinely mis-ranks on some
  workloads — precisely the drift the §9 online correction is for. The two results compose:
  ship the calibrated prior, correct it online per workload.

- **Cross-MODEL validation (2026-06-22) — done; it changed the framing.** Re-ran the α̂
  measurements on **Llama-3.2-1B-Instruct** (BPE, a different tokenizer family than qwen2;
  spec-safe + batchable; lossless gate green). Two clean results:
  - **α̂_grammar is STABLE across tokenizers: 0.205 on llama vs 0.20 on qwen**, same within-run
    compounding (depth-0 0.24 → depth-1 0.09). Grammar's tokenization-fragility is not a qwen
    quirk — `grammarConf = 0.20` is a robust cross-model constant. It stands.
  - **α̂_ngram does NOT cross-calibrate.** match_len → accept AUC drops to 0.64 (qwen 0.82),
    held-out mean ECE rises to 0.29 (qwen 0.14, over the 0.25 bar), and markdown breaks down
    (ECE 0.70). The qwen **sql inversion did not reproduce** (llama sql AUC 0.50, flat) — it
    was workload×model-specific, not a law. **But** every llama n-gram bucket still accepts
    0.71–0.98 ≫ grammarConf 0.20, so the table **mis-calibrates cross-model yet does not
    mis-rank**: the router's ordering (n-gram copy ≻ grammar) holds on both models.
  - **Verdict (C): lean on the online correction, not more static tables.** The shipped
    `ngramAlpha` is kept as a serviceable routing *prior* (correct ordering on both models);
    a per-family static table is rejected (llama's own fit is noisy AUC 0.64, and precise
    calibration is workload- AND model-dependent — static tables can't track that). The
    per-source EMA correction (§9, built) is what absorbs the drift. No anchor/grammarConf
    change — the data doesn't demand one.

Remaining §06 follow-ups (unfunded unless the win justifies): **cross-request** persistence
of the online-correction stats (shared per-model, thread-safe) — now the clearly-justified
next step, since drift is both workload- AND model-dependent; a **byte-level forced drafter**
that would raise α̂_grammar past the tokenization wall (a separate build); and the offline GBM
path (§4) only if a future source proves un-predictable from 1-D features. Validated on
{qwen2.5-coder-0.5b, llama-3.2-1b} — two points is a line, not a law; don't over-generalize.

## 0. The boundary (keep the runtime pure-Go)

The runtime's only two jobs here are: **(a)** emit `SpecTrace` JSONL behind a build
flag / sample rate, and **(b)** at serve time, load a *tiny* exported `α̂` artifact and
evaluate it per step. Everything in §3–§8 below — loading dumps, fitting models,
plotting — is **offline analysis** and lives outside goinfer's no-cgo runtime (use
Python/sklearn/whatever; it never ships in the binary). This preserves the pure-Go,
no-cgo invariant: heavy analysis deps stay in a separate tooling repo/dir, and the
only thing that crosses back into the runtime is a small coefficient table or
serialized tree ensemble. State this boundary in any PR so the invariant isn't
threatened by adding analysis dependencies.

## 1. What this produces

1. **`α̂(features)`** — a calibrated acceptance predictor over *draft-time-only*
   features. Consumed by [04 adaptive depth](./04-adaptive-depth.md) and
   [03 router](./03-router-tree.md).
2. **Diagnostic report** per (model, workload): predictability (AUC / Brier / ECE),
   reliability diagram, rejection-mass decomposition, block-optimality gap.
3. **Investment signal** — where the *fixable* acceptance mass is, i.e. which spoke
   buys the most committed-tokens-per-verify next.

## 2. Data: the `SpecTrace` JSONL schema

One JSON object per **speculated position** (one line). The exact per-position
acceptance `accept_prob = 1 − TV(p,q)` is the primary label; `accepted` (the realized
rejection-sampling outcome) is kept for the reality check in §5.

**Run header** (emit once per run; join to rows on `run_id`):

```json
{
  "run_id": "2026-06-20T18:02:11Z-a1b2c3d",
  "git_sha": "a1b2c3d",
  "harness_version": "1",
  "model": "qwen2.5-coder-1.5b-instruct",
  "family": "qwen2",              // attention/sequence-mixing family
  "quant": "q4_k_m",
  "backend": "cpu-simd",          // cpu-simd | webgpu
  "sampler": {"temp": 0.7, "top_p": 0.95, "top_k": 40, "seed": 1234},
  "workload": "codeedit",         // structured | codeedit | rag | chat | reasoning | agent
  "drafter_config": "ngram(m=3)+grammar",
  "sample_rate": 1.0,             // fraction of positions logged
  "lossless_verified": true       // this run's output == non-spec baseline
}
```

**Per-position row:**

```json
{
  "run_id": "2026-06-20T18:02:11Z-a1b2c3d",
  "seq_id": 42,                   // which request/sequence
  "step": 311,                    // decode step within the sequence
  "pos": 2,                       // depth within the draft block/tree
  "parent": 1,                    // tree parent index; -1 = root (linear => pos-1)
  "source": "ngram",              // grammar | ngram | model | head | plain
  "forced": false,                // grammar pinned this token
  "token": 1917,

  // --- draft-time features (available BEFORE verify; legal inputs to α̂) ---
  "q_top1": 0.93,                 // draft top-1 probability
  "q_entropy": 0.41,              // nats
  "q_margin": 0.78,               // p1 - p2 of the draft
  "streak": 2,                    // consecutive accepts so far this block
  "depth": 2,                     // == pos; kept explicit
  "ngram_match_len": 7,           // source=ngram: matched suffix length (else null)
  "state_norm": 12.3,             // optional: residual/hidden-state norm

  // --- verify-time (target side; LABELS/diagnostics, NOT α̂ inputs) ---
  "p_top1": 0.95,
  "p_entropy": 0.33,
  "p_margin": 0.86,
  "tv": 0.04,                     // total-variation(p, q)
  "kl_qp": 0.06,                  // KL(q || p), optional
  "accept_prob": 0.96,            // = 1 - tv  (exact per-position acceptance)
  "accepted": true,               // realized rejection-sampling outcome
  "corrected_token": null         // token sampled from (p-q)+ when rejected
}
```

Notes: log at the token level (align with the verifier to avoid byte/token skew);
record `sample_rate` if subsampling; never log to the hot path synchronously — buffer
and flush.

## 3. Building the dataset (and avoiding leakage)

- One row per position; join the run header onto each row.
- **Labels:** `accept_prob` (continuous, primary — it's the exact expectation, so
  lower variance) and `accepted` (binary, for the §5 reality check).
- **Features `X`:** the draft-time block only. **Exclude every verify-time column**
  (`p_*`, `tv`, `kl_qp`, `accept_prob`, `accepted`, `corrected_token`) from `X` — they
  are not available when `α̂` is consulted (before paying for verify), so using them
  is leakage that inflates AUC and breaks in production.
- **Splits:** group by `seq_id` (never split positions of the same sequence across
  train/test). Additionally hold out a whole `workload` to measure cross-workload
  generalization, since acceptance is workload-dependent.

## 4. Fitting `α̂`

Climb the ladder; stop when held-out AUC plateaus:

1. **Logistic regression** on `accepted` (or fractional logit on `accept_prob`).
   Interpretable — coefficients tell you which signals carry the relationship.
2. **Gradient-boosted trees**, with monotonic constraints where they make physical
   sense: `α̂` increasing in `q_top1`, `q_margin`, `ngram_match_len`; decreasing in
   `q_entropy`, `depth`. Monotonicity buys robustness and easier calibration.
3. **Tiny MLP** only if 1–2 plateau below a useful AUC.

**Fit per source.** Train `α̂_grammar`, `α̂_ngram`, `α̂_model`, `α̂_head` separately —
their feature supports differ (`grammar`: `forced→1`; `ngram`: `match_len` dominates),
and [03 router](./03-router-tree.md) needs per-source `α̂_s` anyway.

Report per model/workload: held-out AUC, Brier, log-loss, feature importances.

## 5. Calibration (the part that matters most for adaptive depth)

For [04](./04-adaptive-depth.md), **calibration beats discrimination**: an
over-confident `α̂` produces over-long drafts that waste verify width. So check, on a
held-out fold:

- **Reliability diagram:** bin `α̂` into deciles; plot mean predicted vs empirical
  accept rate. The diagonal is perfect; above it = under-confident, below = over.
- **ECE / MCE:** expected and maximum calibration error.
- **Brier decomposition:** reliability − resolution + uncertainty.

If miscalibrated, fit **isotonic regression** (or Platt / temperature scaling) on a
held-out fold — never on the training fold. Shipping bar into 04: monotone reliability
and **ECE < ~0.03** (tune the target to how aggressively 04 extends depth).

> **Which consumer this bar actually governs (2026-06-21).** As shipped, the calibrated
> `ngramAlpha` table does **not** drive [04 adaptive depth](./04-adaptive-depth.md): the
> `AdaptiveDepth` controller runs its own EMA of *realized* acceptance (`Observe`/`alpha`),
> so depth self-calibrates online and the ECE-<0.03 bar above — which was written for depth —
> does not gate the table. The table feeds `NgramDrafter.Confidence()` → the
> [03 router](./03-router-tree.md) for source *ranking*, where **ordering** (AUC 0.82) matters
> more than absolute ECE, so the shipped mean ECE 0.14 is acceptable for its real consumer.
> The residual risk is therefore a *ranking* failure, not a depth one: on workloads where the
> `match_len → accept` relation **inverts** (sql AUC 0.32, §3) the table actively mis-ranks,
> and the online correction that catches it is **per-request** (it resets each request), so a
> fresh sql-like request mis-ranks for ~10 rounds before the EMA demotes the source.
> Cross-request persistence of those stats is the documented follow-up.

## 6. The predictability verdict (the original question, quantified)

- **Headline:** held-out AUC + ECE of `α̂` per workload. High AUC + low ECE ⇒ success
  is predictable from cheap signals ⇒ 03/04 will pay. Low AUC for a source ⇒ its
  acceptance is ~random given cheap features ⇒ don't gate it, just use its base rate.
- **Minimal predictor:** report AUC of `q_top1` *alone*. It often captures most of the
  signal — meaning the runtime may need only a calibrated 1-D function of draft top-1
  prob, not a tree ensemble, on the hot path. Establish this before shipping anything
  heavier.
- **Sanity:** `α̂` on `forced=true` rows must be ≈1 — a built-in check that the grammar
  source and the instrumentation agree.

## 7. Rejection-mass decomposition (where to invest)

For **rejected** positions, scatter `accept_prob` against `p_entropy` and split the
lost acceptance:

- **Irreducible** — high `p_entropy` (target genuinely uncertain). Even `q=p` caps the
  gain here; not worth more drafter effort.
- **Fixable** — low `p_entropy` (target near-certain) yet high `tv` (the draft simply
  missed). This is the addressable budget.

Then **bucket the fixable mass** by `source` / token type / context:

| Fixable mass concentrated in… | Ship |
|---|---|
| structural / grammar-legal-but-not-yet-forced tokens | [01 grammar-fused](./01-grammar-fused.md) |
| positions whose token appears in prompt/session | [02 cache/n-gram](./02-cache-ngram.md) |
| novel prose / reasoning | [05 EAGLE-3 head](./05-eagle3-head.md) |

For each bucket, estimate the **committed/verify uplift** if that mass were captured —
an upper bound on each spoke's payoff, which orders the backlog with numbers instead
of intuition.

## 8. Block-optimality gap (the verification-side lever)

Token-by-token rejection is not block-optimal ([00-core](./00-core.md) §5). On a
*sample* of contexts (this is expensive — offline only), compute the optimal-coupling
accepted length (maximal coupling between the draft-sequence and target-sequence
distributions) and compare to the realized token-by-token accepted length.

- **Small gap** ⇒ per-token verify is fine; spend effort on drafters.
- **Large gap** ⇒ better *verification* (block verification) is the lever, independent
  of how good the drafter gets.

Report the gap per workload so the team doesn't over-invest on the wrong side of the
loop.

## 9. Feeding it back into the runtime

- **Export** `α̂` as a small artifact (coefficient table or compact GBM) versioned by
  `git_sha` + `harness_version` + model + workload-mix. Keep per-step evaluation cheap
  — it's on the hot path in [04](./04-adaptive-depth.md)/[03](./03-router-tree.md).
- **Online correction:** maintain a running per-source empirical accept rate as a
  shrinkage fallback when `α̂` is uncertain or the live workload drifts from the
  training mix ([03](./03-router-tree.md) already calls for this).
- **Re-fit cadence:** when a new family or workload lands, re-dump traces and re-fit;
  track `α̂` versions next to the parity gates so a predictor change is auditable.

## 10. Deliverables checklist

- [ ] `SpecTrace` JSONL emitter behind a flag (§2), lossless-verified runs only.
- [ ] Trace dumps for each workload suite (§00-core harness), keyed by run header.
- [ ] Fitted per-source `α̂` + calibration map, exported artifact (§4–§5, §9).
- [ ] Diagnostic report: AUC/Brier/ECE table, reliability diagrams, rejection-mass
      decomposition, block-gap (§5–§8).
- [ ] Backlog ordering: fixable-mass uplift per spoke (§7).

## References

- See [00-core](./00-core.md) §1–§5 for the acceptance identity, length/speedup math,
  and the block-optimality framing this playbook operationalizes.
- *A Theoretical Perspective for Speculative Decoding Algorithm*, arXiv:2411.00841.
- *Token-Driven GammaTune*, arXiv:2504.00030 — adaptive-`γ` calibration in the wild.

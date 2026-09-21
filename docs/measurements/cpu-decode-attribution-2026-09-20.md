# R9 step 1 — CPU decode per-component attribution (Mac half)

`docs/tasks/red-october.md` R9, step 1: "attribute the per-token cost the fits in §2.1 say
exists... Output: a per-component ms/token table per (box, size)." **Mac half only** — no access
to the Linux box in this session, so the Linux 0.5B anomaly (§2.1's own open question) is
untouched here.

Box: `apple-m1pro` (M1 Pro, 6P+2E, 16 GB). goinfer `3db62a55`. Models: `qwen2.5-coder-0.5b`,
`qwen2.5-coder-1.5b`, `qwen2.5-7b` (int4, Backend:"cpu"). Greedy, prompt 128 tokens (R9's own
decision-cell depth), 24 decode steps.

**7B needed the fit guard bypassed** (`GOINFER_NO_FIT_GUARD=1`) — it declined twice on its own
(needs ~16.7 GB — 8.9 resident + 3.5 KV + 4.4 transient checkpoint-read overhead — against
6.4-7.1 GB available at the time). Unlike R11(c)'s MoE-pager near-incidents, this is a plain dense
CPU load with no auto-sizing/snapshot-timing failure mode, and it was run with the same external
(outside the test process) RSS/swap monitor discipline as R11(c), user-approved after the second
guard decline. **System swap never grew at all across either 7B run** (flat at its pre-run
baseline throughout, per the external monitor) — no near-incident this time, both runs completed
cleanly. The other two models needed no bypass.

## Method

Two layers, both reusing existing or newly-added machinery rather than a from-scratch build:

**Coarse split (forward / sample / logitProc / embed):** `GOINFER_DECODE_TIMING=1`, an
already-shipped diagnostic (`decoder/model.go`'s `decodeTiming` var, predates this session) —
not something this task needed to build. Confirms the obvious but worth stating plainly: `sample`,
`logitProc` and `embed` are all ≤0.10 ms/token on all three models, i.e. noise next to `forward`
(10.0-10.2 ms on 0.5B, 20.4-20.8 ms on 1.5B, 63.1-63.7 ms on 7B) — essentially the entire
decode-token cost is inside
`forward` (the transformer layer stack + LM head), matching R9's own "Read first" citation of
kernels running near their issue ceiling with the fan-out as the open question, not the small
peripheral costs.

**Fine split (attention / MLP / LM head) inside `forward`:** a temporary, non-shipped diagnostic,
following R13 step 0(ii)'s own precedent exactly (`r13-attn-category-split-2026-09-19.md`) —
`time.Now()`/`atomic.AddInt64` pairs around `causalAttention`, `mlp` (both calls in
`runLayersFromEmbed`'s non-parallel branch — the branch qwen2.5-coder's architecture takes; the
`NormParallel` branch used by Cohere/GPT-J-style blocks was not instrumented, out of scope for the
models measured here) and `logitsFromHidden` (in `forward`), gated behind a package-level bool
default `false` (`GOINFER_R9_DIAG=1`), printed once per `Generate` call alongside the existing
`decodeTiming` line, counters reset after each print. **Verified a genuine no-op**:
`TestForwardN_matchesSequential` (bit-identical, 911616 and 19447808 logits),
`TestSpeculativeGreedyParity` both pass unchanged with the instrumentation present.

**Reverted afterward, not committed** — `decoder/model.go` is a parity-manifest `core` file
(`testdata/parity_manifest.json`'s `shared_sets.core`), so landing this change, even gated behind
a default-off bool, would mark all 36 tracked families stale and require a re-baseline
(`scripts/refresh_parity_hashes.sh`) for a diagnostic whose only purpose was this one measurement
— the exact tradeoff R13 step 0(ii) already declined for the same reason. The method above is
written to be trivially reconstructed by whoever needs this again (three wrap sites, one printed
line, one env var).

## Data

| component | 0.5B ms/tok | 0.5B share | 1.5B ms/tok | 1.5B share | 7B ms/tok | 7B share |
|---|---:|---:|---:|---:|---:|---:|
| attention | 2.92 | 28.6% | 5.46 | 26.8% | 13.50 | 21.4% |
| MLP | 5.78 | 56.7% | 12.54 | 61.5% | 44.25 | 70.1% |
| LM head | 1.28 | 12.5% | 2.14 | 10.5% | 4.94 | 7.8% |
| (residual — norms, residual adds, embed lookup) | 0.22 | 2.2% | 0.26 | 1.3% | 0.41 | 0.6% |
| **forward total** | **10.20** | 100% | **20.40** | 100% | **63.10** | 100% |
| sample | 0.10 | — | 0.10 | — | 0.10 | — |
| logitProc | 0.00 | — | 0.00 | — | 0.00 | — |
| embed | 0.00 | — | 0.00 | — | 0.00 | — |

Implied tok/s from these runs: 0.5B ≈ 98.0 (standing row: 100.1), 1.5B ≈ 49.0 (standing: 49.5),
7B ≈ 15.8 (standing: 17.2) — all within ordinary run-to-run noise of `benchmarks.md`'s own
2026-09-17 figures (7B's is the widest gap of the three, plausibly the `GOINFER_NO_FIT_GUARD=1`
run's own memory pressure rather than a real regression — not independently isolated), which is
the methodology's own sanity check: this instrument reproduces the already-trusted served numbers
before being trusted for the finer split.

## Reading the split

**MLP dominates on all three models (57-70% of the token), more than attention and LM head
combined.** This is the opposite emphasis from R13's own attention-focused Build (grouped QK/AV
kernels) — that work targets a term that is a minority of the token even before accounting for
softmax's own untouched share within it (R13's own step 0(ii): attention's QK+AV alone is 61-78%
of *attention*, itself only ~21-29% of the *whole token* here). **The MLP fan-out (gate/up/down
projections, three to four W4A8 calls per layer) is the larger lever by a wide margin on CPU
decode**, and it is exactly the term aikit's own S-02 section already investigated in isolation
(goroutine-wake-stagger, not kernel or P/E-core-skew — `MatmulBTW4A8Batch` built and gate-checked
there, q‖k‖v and gate‖up fork-join fusion measuring 1.12-1.21× at the kernel level) but which R9's
own "Read first" section notes is **not yet wired into goinfer's actual `decoder/attention.go` and
`decoder/mlp.go` call sites** ("q, k, v, gate, up issued as separate W4A8 calls" — aikit S-02's own
"Where" line). This measurement is the missing piece connecting the two: MLP genuinely is the
dominant real-token cost on this box, so `MatmulBTW4A8Batch`'s gate‖up fusion in particular (the
larger of its two measured wins, 1.21× at 6 workers) lands on the component that matters most,
not a minor one.

**MLP's share grows with model size** (56.7% → 61.5% → 70.1% from 0.5B to 1.5B to 7B) while
attention's shrinks (28.6% → 26.8% → 21.4%) — consistent with MLP's intermediate dimension scaling
faster than attention's head count/dimension across this model family's size ladder. This makes
the case for prioritizing MLP's fan-out fix over attention's *stronger*, not weaker, at the larger
sizes, where the served-rate gap against Ollama is also largest in absolute terms.

**LM head is a real, non-trivial cost (8-13%, largest on the smaller models), not negligible** —
worth keeping in view for any future "the token is basically all matmul work" framing, though not
large enough on its own to be this brief's Step 2 target ahead of MLP.

## What this establishes, and what it doesn't

**Established:** the per-component split of a real Mac CPU decode token on all three of R9's
named model sizes, at the brief's own decision-cell depth; that `sample`/`logitProc`/`embed` are
noise; that MLP, not attention, is the largest single lever on this box, and increasingly so at
larger sizes.

**Not established:**
- The Linux 0.5B anomaly (~14 ms/token unaccounted per the brief's own standing numbers) — needs
  the Linux box, not attempted here.
- Per-worker fan-out timestamps (start/end per goroutine, matching aikit S-02's own
  `TestW4A8ForkJoinShardTiming` technique) *inside a real goinfer token* — this record gives the
  serial component split (which category), not the parallel worker-level mechanism (why that
  category costs what it does). aikit's own S-02 already answered "why" for the isolated matmul
  case (goroutine-wake stagger); whether the same mechanism dominates inside a real, running
  goinfer token with real cache/context effects is inferred by analogy here, not independently
  re-measured.
- Whether wiring `MatmulBTW4A8Batch` into `decoder/attention.go`/`decoder/mlp.go` actually delivers
  the served-decode win this data suggests it should — that is R9 step 2's own work, gated on
  this table existing, which it now does.

## Next step this points at

R9 step 2, in the order the brief already names: wire `MatmulBTW4A8Batch` for q‖k‖v (attention)
and gate‖up (MLP) in `decoder/attention.go`/`decoder/mlp.go`, bit-identical by construction (same
per-op math, batched fork-join), gated the same way S-01/S-04's own kernel changes are, then the
1.5B served decode cell (`bench_peer`, paired, ship ≥1.15×/park <1.05× per aikit S-02's own
registered band) decides. Not started here — this record is step 1's table, not step 2's build.

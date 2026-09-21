# R9 step 1 — CPU decode per-component attribution (Mac half)

`docs/tasks/red-october.md` R9, step 1: "attribute the per-token cost the fits in §2.1 say
exists... Output: a per-component ms/token table per (box, size)." **Mac half only** — no access
to the Linux box in this session, so the Linux 0.5B anomaly (§2.1's own open question) is
untouched here.

Box: `apple-m1pro` (M1 Pro, 6P+2E, 16 GB). goinfer `0c6b8d23`. Models: `qwen2.5-coder-0.5b` and
`qwen2.5-coder-1.5b` (int4, Backend:"cpu"). Greedy, prompt 128 tokens (R9's own decision-cell
depth), 24 decode steps. `qwen2.5-7b` **not measured**: the fit guard declined it both times
attempted (needs ~16.7 GB — 8.9 resident + 3.5 KV + 4.4 transient checkpoint-read overhead —
against 6.4-7.1 GB available at the time), and it was not bypassed given this session's own
recent, repeated near-incidents with large loads on this exact machine (see
`metal-moe-autopager-m26-2026-09-20.md`) — a 16.7 GB ask on a 16 GB machine is not comparable in
risk to those, but wasn't worth pushing today regardless.

## Method

Two layers, both reusing existing or newly-added machinery rather than a from-scratch build:

**Coarse split (forward / sample / logitProc / embed):** `GOINFER_DECODE_TIMING=1`, an
already-shipped diagnostic (`decoder/model.go`'s `decodeTiming` var, predates this session) —
not something this task needed to build. Confirms the obvious but worth stating plainly: `sample`,
`logitProc` and `embed` are all ≤0.10 ms/token on both models, i.e. noise next to `forward`
(10.0-10.2 ms on 0.5B, 20.4-20.8 ms on 1.5B) — essentially the entire decode-token cost is inside
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

| component | 0.5B ms/token | 0.5B share | 1.5B ms/token | 1.5B share |
|---|---:|---:|---:|---:|
| attention | 2.92 | 28.6% | 5.46 | 26.8% |
| MLP | 5.78 | 56.7% | 12.54 | 61.5% |
| LM head | 1.28 | 12.5% | 2.14 | 10.5% |
| (residual — norms, residual adds, embed lookup) | 0.22 | 2.2% | 0.26 | 1.3% |
| **forward total** | **10.20** | 100% | **20.40** | 100% |
| sample | 0.10 | — | 0.10 | — |
| logitProc | 0.00 | — | 0.00 | — |
| embed | 0.00 | — | 0.00 | — |

Implied tok/s from these runs: 0.5B ≈ 98.0 (standing row: 100.1), 1.5B ≈ 49.0 (standing: 49.5) —
both within ordinary run-to-run noise of `benchmarks.md`'s own 2026-09-17 figures, which is the
methodology's own sanity check: this instrument reproduces the already-trusted served numbers
before being trusted for the finer split.

## Reading the split

**MLP dominates on both models (57-62% of the token), more than attention and LM head
combined.** This is the opposite emphasis from R13's own attention-focused Build (grouped QK/AV
kernels) — that work targets a term that is a minority of the token even before accounting for
softmax's own untouched share within it (R13's own step 0(ii): attention's QK+AV alone is 61-78%
of *attention*, itself only ~27-29% of the *whole token* here). **The MLP fan-out (gate/up/down
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

**LM head is a real, non-trivial cost (10-13%), not negligible** — worth keeping in view for any
future "the token is basically all matmul work" framing, though not large enough on its own to be
this brief's Step 2 target ahead of MLP.

## What this establishes, and what it doesn't

**Established:** the per-component split of a real Mac CPU decode token on two model sizes, at
the brief's own decision-cell depth; that `sample`/`logitProc`/`embed` are noise; that MLP, not
attention, is the largest single lever on this box.

**Not established:**
- The Linux 0.5B anomaly (~14 ms/token unaccounted per the brief's own standing numbers) — needs
  the Linux box, not attempted here.
- 7B on the Mac — declined by the fit guard both times, not forced.
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

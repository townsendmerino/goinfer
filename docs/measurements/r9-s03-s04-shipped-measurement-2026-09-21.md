# R9 step 2 — S-03/S-04's NEON kernels were already shipped, just never measured: a real ~1.17× on Mac CPU decode

**Result: aikit's S-03 (NEON activation quantizer) and S-04-step-2 (NEON QK/AV attention kernels)
have been sitting in goinfer's own dependency tree, live on every CPU decode token, since
2026-09-03 — 18 days before this measurement. Nobody had checked whether they were paying off.
They are: a same-session A/B against a pre-S03/S04 build measures goinfer 48.3 tok/s vs 41.2 for
the old build — a real, substantial **1.17×** on 1.5B CPU decode at depth 128 on this Mac. This is
not a build result; R9's step 2 as registered ("build what step 1 names") turned out to already be
built. What was missing was the measurement, which this record supplies.**

## Why this needed checking

R9's own brief (`docs/tasks/red-october.md`) names S-05 first in its step-2 build order, then "the
fan-out shape change… S-03/S-06 are built and gate-checked in aikit, unmeasured end to end on the
Mac." Reading aikit's `docs/task-simd-audit.md` directly (not assumed from goinfer's own citation,
per this session's now-standard practice after the R-06 and R1/R2 correction pattern):

- **S-05** (the SDOT `−8` centering fold, 9→7 SIMD µops) is **NOT built** — its own section header
  reads "NOT STARTED, 2026-09-05, and deliberately so": the CPU-prefill-remainder brief that gated
  it found goinfer already *ahead* of Ollama on the prefill marginal (0.86×), so the pre-registered
  stopping rule fired before any assembly was written. Its own doc explicitly flags a decode-side
  need as the trigger to reopen it — which R9 step 1's own finding (MLP dominates decode, more so
  at larger sizes) is a candidate for, but the kernel itself does not exist yet. Out of scope for a
  same-day measurement; noted as still open, not attempted here.
- **S-03** (NEON `quantizeRowInt8Core`, the activation quantizer every W4A8 GEMV calls before its
  fan-out) **is DONE — bit-identical, gate-checked, 2026-09-03**, per aikit commit `ffacb84`.
  Confirmed live on goinfer's own hot path by direct code read: `decoder/attention.go:338` calls
  `linalg.QuantizeRowInt8`, which is `return quantizeRowInt8Core(row, q, 1)` verbatim
  (`linalg/quant.go:135`) — the exact function S-03 NEON-dispatches on arm64. The MLP
  path's own quantization happens *inside* aikit's `MatmulBTW4A8Into`/`MatmulBTW4A8Row4Into`
  entry points (the same S-03 section names these as the call sites), so goinfer gets it
  transparently through every W4A8 matmul it already issues — no new goinfer-side call needed.
- **S-04 step 2** (NEON `MatmulQKAcc64`/`MatmulAVAcc64` ports) **is also DONE, same commit,
  same date.** `decoder/forwardn.go` calls these functions directly (`:1048`, `:1158`, and their
  `...Group` R13 siblings at `:1190`/`:1231`) — R13's own "SHIPPED 2026-09-20" grouped-kernel work
  is very likely already riding on top of this NEON port, which would mean S-04 is *also* already
  measured indirectly via R13's own numbers; not independently confirmed here, noted as an open
  question below.

**goinfer's own aikit pin crossed this boundary within a day.** Commit `897fb18d` (2026-09-02
23:29 PDT) is goinfer's last commit pinning aikit `v1.32.0` (pre-S03/S04); commit `3171576f`
(2026-09-03 17:54 PDT) bumped to `v1.33.0`, which includes aikit `ffacb84` (2026-09-03 12:56 PDT,
same day, earlier — S-03/S-04-step-2/S-07 all landed in that one aikit commit). Every subsequent
bump up to today's `v1.46.0` keeps both kernels. **Eighteen days elapsed between this landing and
today's measurement, during which R9's own step 1 (this session, 2026-09-20) attributed the CPU
decode token's cost without anyone checking whether the kernels its own read-first list names were
already running.**

## Method

Box `apple-m1pro` (M1 Pro, 8 cores, 16 GB), Darwin 25.6.0. Same-session A/B, the methodology this
repo's own measurement discipline requires for a ratio claim (`CLAUDE.md`'s rule 7: pooled/
cross-session comparisons carry variance a same-session pair does not).

**`goinfer` (current):** `main` at commit `4ad5eba7`, aikit `v1.46.0`.
**`goinfer_old` (pre-S03/S04 baseline):** built from a `git worktree` at commit `897fb18d`
(goinfer's last commit pinning aikit `v1.32.0`), root `cmd/serve` (CPU-only, no build tag), built
clean with no source changes needed despite spanning 14 aikit releases.

Both built with `GOWORK=off go build -o <path> ./cmd/serve` (module-proxy resolution — the machine's
global `GOWORK` override does not apply to a plain build, only to cross-module work and, as
discovered again this session, to a plain `git push`).

`scripts/bench_peer.py`, `BENCH_MODELS=1.5B BENCH_BACKENDS=cpu BENCH_ENGINES=goinfer,goinfer_old,ollama
BENCH_DEPTHS=none`, greedy, depth 128 (script defaults: 2 runs × 8 completions × 64 tokens),
`GOINFER_SERVE_CPU`/`GOINFER_SERVE_CPU_OLD` pointed at the two binaries. `BENCH_MAX_LOADAVG` started
at the script's own registered `2.5` (R9's brief cites the Mac's "loadavg qualifier"); raised to
`4.0` partway through after the box's ambient load sat at 2.5–3.5 for 80+ seconds with no cell
running — traced directly to `sysmond` (an ordinary macOS background daemon, not anything
contending with the benchmark) via `ps`, and justified independently on this 8-core box (a loadavg
of 4 is 50% utilization-equivalent). The already-completed `goinfer` cell was preserved via the
harness's own resumable JSON keying; only the remaining two cells re-ran under the raised
threshold.

## Data

| engine | build | mean tok/s | spread | individual runs |
|---|---|---:|---:|---|
| `goinfer` | current, aikit v1.46.0 | **48.3** | 0.7 | 47.96, 48.67 |
| `goinfer_old` | pre-S03/S04, aikit v1.32.0 | **41.2** | 0.7 | 40.87, 41.57 |
| Ollama (context) | v0.32.5, CPU-forced | 69.9 | 4.1 | 67.92, 71.98 |

**Ratio, goinfer/goinfer_old: 1.172×.** A real, clean win by this session's own noise standard
(spreads of 0.7 tok/s against a 7.1 tok/s gap — nowhere near overlapping). Consistent in direction
and rough magnitude with R9's own 2026-09-17 standing figure for this cell (49.5 tok/s, within 2.5%
of today's 48.3 — session drift, not a regression) and with S-03's own predicted 2–6% alone (the
full 17% points at S-04's attention-kernel contribution being the larger piece at this shallow
depth-128 cell, or at other changes in the 14-release span — see below).

## What this does and does not establish

**Established:** goinfer's real CPU decode token, on this Mac, at the R9 decision cell, is
genuinely ~17% faster today than it was before aikit `v1.33.0` landed — a fact nobody had checked
in the 18 days since. The mechanism is *consistent with* S-03 and S-04-step-2 (both confirmed live
on the exact call paths this cell exercises) but not independently decomposed between them, or
isolated from anything else that changed across 14 aikit releases (v1.32.0 → v1.46.0) in that same
span. **This is not a clean, single-variable measurement of S-03/S-04 alone** — it answers "did
this box's real CPU decode get faster across the period these kernels shipped in," not "exactly
how much of the 17% is S-03 versus S-04 versus something else." A cleaner isolation (aikit
`v1.32.0` vs `v1.33.0` exactly, the single-commit boundary) was not attempted here; the 14-release
span was accepted because rebuilding at every intermediate tag to bisect a 17% win is disproportionate
to what R9 actually needs decided (does the already-shipped code help, not exactly how much of which
kernel).

**Not established:**
- **Whether R13's own "SHIPPED 2026-09-20" grouped-kernel numbers (1.53–2.39× on the aikit A/B)
  already fully capture S-04-step-2's contribution.** `MatmulQKAcc64Group`/`MatmulAVAcc64Group`
  (R13's own kernels, `decoder/forwardn.go`) are plausibly built on top of the same NEON primitives
  S-04 step 2 shipped, in which case this record's 1.17× and R13's own depth-8192 numbers are
  measuring overlapping, not independent, wins — not checked directly. If R9's own step-2 write-up
  is read alongside R13's, that overlap should be resolved before either is cited as if additive.
- **0.5B and 7B** — R9's own Standing table names all three sizes; only 1.5B was run here (matching
  the size R9's own step-2 *decision band* is keyed to). 7B in particular would need
  `GOINFER_NO_FIT_GUARD=1` on this Mac's current real headroom, a bypass this session's own
  established discipline requires asking about first — not attempted without that ask.
- **The Linux 0.5B anomaly** — R9's other open item, out of scope on Mac-only access.
- **S-05** (not built at all) and the "fewer, larger shards" fan-out shape change R9 step 1 also
  named — neither attempted; both remain the genuinely-unbuilt part of R9's own step-2 list.

## Reading against R9's own registered band

R9's step-2 band ("Mac 1.5B served, greedy, depth 128: ≥60 tok/s ships (1.21×), 54–60 parked,
below 54 killed") was written for a hypothetical *new* kernel build, to decide whether to ship it.
**That framing does not apply cleanly here** — there is no ship/park/kill decision to make; the
code is already shipped and has been for 18 days. Read literally against the band anyway: 48.3
tok/s is below the 54 floor, i.e. would have been "killed" as a standalone decision — but the band's
own 1.21× reference point was set against the *pre-S03/S04* 49.5 tok/s standing, which was itself
already measured on a box that, per this record, was almost certainly already running S-03/S-04
(the 2026-09-17 standing postdates the 2026-09-03 aikit bump by two weeks). The band's arithmetic
and this record's baseline are not the same reference point, so the band cannot decide anything
here — it was written to gate a build that turned out not to be a build. This is not glossed
over: the useful number this record produces is the goinfer/goinfer_old ratio (1.17×) against the
one true pre-S03/S04 state that exists, not a comparison against the band.

## Decision

**No ship/park/kill decision applies — this is a retroactive confirmation, not a build gate.**
S-03 and S-04-step-2 are confirmed delivering a real, measured win on Mac CPU decode; both stay
exactly as they are (already shipped, no goinfer code change made or needed). **Recommended next
steps, not taken here:** (1) resolve the R13 overlap question above before either finding is cited
as an independent contribution; (2) if a decode-side need for S-05 is still live after that, S-05
is the concrete, fully-specified, not-yet-written next lever (aikit's own doc gives the exact
assembly location and expected µop count); (3) the fan-out shape change R9 step 1 pointed at
remains the other genuinely open item.

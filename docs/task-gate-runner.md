# Task: one Go gate-runner over `go test -json` — collapse the tallying shell + census Python

> **Status: IN PROGRESS — FIVE of the six are migrated (2026-08-20/21).** `cmd/gate` carries
> `census`, `heavy`, `parity`, `composition` and `selector`; `scripts/skip_census.py`,
> `heavy_gate.sh`, `parity_sweep.sh`, `sweep_composition.py` and `selector_coverage.py` are deleted.
> **`gpu_gate.sh` is the one that remains** (it needs the GPU box and derives its check set from
> `ci_checks.py`). See §8 for what each migration found.
>
> Tracked as QUEUE **E8**. Sibling of E7 (no-Python) but a distinct idea: E7 migrates scripts
> one-for-one; **E8 recognizes that several scripts are one program** and consolidates them.
> Drafted 2026-08-12 from the E7 inventory + the item-6 shell audit.

## 1. The insight — six scripts are one program

Three tallying **shell** gates and three census **Python** scripts do the same thing: run
`go test -json` across a matrix of *package × family × quant × build-tag*, consume the structured
event stream, tally PASS/SKIP/FAIL (SKIPs bucketed by reason), and apply a decision. They differ only
in *which matrix* and *which decision*:

| today | kind | what it decides |
|---|---|---|
| `scripts/parity_sweep.sh` | shell gate | T3 parity sweep across families/quant/loader — pass/fail tally |
| `scripts/gpu_gate.sh` | shell gate | GPU-backend check-set (derives from `ci_checks`), backend-detected |
| `scripts/heavy_gate.sh` | shell gate | heavy/model-loading tests — tally, skip-with-reason when assets absent |
| `scripts/skip_census.py` | census | PASS/SKIP/FAIL census, SKIPs bucketed by reason |
| `scripts/sweep_composition.py` | census | parity-sweep coverage composition along family×quant×loader |
| `scripts/selector_coverage.py` | census | tests that EXIST vs tests a selector RUNS |

**One runner** — `cmd/gate` (or a `tools/`-module binary) — parameterized by a committed matrix
config, subsumes the deciding-and-tallying core of all six. Not 9→8; roughly **6 scripts → 1 runner +
configs.** (E7 still owns the other stdlib Python — `queue_citation_lint`, `bench_peer`, the
opportunistic tail — those are not this shape.)

## 2. Why Go here is *strictly better*, not just same-language

This consolidation dissolves the exact footgun classes the item-6 audit flagged — by construction, not
by discipline:

- **The `-e`/tally tension disappears.** The gates omit `set -e` deliberately (run N families, tally;
  `-e` would abort on the first failure and lose the count) — a discipline you must remember not to
  "fix." Go has no implicit abort-on-error: run each cell, capture `rc`, append to a results struct,
  never lose the count. The requirement becomes the natural shape of the code.
- **`PIPESTATUS` capture vanishes.** `os/exec` returns each subprocess's exit code directly.
- **The silent-skip anti-pattern can't recur.** `command -v staticcheck && staticcheck` (the one
  `scripts/mutation_check.sh`'s header records fixing) quietly *passes* when the tool is absent; `exec.LookPath`
  returning not-found is an explicit error. Same for backend detection — "GPU absent → SKIP **with
  reason**" is a decision that belongs in Go, and the reason-bucketing `skip_census` does comes free
  because the runner already holds structured results.

Same theme as `skip_census` becoming "a reader, not a parser": the whole tallying layer gets safer
because it stops scraping text and starts consuming `go test -json` events.

## 3. What stays shell — deliberately

The line QUEUE already draws: **decides-things → Go; orchestrates-one-command → shell.** Pure glue
stays shell; wrapping it in a Go binary adds a build artifact for zero benefit:

- `cuda/build_ptx.sh` — sets arch, calls `ptxgen`. Thin. Stays. (Its `ptxgen` payload is E7's nvrtc
  item; the shell wrapper is fine.)
- The two demo asset-build scripts under `demo/chat/`, `demo/agent/`. Stay.
- Any env/PATH/`GOINFER_*` export-then-run-one-command wrapper. Stays.

The runner **shells out to** `go test -json` via stdlib `os/exec` — it orchestrates `go test`, it does
not reimplement it. No new dependency; nothing in the main `go.mod`.

## 4. Architecture

- **Location:** `cmd/gate` in the relevant module, or a `tools/` module with its own `go.mod` if any
  non-stdlib parser is ever needed (it should not be — `go test -json` is stdlib `encoding/json`).
  Per E7's constraint, **a consumer's module graph must not grow because a gate changed language.**
- **Matrix config:** a committed Go file or small JSON — `{package, families[], quant[], tags[],
  requires: [gpu|heavy|none], decision: sweep|checkset|census}`. Each shell gate + census script
  becomes one config, not one program.
- **Core loop:** for each cell, `exec` `go test -json -run … -tags …` with the cell's env; stream-parse
  the `TestEvent` JSON; classify PASS/SKIP/FAIL and bucket SKIP reasons; never abort the loop on a
  failing cell. Emit both the machine tally and the human census (the scope line — see §5).
- **Hardware detection in Go:** GPU presence via `exec.LookPath("nvidia-smi")` or a gocudrv probe;
  absence → SKIP-with-reason, not a silent pass. Heavy-asset presence → same. This is the
  `gpu_gate`/`heavy_gate` backend-detection logic, moved to where it can't fail open.

## 5. Acceptance (E7's a–d, verbatim — they were written for this)

- **a. Agree before swap.** The Go runner and the shell gate / Python census must produce the **same
  verdict and the same tally** on the current tree. Any disagreement is investigated before the swap.
- **b. Mutation-check both ways.** Introduce a failure the gate exists to catch → runner goes RED;
  remove it → GREEN. Include the tally-integrity case: a mid-matrix failure must be *counted*, not
  lost (the property `-e` would have broken).
- **c. Delete the shell/Python in the same commit** that lands its config — never two sources of truth.
- **d. The scope line survives.** Whatever the script printed about *what it validated and what it
  skipped and why*, the runner prints too. Losing the skip-reason census is losing the point.

## 6. Sequencing

- **After v0.13.0**, behind §C1 + the CUDA gate — same freeze as E7.
- **Order within E8:** build the runner + the `heavy_gate`/census cell first (no hardware needed to
  develop the tally logic against a local package), prove agreement, then `parity_sweep`, then
  `gpu_gate` (needs the GPU box). The three census Python scripts fold in as configs as the matching
  gate lands — so E8 and E7's census migrations converge rather than duplicate.
- **Do not harden shell you are about to delete.** The item-6 audit's `pipefail` fixes for
  `scripts/bench_compare.sh` and `scripts/mutation_check.sh` are **moot** if those are on the migration list
  (`bench_compare` → the bench-peer Go successor; `mutation_check`'s deciding half → Go/this runner).
  Add `pipefail` only to the survivors — the pure-glue shells in §3. Fix the survivors, not the
  condemned.

## 7. Not in scope

- E7's non-matrix Python (`queue_citation_lint`, `bench_peer`, the opportunistic tail) — different
  shape, stays in E7.
- The nvrtc build helper (E7 build-tooling item) and the `oracle/` reference-forward plan
  (`docs/task-oracle-refforward.md`) — independent; share nothing with this.
- Any change to what the gates *decide* — E8 changes the *substrate* (shell → Go over `go test -json`),
  not the pass/fail criteria. A verdict change would violate acceptance (a).

## 8. What has landed, and what building it found

**Landed 2026-08-20:** `cmd/gate` — the event core (`event.go`), the cell runner (`run.go`), the
skip bucketing (`skips.go`), the two report shapes (`report.go`) and the committed matrix configs
(`configs.go`) — plus `gate census` and `gate heavy`, replacing `scripts/skip_census.py` and
`scripts/heavy_gate.sh`, deleted in the same commit (acceptance c). Stdlib only; no module's
graph grew.

**Acceptance (a) — agreement, and how it was made exact.** The census comparison does not run
`go test` twice. One `go test -json -tags goinfer_testhooks ./...` stream was captured to a file
and BOTH programs were pointed at the same bytes (`skip_census.py FILE` / `gate census -stream
FILE`), so any difference is a difference between the two programs rather than between two runs of
a suite with asset-gated skips in it. Result: **byte-identical output and identical exit code in
both modes** — PASS 884 / FAIL 0 / SKIP 79 of 963, buckets `missing-fixture 2 · heavy-model 53 ·
integration-env 21 · other 3`, rc 0; and with `GOINFER_REQUIRE_FIXTURES=1`, rc 1 with the same two
fixture skips named. The heavy tier was compared on THREE cells, chosen so that
agreement could not be free: a real-checkpoint filter (granite / nemotron / mellum — **PASSED 11,
GREEN, rc 0 from both**, per-cell counts identical), a second real filter (**4, GREEN**), and an
all-skip filter (**PASSED 0 / SKIPPED 5, RED, rc 1 from both**, same five tests) which is the only
one that exercises the skip list and the zero-pass rule.

**Three things the migration found, none of them cosmetic:**

1. **The two scripts disagreed about subtests, and neither said so.** `heavy_gate.sh` counted
   `grep -cE '^--- PASS: '`, anchored at column 0 — top-level tests only, since `go test` indents
   subtest result lines. `skip_census.py` keyed on `(Package, Test)` straight out of the JSON,
   which counts every subtest as its own result. Both are defensible and they are not the same
   number. Preserved as a per-config knob (`TopLevelOnly`), because acceptance (a) is per-gate:
   each migrated gate reproduces ITS OWN tally, not a newly-imposed house style.
2. **Keying results on `(Package, Test)` silently undercounts a matrix.** Caught by the
   tally-integrity mutation test, not by review: two cells running the same package — which is
   exactly what `parity_sweep` and `gpu_gate` do across tag/quant combinations — collapsed into
   one key, so N runs of a test reported as one. The key carries the cell now. The bug never
   reached the census (one cell) but would have landed squarely in the two gates still to migrate.
3. **The census exits 0 on a package-level failure.** `skip_census.py` computes `rc = 1 if nfail
   else 0` and prints package-level fails without counting them — so a build error or a native
   crash in one package exits GREEN today. E8 changes the substrate, not what a gate decides
   (acceptance a and §7), so the runner reproduces it exactly — behind a `PkgFailIsFailure` bool,
   and it now PRINTS that it is suppressing one. A suppressed failure nobody is told about is
   indistinguishable from no failure, which is the defect the census exists to prevent.
   **Flipping that bool is a verdict change and therefore Francis's call, not a refactor.**

**Acceptance (b) — mutation-checked both ways, in-tree** (`cmd/gate/gate_test.go`, against the real
toolchain over scratch modules): census RED on a failing test and GREEN with the defect removed;
the tally-integrity case (a failure in cell 1 must not stop cell 2, and both counts survive into
the verdict — the property `set -e` would have destroyed); zero-pass is RED; a package that exits
hard without a per-test failure is RED and named as HIDDEN; an empty stream is RED, not clean; the
missing models dir REFUSES with exit 2 rather than reporting a verdict.

4. **`heavy_gate.sh`'s skip list printed durations, not reasons.** Its header promised "every skip
   is listed with its reason so a missing model can't masquerade as coverage", and its verdict
   printed `TestQwen3Embedding_vectorParity (0.00s)` — the `sed` stripped the `--- SKIP: ` prefix
   from the result line, and `t.Skipf`'s message is on the NEXT line, which it never captured. So
   the gate reported WHICH tests skipped but never WHY, which is the half that distinguishes "the
   checkpoint is absent" from "the gate is misconfigured". The runner prints package, test and
   reason, because it holds the structured result rather than a grep of it. This is the one place
   the migration is not behaviour-preserving, and it is strictly in acceptance (d)'s direction.

5. **Deleting a script found a stale second copy of a generated index.** `docs/QUEUE.md` carried
   TWO SHA indexes: the live `CITATION-INDEX` block, and an orphaned `SHA-INDEX` block whose
   regenerate instruction named `scripts/queue_sha_lint.py` — a script that no longer exists.
   `--update` leaves it untouched (verified), it held 35 rows against the live 40, and its
   `8fecfad` row's subject had DRIFTED from the real one (`ci: scripts/heavy_gate.sh …` vs
   `ci: heavy_gate.sh …`) — which is how a second source of truth announces itself. It was also
   inflating the lint's own coverage number, since its index rows were being scanned as prose and
   counted as citations (40 "validated commit citations" became 16 when it went). Removed rather
   than allowlisted: the allowlist is for a reference that cannot be made to resolve without
   falsifying the text around it, and this one could.

**Acceptance (d) — the scope line survives**, and gained provenance: every gate now prints commit,
dirty flag, UTC date, host and its own matrix before the verdict, which only `heavy_gate.sh` did.

## 9. Step 2 — the parity sweep, and the two censuses that had to come with it

**Landed 2026-08-21:** `gate parity`, `gate composition`, `gate selector`, replacing
`scripts/parity_sweep.sh`, `scripts/sweep_composition.py` and `scripts/selector_coverage.py`, all
deleted in the same commit.

**The two censuses were NOT optional here, and §6's "fold in as the matching gate lands" understated
it.** Both of them *parsed the shell script*: `sweep_composition.py` regexped the `GATES=(…)` array
out of `parity_sweep.sh`, and `selector_coverage.py` did the same to derive the sweep half of its
selector. Deleting the sweep would have broken both outright — so this was a hard dependency, not a
convergence. Both now read the same Go gate list the sweep checks, which removes the second copy
rather than reimplementing the parse.

**The sweep is a CHECKSET, not a tally, and that is why it needed its own decision.** `gate heavy`
asks "did anything fail?"; the sweep asks, of each NAMED gate, "is this one green?" — which makes
**MISSING** (the gate produced no result at all: a renamed test, a `-run` filter that stopped
matching, a package that failed to build) a distinct and worse outcome than a failure, because a
pass/skip/fail tally cannot see it. Four outcomes, three of them blocking, plus two deliberate
non-blocking classifications preserved exactly: a FAIL the ledger calls FIRST-RUN is an ITEM (it
asserts a delta with no second point to compute it from), and a SKIP in `assetNeverBuilt` is a
COVERAGE GAP (no invocation on any machine could clear it, and a permanent blocker is not a gate —
it is an override habit).

**Acceptance (a): both sides run, EMIT_MANIFEST=1, with the mutated files reverted between them** so
each started from the same committed baseline and its diff was attributable. Identical verdict
(**2 BLOCKER(S), rc 1**), all **50 per-gate rows identical**, emitter-coverage section identical,
and the manifest+matrix mutation **byte-identical (11 028 bytes both sides)**. The composition
census agrees byte-for-byte in both its plain and `-v` forms; the selector census agrees
byte-for-byte. Two intended differences, both away from naming a deleted file or from a shell
artefact: the cross-gate label reads `gate parity` rather than `parity_sweep.sh`, and the catch-all
no longer emits the trailing space `sed -E 's/\(.*//'` left behind.

**What the run found — a real red that is NOT E8's:** `TestQwen35GGUF_gate` and
`TestQwen35GGUF_weightDiff` fail on this tree, identically under the unmodified shell script, so the
migration neither caused nor masks them. The gate measures **argmax 68/80 (85.0%), min cosine
0.98740 against a 0.992 floor**, where the value recorded in the test's own comment for this box is
**0.99298**. The ledger classifies both as *confirmed before this gate last changed*. Owner's call;
recorded here because a migration that quietly absorbed a red would be the worst possible outcome
for a gate whose entire job is to refuse one.

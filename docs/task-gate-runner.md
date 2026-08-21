# Task: one Go gate-runner over `go test -json` — collapse the tallying shell + census Python

> **Status: ALL SIX MIGRATED, PLUS `mutation_check` (2026-08-20/21).** `cmd/gate` carries `census`,
> `heavy`, `parity`, `composition`, `selector`, `gpu` and `mutation`; all seven scripts are deleted. **The Metal half of `gate gpu` is
> ported but NOT yet verified against the script it replaces** — no machine here has a Metal device,
> so its four groups need one run on the Mac before that half is trustworthy. See §8–§10 for what
> each migration found.
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
- **The silent-skip anti-pattern can't recur.** `command -v staticcheck && staticcheck` (the one the
  mutation checker's header records fixing — now `cmd/gate/mutation.go`) quietly *passes* when the tool is absent; `exec.LookPath`
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
  `scripts/bench_compare.sh` and `scripts/mutation_check.sh` were **moot** because both were on the
  migration list. `mutation_check`'s deciding half is now `gate mutation` (§11) and the script is
  gone, so its `pipefail` finding died with it — vindicating the rule. `bench_compare` still awaits
  the bench-peer Go successor. Add `pipefail` only to the survivors — the pure-glue shells in §3.

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

**What the run found — a real red that is NOT E8's, now BISECTED to one commit:**
`TestQwen35GGUF_gate` and `TestQwen35GGUF_weightDiff` fail on this tree, identically under the
unmodified shell script, so the migration neither caused nor masks them.

**Culprit: `6d4fc79` "qwen35 family: quantize the projections that were f32 at every quant".**
Adjacent pair, same box, same assets, same toolchain on both sides:

| | argmax | min cosine | mean | worst gap | verdict |
|---|---|---|---|---|---|
| `33879dd` (parent) | 69/80 (86.2%) | **0.99298** | 0.99846 | 0.0039 | PASS |
| `6d4fc79` | 68/80 (85.0%) | **0.98740** | 0.99608 | 0.0080 | FAIL (floor 0.992) |

**A hypothesis worth recording because it was WRONG:** the Go 1.27 bump looked like the obvious
suspect — the right size of effect, the right era, and this repo has a documented arm64/amd64 FMA
divergence. It is refuted: the gate produces the **bit-identical** 0.98740 at `go 1.26.6`
(`357b7db`) and at `go 1.27.0` (`d7c41c4`). Two probes, ~30 minutes, and the plausible story died.

**This is not an unnoticed regression — it is a deliberate trade whose blast radius was measured on
the wrong gate.** `6d4fc79` made the DeltaNet and attention projections honour `Options.Quant`
instead of staying f32 (1.60× decode, 7.4× TTFT), and its message records the accuracy cost in those
words — but for `TestQwen35Real_gate2FullModel` (int8 vs bf16, floor ≥0.98, 0.99333 → 0.99069, still
green), whose manifest entry it re-baked. Its SIBLING, this gate, carries a hard-coded **0.992**
floor set at `2583a2b` and never revisited, and the same trade moved it **0.99298 → 0.98740**,
crossing that floor by 0.0046. Coherent prompts also went 8/10 → 7/10.

**Why it shipped silently:** the gate is `//go:build realckpt` and heavy, so CI never builds it and
only the release sweep runs it. The trade landed between sweeps.

**The remedy is a judgement call and is NOT taken here.** Re-baselining the floor to match the
deliberate trade is the cheapest option and is consistent with what the commit already did for the
sibling gate — but `6d4fc79`'s own stated standard was "evidence the new path is RIGHT, not merely
different, because re-baking a golden on *it changed* is how a regression gets blessed", and that
evidence does not yet exist for the Q8_0 path.

## 10. Step 3 — the GPU gate, the last of the six

**Landed 2026-08-21:** `gate gpu`, replacing `scripts/gpu_gate.sh` (715 lines, the largest of the
six), deleted in the same commit. E8 is structurally complete: 6 scripts → 1 runner + configs.

**Acceptance (a), CUDA half, both sides running the FULL gate including the ~28-minute heavy tier,
sequentially so the card was never shared** (a stray process on the GPU is the one thing this gate's
own header says invalidates every memory-sensitive result below it):

| | shell | Go |
|---|---|---|
| check groups | 9 declared → 9 reported | **identical** |
| verdicts | 9 pass, 1 skip, 2 fail | **identical** |
| the 12 per-check verdict lines | — | **identical** |
| which tests failed | `TestQwen36_35B_cache`, `TestMoERouteDemandThreshold` | **identical** |
| partition reconciliation | marked=4 skipped-in-main=4 ran-in-drain=4 | **identical** |
| final verdict | FAIL, rc 1 | **identical** |
| skip notes block | 4 entries | **identical** |

Only two lines differ, and one of them was a defect I introduced and had to fix:

1. **Heavy-tier wall clock, 2183s vs 2121s.** Inherent; the gate times a 28-minute tier.
2. **"ran 238 tests" vs "ran 176".** THE SHELL WAS RIGHT AND THE FIRST GO VERSION WAS WRONG, for a
   reason worth writing down: `go test -v` does **not** indent a subtest's `=== RUN` line (only its
   `--- PASS` result line is indented), so `grep -cE '^=== RUN'` counts subtests, while
   `grep -cE '^--- SKIP'` beside it does not. Reproduced exactly, and verified by re-parsing the
   saved `-json` stream rather than re-running the 35-minute gate: 238 `=== RUN` lines, 22 top-level
   skips. **Which means the line "ran 238 tests, skipped 22" mixes units** — tests-and-subtests
   started against top-level skips. Preserved because E8 changes the substrate and not what a gate
   reports; flagged here because a number whose unit is ambiguous is the denominator problem in
   miniature, and it should probably say which it is.

**What the run found — two reds, neither E8's, and both reported identically by the script:**

- **`TestQwen36_35B_cache`** — `serialized weights: format version 2, this build reads 3..6`. The
  35B `.giw` on this box predated a format bump. **FIXED 2026-08-21:** regenerated from the Q8_0
  GGUF via `cmd/prequant -quant int4` (2m25s, 21 120 MB, peak RSS 43 GB, weights blob now GINFW v6).
  It is 26.4 GB → 22.1 GB smaller than the bundle it replaces, which is `6d4fc79` again — the
  projections that used to be stored f32 are now int4 in the bundle too. Regenerating then exposed a
  SECOND defect behind it: the test's tokenizer step handled a directory and a `.gguf` but not a
  `.giw`, which is its own DEFAULT path, so the default invocation could never reach the decode it
  exists to measure (it died as `parse …int4.giw: invalid character 'G'` — the bundle magic read as
  JSON). Every passing run had set `GOINFER_QWEN36_35B` to the `.gguf`. Both fixed; the test now
  passes at 14.09 tok/s, 75.5% expert-cache hit rate, naming Paris and the Eiffel Tower.
- **`TestMoERouteDemandThreshold`** — the demand identity is broken in the COLD regime: measured
  192 675 840 B against an expected 289 603 584..295 895 040 B. An older drain log on this box shows
  the same test failing with *different* numbers (threshold 141 557 760 B, peak/residual 1.02× where
  this run measured 1.39×), so the quantity it pins is moving between runs. The test's own message is
  explicit about what that means: `docs/QUEUE.md` A1/A5/A7/A9 need **re-deriving, not editing**.
  Note the shape — this is a gate that MEASURES THE DEVICE, so it is not byte-deterministic, and
  agreement for it can only be claimed at the verdict and group level, which is what the table above
  reports.

**Acceptance (b)** (`cmd/gate/gpu_test.go`): group reconciliation in both directions (a declared
group that emits nothing is a FAIL; an emitted-but-undeclared group is a FAIL); all four verdict
states, including that a dirty tree with every check green is INCONCLUSIVE and **not** PASS, and that
zero checks run is NO GATE; zero-matched-tests is detectable and subtests do not inflate it; the drain
group derives from its marker and is a set; `detail()` falls back to the raw tail for a failure with
no assertion line; every skip reaches the notes block; the PTX version comes from the artifact, never
from the box's default; and `vramNote` states the reading **without naming a mechanism** — asserted
negatively, because the obvious mechanism is disproven and naming one a gate cannot see is how the
last three explanations became someone's wasted afternoon.

**One dependency died with the script:** the Mac needed Homebrew bash to run this gate at all, because
macOS's stock `/bin/bash` 3.2 cannot run `declare -A`. A Go binary has no such requirement.

**Still owed: the Metal half.** Its four groups (suite, cgo-free, lifecycle, prefill) are ported from
the same source but have never run — this box has no Metal device. Until someone runs
`go run ./cmd/gate gpu` on the Mac and compares it against the deleted script's last known output,
that half is code review, not evidence.

## 11. `gate mutation` — the gate's own gate

**Landed 2026-08-21:** `gate mutation`, replacing `scripts/mutation_check.sh` (99 lines), deleted in
the same commit. §6 named it as the one remaining candidate; this closes it.

**This is the migration where the argument for Go is strongest, because the script's own header is
the evidence.** It records two defects that the tool produced *about itself*, both of which reported
a mutation as verified while nothing had been exercised:

- `command -v staticcheck >/dev/null && staticcheck …` — the binary was absent, the `&&`
  short-circuited, the whole check evaluated to nothing, and it was reported as clean.
- `python3 lint.py 2>&1 | head -3; echo "exit=$?"` — `$?` read `head`'s status, not the lint's, so a
  red mutation printed `exit=0`.

The shell defended against that by *discipline* — "the status path here contains NO PIPES and no
`&&` chains" — a rule someone has to keep remembering, inside the mechanism built to prevent exactly
that class. In Go there is no status path to get wrong: `exec.Cmd.Run` returns the command's own
error, and an absent `sed` is a `LookPath` failure rather than a short-circuit that evaluates to
success. **A mutation checker that silently reads the wrong status certifies a gate as falsifiable
when nothing ran: G-01 inside the anti-G-01 mechanism.**

**Acceptance (a):** four scratch scenarios — happy path, vacuous expression, already-red baseline,
and a gate blind to its own mutation — run against identical scratch git repos with identical
arguments. **Byte-identical output, identical exit codes, and identical restored file contents** on
every one, modulo the tool's own name in its header. Then the documented real example
(`int4-quantizer` over `decoder/weightmat.go` against `TestInt4_forwardParity`, 3 × 35s per side):
identical transcript, `rc=0`, and `git diff --quiet` confirming both sides put the tree back.

**Acceptance (b)** (`cmd/gate/mutation_test.go`) — the meta-case, since a tool that proves gates are
falsifiable must itself be falsified: each test is a way this tool could certify a gate while nothing
was exercised. A vacuous expression is rejected; an already-red baseline is rejected **before the
file is touched** (asserted, not assumed); a gate that survives its mutation is reported as blind; a
verify command that cannot START is not a green baseline; misuse is exit 2, distinct from a finding
(1) and a pass (0); and **the subject file is asserted byte-identical after every failure path**,
because this tool edits source in place and a failed check that leaves a deliberately broken tree
behind would make the next unrelated command fail for reasons no one could attribute.

**One property the shell had that a `defer` would have lost:** restore-on-interrupt. `trap restore
EXIT INT TERM` covers Ctrl-C; a deferred call does not run on a signal. The Go version installs an
explicit `signal.Notify` handler that restores and exits 130, because the window between "mutation
applied" and "restore" is precisely when an impatient operator hits Ctrl-C.

**AND THE MIGRATION IMMEDIATELY FOUND A BUG THE SCRIPT HAD CARRIED ITS WHOLE LIFE.** `sed -i` is not
portable: GNU sed takes an OPTIONAL suffix attached to the flag (`-i.bak`), BSD sed (macOS) takes a
REQUIRED separate one — so `sed -i EXPR file` means "backup suffix EXPR, script file" on a Mac and
dies as `sed: 1: "subject.txt": unterminated substitute pattern`. The shell script had exactly that
invocation and nobody ever saw it, because it is a hand-run operator tool and nobody hand-ran it
there. The Go port's unit tests put it under CI's **darwin** job, which went red within one push.

Fixed by dropping `-i` altogether: read sed's **stdout** and let Go write the file, which is portable
on both and puts the file write on the side of the line that owns state anyway. Pinned by
`TestMutation_sedInvocationIsPortable` so the flag cannot come back. Note the shape — the migration
did not introduce a portability bug, it **exposed** one, by moving a tool from "run by a person on
one machine" to "run by CI on both".

**The do-not-harden-the-condemned rule, vindicated:** the item-6 audit filed a `pipefail` finding
against this script. It was never fixed, and the finding died with the file. That is the outcome §6
predicted, recorded here because it is cheap evidence for a sequencing rule that otherwise reads as
a matter of taste.

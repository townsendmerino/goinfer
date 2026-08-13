# Task: one Go gate-runner over `go test -json` — collapse the tallying shell + census Python

> **Status: PLAN (not started).** Tracked as QUEUE **E8**. Sibling of E7 (no-Python) but a distinct
> idea: E7 migrates scripts one-for-one; **E8 recognizes that several scripts are one program** and
> consolidates them. Same freeze as E7 — docs land now, code after the v0.13.0 tag (§C1 + CUDA gate
> first). Drafted 2026-08-12 from the E7 inventory + the item-6 shell audit.

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

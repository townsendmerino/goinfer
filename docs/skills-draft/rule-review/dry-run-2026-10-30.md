# Rule review — dry run, 2026-10-30

The rule-review skill draft (`SKILL.md` beside this) run once by hand, by mac-dx01, to show its output before Francis
approves it. Inputs: goinfer `CLAUDE.md` as #512 has it, ken `CLAUDE.md` as ken#108 has it, tack `CLAUDE.md` at tack main
`a054370`, goinfer `cmd/claude-guard` as #511 has it. The 60-day window starts 2026-08-31. **Nothing here is a decision.**

## 1. Guards

| rule | refusals, last 60 days | wired since | proposal |
|---|---|---|---|
| GA1, NV1, SC1, TL1, WT1 | none: no log on the Mac (`~/.local/state/claude-guard/` does not exist) | not wired yet (DX5: per repo, the coordinator's PRs after #511 merges) | too young to judge; first real review in December |
| pkill guard (`~/.claude/hooks/pkill_guard.py`) | not logged: it refuses without writing a log | before this review | give it a log line in the same format, so the next review can count it |

## 2. Prose rules with an incident older than 60 days

Keyword hits since 2026-08-31 are leads; a dry run reads the first ones only, so "recurred?" below is provisional where it
says so.

| rule (where it lives after #512) | incident | recurred since? | enforced by | proposal |
|---|---|---|---|---|
| staticcheck can check nothing (CLAUDE.md "Working in the tree") | 2026-08-28 | **yes**, 2026-10-08 (Go 1.27.2 export data, a canary that imported nothing passed) | `go run ./cmd/lint` (#509) | done in #512: the prose is one command |
| a doc comment claiming coverage is not coverage (CLAUDE.md "Tests"; story in the model-bring-up skill) | 2026-08-28 | 7 keyword hits; the first two (`d36fb7de`, `b17fb076`) are a census classification and a comment cleanup, not the failure | no | keep: the failure is silent, and the check is a reading habit no script can do |
| a guard that inverts is worse than none (CLAUDE.md "Measurement discipline") | 2026-08-28 | 2 hits; `7c4089c6` is a fit-guard pricing fix, not an inversion | `scripts/bench_peer.py` cites it in code | keep |
| a marginal stopping rule assumes a curve shape (CLAUDE.md "Measurement discipline") | 2026-08-28 | the hits are SB2's registered stopping rules (2026-10-30), the rule in use, not the failure | no | keep: one line |
| a retraction reaches every page (CLAUDE.md "Measurement discipline") | 2026-08-28 | 43 hits on "retract"; retractions are routine, and whether one missed a page needs each read | no | keep; the full review should read the October retractions for a missed page |
| a minimal repro can hide the bug (CLAUDE.md "Tests") | 2026-08-31 | at the window's edge | no | next month |

## 3. Rules copied across repos

| rule | goinfer (#512) | tack (a054370) | same meaning? | proposal |
|---|---|---|---|---|
| staticcheck's pin and canary | `go run ./cmd/lint` | "Use the version goinfer's CI pins (built from source, as goinfer's `CLAUDE.md` describes)" plus the canary instructions | **no longer**: goinfer's `CLAUDE.md` stops describing the build once #512 lands; tack has its own `tools/lint` and pin | tack: "`go run ./tools/lint` (it checks the pin and runs the canary)" |
| the `GOOS` trap | never combine `go run …staticcheck` with `GOOS=linux`; the lint sets `GOOS` | "Use an installed native binary and set `GOOS` only for the analysis target" | same meaning, but **claude-guard SC1 refuses tack's instructed form** (`GOOS=linux staticcheck …`) once the hook is wired in tack | tack's line becomes "`go run ./tools/lint` sets `GOOS` itself", **before or with** the tack wiring PR |
| how product code reaches `main` | PR with `--auto`, never a plain merge (2026-10-26) | "Push finished work, and read CI; do not watch it (owner rule, from goinfer)": the goinfer push rule that the PR rule replaced | **no** | Francis: does tack follow the PR and auto-merge rule? |
| a red CI on `main` | report it the moment you see it | fixed first: fix forward or revert before other work | stricter in tack | Francis: intended per repo, or one rule? |
| stating a run's wall time | the start time and the expected finish time | "the expected wall time" for any run over a minute | looser in tack (Francis's global rule asks for start and finish) | tack: "as a start time and an expected finish time" |
| the heavy-load void rule | thresholds in the timed-run skill (load over half the cores; free memory under the fit need + 2 GB) | "a cell started under heavy load is void", no thresholds | tack's is undefined without goinfer's | tack points at goinfer's timed-run skill for the thresholds |
| never `git add -A`; `main` moves under you; the testing tiers; a skip is not a pass | | | same meaning | none |
| an older linter checks nothing | staticcheck behind a canary (`cmd/lint`) | the same, `tools/lint` | ken (#108): "a golangci-lint older than the toolchain … checks nothing", with no canary | ken: consider a canary in `make lint` the same way |

## Proposals for Francis (from this dry run)

1. tack's staticcheck and `GOOS` lines move to `go run ./tools/lint` before the hook is wired in tack, or SC1 refuses
   what tack's `CLAUDE.md` instructs.
2. Say whether tack follows goinfer's PR and auto-merge rule, and whether "a red main is fixed first" is tack's alone.
3. Give the pkill guard a refusal log, so the next review can count it.

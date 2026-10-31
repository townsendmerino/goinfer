# Task: Claude Code around the repos — shorter instruction files, rules as scripts, a refusal hook, a monthly rule review (DX1–DX10) — 2026-10

> **Status 2026-10-30 (filed 09:10 PDT, updated 10:59 PDT): active.** DX1–DX4 and DX6–DX8 are merged; DX5 (the wiring) is the coordinator's, and DX9 (installing the skills) is Francis's. Scoped from the "Claude Code around your repos" half of Francis's
> suggestions of 2026-10-30 (the tack half is mac-tack5's and is not tracked here). Started by mac-dx01 on Francis's yes
> (09:04, through the coordinator). One doc for the three repos (goinfer, ken, tack), on Francis's ask: one doc, not
> several. Facts below were read at goinfer main `e54732dc`, ken main `74ef71d` and tack main `7db71f6`.

<!-- ledger -->
| id | state | next | evidence |
|---|---|---|---|
| DX1 | done | — | filed 2026-10-30 (docs-only PR) |
| DX2 | done | — | `cmd/lint` (`go run ./cmd/lint`); VERDICT: PASS 6/6 on the Mac at e54732dc |
| DX3 | done | — | both end with a VERDICT line; `test_touched.py` `VerdictLine`, `test_queue_citation_lint.py` `TestVerdictLine` (red on the old scripts) |
| DX4 | done | — | `cmd/claude-guard` (#511); review follow-ups #514 (fails closed on what it cannot parse, PX1); `TestRules`, `TestRun_refusalExitsTwoAndLogs` |
| DX5 | active | the coordinator opens the three settings PRs (goinfer, ken, tack) once #514 merges; tack's staticcheck/GOOS line changes first (SC1 refuses it) | Francis 2026-10-30 09:45: per repo (option A) |
| DX6 | done | — | #512 (b385e340): `CLAUDE.md` 594 → 181 lines; `docs/skills-draft/timed-run/`, `docs/skills-draft/model-bring-up/` |
| DX7 | done | — | ken#108 (ad85b100): ken `CLAUDE.md` 129 → 52 lines; `KEN_MCP_MMAP_MODEL` into the README |
| DX8 | done | — | #513 (2ca87227): `docs/skills-draft/rule-review/SKILL.md`; its dry run's three proposals are with Francis |
| DX9 | done | — | the three skills moved to `.claude/skills/` (timed-run, model-bring-up, rule-review) by the coordinator, 2026-10-31; `CLAUDE.md`'s two pointers follow; the rule-review dry run is now `docs/measurements/dx-2026-10/rule-review-dry-run-2026-10-30.md` |
| DX10 | done | — | the two stale Oct 21 worktrees under `goinfer/.claude/worktrees/`: checked 2026-10-30, gone; the one there now is `mm6-gnp4-check-2026-10-30`, today's, locked by its session |
| DX11 | active | the coordinator reviews and merges the goinfer and tack PRs; Francis or the coordinator picks the slimming option for tack's registration doc | Francis 2026-10-30 10:58 (standing rule 3, through the coordinator) |

## Why

Six sessions run under goinfer's `CLAUDE.md` on a given day. At 594 lines (about 13,000 tokens) it is read once per
session and paid for on every turn after, and tack's bundle loads only its first section, which today is the
model-archive rule. Several of its rules have failed as prose more than once (staticcheck that checks nothing, a lint
exit code read through a pipe, `git add -A`, `--no-verify`); a rule that a script or hook enforces does not need the
paragraph that explains how it was broken. ken's `CLAUDE.md` is about 7,700 tokens, most of it a repeat of its README,
`ARCHITECTURE.md` and `internal/envcfg`.

## Rules for every item

- **One PR per item per repo.** Branches come from plain `git worktree add` outside any `.claude` directory.
- **No session writes under any `.claude` directory.** Skills are drafted under `docs/skills-draft/<name>/SKILL.md`;
  Francis moves them (DX9). The hook's wiring is a snippet in the PR body; Francis adds it (DX5).
- **A cut changes no rule's meaning.** Every rule removed from an instruction file is either moved (the PR names
  where) or listed in the PR body with why it can go (a script now enforces it, or it repeats a doc the file points
  to). The `CLAUDE.md` PRs (DX6, DX7) are not auto-merged: Francis reads them, and they land at a quiet moment with
  the running sessions told to re-read.
- Tooling PRs (DX2, DX3, DX4) are product code: a PR with `--auto`, red first where the check is testable.

## Items

### DX2 — goinfer's staticcheck as a gate (`cmd/lint`)

Port tack's `tools/lint`: check the binary is the version `.github/actions/staticcheck/go.mod` pins, prove it
analyses by running it on a canary package (an unused struct field, in a file that imports a standard-library
package) for each target, judge the real run only where the canary fired, and end with one
`VERDICT: PASS|FAIL|INCONCLUSIVE` line (exit 0, 1, 2). The cells mirror CI's matrix: the root module, `gpu/` and
`cuda/` with their tags, `demo/agent/`, all with `GOOS=linux` as CI runs them. A missing `go.work` (a fresh
worktree) is an INCONCLUSIVE cell that says so, not a pass.

- **Gate:** tier 0. A fake staticcheck that prints nothing and exits 0 must read INCONCLUSIVE in every cell after the
  version check (tack's `TestBrokenStaticcheckIsInconclusive`, red on a gate that trusts an empty run); the canary
  must fire on the pinned build on the Mac, run once by hand and recorded in the PR.
- **Shrinks `CLAUDE.md`:** the staticcheck build line, the canary paragraph and the `GOOS` trap become one line,
  "`go run ./cmd/lint`".
- **Francis approves:** nothing beyond the PR (auto-merge).

### DX3 — VERDICT lines on `touched.py` and the citation lint

Both end with `VERDICT: PASS — …` or `VERDICT: FAIL — …` as their last line, in aikit's format, so `| tail` shows the
outcome and the "read the exit code directly" trap no longer applies to a reader. The exit codes do not change (the
pre-push hook reads them).

- **Gate:** tier 0, a test per script that runs it on a red case and asserts the last line is `VERDICT: FAIL` (red
  before the change: no such line).
- **Francis approves:** nothing beyond the PR (auto-merge).

### DX4 — the PreToolUse refusal hook (`cmd/claude-guard`)

One stdlib-only Go program. Claude Code passes it the tool call as JSON on stdin; it exits 2 with a message naming
the right command when a rule matches, and 0 otherwise (and on any input it cannot parse: it fails open). Each
refusal is appended as one JSON line (time, rule id, tool, cwd, session, the command) to a log file, which DX8 reads.

| rule | refuses | names instead |
|---|---|---|
| GA1 | `git add -A`, `--all`, `git add .` | stage explicit paths |
| NV1 | `--no-verify`, `git commit -n`, `-c core.hooksPath=…` | report the hold to Francis |
| SC1 | staticcheck with `GOOS` set (`go run …staticcheck` or an installed binary) | `go run ./cmd/lint` (DX2) |
| TL1 | `rm`/`unlink`/`mv` of `~/.goinfer-timing.lock` | `timing_lock.py status`, then wait |
| WT1 | the EnterWorktree tool | `git worktree add <path outside .claude> -b <branch>` |

Commands are parsed as a shell would split them: a quoted argument or a heredoc body (a commit message that quotes
`git add -A`) is not a command, and `bash -c '…'` and `ssh host '…'` are checked inside.

- **Where it lives (proposed):** goinfer's root module, `cmd/claude-guard`. It needs no new module (so no new CI job
  and no tag), CI's root test job runs its tests, and each machine builds it once
  (`go build -o ~/go/bin/claude-guard ./cmd/claude-guard`). Wired once in each machine's user settings, it covers
  goinfer, ken and tack together, so the three repos share one rule set and one log. The alternative, a separate
  small module or repo, buys independent tagging at the cost of a module to wire into CI; nothing needs that today.
- **Gate:** tier 0, a table test per rule with the refused forms and the near misses that must pass (a quoted
  mention, a heredoc body, `git add path`, `git push -n` as dry run), and a test that a refusal writes its log line.
- **Francis approves (DX5):** the wiring. The PR body carries the `settings.json` snippet; the program does nothing
  until it is wired.

### DX5 — wiring the hook

The program does nothing until a settings file runs it. Each machine (the Mac, nobara, mbp2) builds it once from a
goinfer checkout:

```sh
go build -o ~/go/bin/claude-guard ./cmd/claude-guard
```

and one of two settings files runs it on every Bash and EnterWorktree call:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|EnterWorktree",
        "hooks": [
          {
            "type": "command",
            "command": "\"$HOME/go/bin/claude-guard\"",
            "timeout": 10,
            "statusMessage": "claude-guard: checking the call against the repo rules"
          }
        ]
      }
    ]
  }
}
```

**Decided: per repo** (Francis, 2026-10-30 09:45). The coordinator writes the three files, since no session writes under
`.claude`.

- **Per repo (chosen):** the block goes in each repo's tracked `.claude/settings.json` (goinfer and tack have none
  yet; ken's has a `permissions` key, and `hooks` goes beside it). Every machine and every worktree of the repo gets it
  on the next pull, and the rules apply only to the repos they were written for.
- **Per machine:** the block is merged into `~/.claude/settings.json`'s existing `PreToolUse` list, beside the
  pkill guard. One edit per machine, but it also refuses `git add .` in every other project on that machine.

A machine without the binary fails open: the hook exits 127, which Claude Code reports as a hook error and does not
block on. Refusals are logged to `~/.local/state/claude-guard/refusals.jsonl` on each machine.

### DX6 — goinfer's `CLAUDE.md` to about 150 lines

By the file's own code-comment rule: say what is true now, point to the evidence, move the history.

- "Working in the tree" goes first (sessions read top-down, and tack loads only the first section).
- The Tests section's bring-up stories (LFM2, internlm2, the gzip goldens) move to a **model-bring-up** skill draft;
  its one-sentence rules stay.
- "Testing tiers", "Day loop", "Run budget" and "Gate tiers" move to a **timed-run** skill draft beside the
  night-start skills; about eight lines stay.
- The model-archive section goes to two lines: `bench_peer.py` refuses archive paths itself, and
  `docs/benchmarks.md` § "Model storage" is the authority.
- The staticcheck paragraphs shrink to DX2's one line.
- **Gate:** the PR body lists every rule removed and where it went or why it can go; a reviewer can check each
  against the old file. No rule changes meaning.
- **Francis approves:** the PR (not auto-merged), and the two skill installs (DX9).

### DX7 — ken's `CLAUDE.md` to about 40 lines

Cut: the toolchain paragraph (a bump-by-bump CVE history; `go.mod` is the pin), the env-var list (the README has it,
and `internal/envcfg` validates it), the five install snippets (the README), and the architecture section (it
repeats `ARCHITECTURE.md` and carries K17's figures). Keep: what ken is, the commands, and the hard rules: pure Go
with no cgo, the stdout contract of `ken-mcp`, float64 accumulation through `mapping[]`, sole editor. The verbatim-port
rule is stated as ken's ADR-050 has it (Francis, 2026-10-30 08:22): ranking may diverge from semble, one measured and
accepted change at a time, with Francis's yes.

- **Gate:** as DX6, the PR body's removed-rule list. Nothing under `internal/search` is touched (mbp2-cb1's branch).
- **Francis approves:** the PR (not auto-merged).

### DX8 — the monthly rule-review skill

On the doc-review pattern, drafted at `docs/skills-draft/rule-review/SKILL.md`. Once a month it lists:

- guards with no refusals in 60 days, from DX4's log;
- prose rules whose cited incident is older than 60 days and has not recurred (a commit, record or memory naming the
  same failure since);
- rules copied across repos that have drifted (tack carries several of goinfer's: `git add -A`, staticcheck, the
  `GOOS` trap, the timing lock).

It proposes cuts; Francis decides each one.

- **Gate:** one dry run over today's three files, its output attached to the PR.
- **Francis approves:** the draft and its install (DX9).

### DX11 — rigor matched to stakes

Francis, 2026-10-30 10:58, standing rule 3 (through the coordinator): full preregistration and gates only when a number will
be published or will settle a direction; ordinary engineering is a red-first test, `touched` and the lint, and a PR; a
registration fits on one screen; when unsure, the lighter tier, said in the PR. It goes into goinfer's and tack's `CLAUDE.md`
as a section of its own. In goinfer it also narrows two lines that said otherwise ("pre-registered at every tier" in
"Testing tiers"; "pre-register the decision rule for anything that will be argued about" in "Measurement discipline") and
scopes the timed-run skill draft.

**Proposal (not done): slimming tack's `docs/measurements/e1-calibration-preregistration-2026-10-18.md`** (2,404 lines at
tack `38e9c2c`; §4, the loop changes, is about 360 lines and §12, amendment C6, about 1,170). 39 files cite it, and tack's
code cites its section numbers (§12.2, §12.5, §12.9), so every § number must keep resolving.

- **(a) Archive the closed sections (recommended).** Move the bodies of the closed amendments (§7 C1, §8 C2, §10 C4, §11 C5,
  each done or void) and of §12's graded subsections to `e1-calibration-preregistration-2026-10-18-archive.md`, verbatim and
  pinned. Each moved section keeps its heading in the live doc, with its outcome in one or two lines and a link. The rules
  code cites (§12.2's void rule, §12.5, §12.9) stay in the live doc. A tack-side test lists the § numbers cited from code and
  fails if one stops resolving. Expected size: the open amendments (§9 C3, §12's open part, §13 C7, §14 C8) and the rules,
  a few hundred lines.
- **(b) One file per amendment from now on.** New amendments (C9 onward) get their own registration file, one screen each,
  and the base doc only lists them. It can be combined with (a).
- **(c) Leave it,** and apply the one-screen rule only to new registrations. Cheapest, but the 2,404 lines stay in every
  session's reach.

## Order

DX1, then DX2–DX4 (they shrink DX6), then DX6 and DX7, then DX8.

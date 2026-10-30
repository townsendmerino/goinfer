---
name: rule-review
description: "Monthly review of the rules Claude Code sessions work under in goinfer, ken and tack: guards in claude-guard with no refusals in 60 days, prose rules whose incident is older than that and has not recurred, and rules copied across repos that have drifted. It proposes cuts and fixes; Francis decides each one. Only when Francis invokes it."
disable-model-invocation: true
---

# Rule review

Francis runs this once a month. It reads the rules and the evidence for each, and proposes. **It changes no rule, guard or
instruction file**: every proposal goes to Francis, and an approved one lands as its own PR in the repo that owns it.
No session writes under any `.claude` directory.

Output: one record, `docs/measurements/rule-review-<YYYY-MM>/review.md` in goinfer, holding the three tables below and a
proposals list, each proposal with its evidence and a recommendation. Commit it docs-only. Then put the proposals to Francis
in one round (the coordinator's status-file `DECISION:`, or the session's own report).

## 0. What is a rule here

- **Guards:** each rule id in goinfer's `cmd/claude-guard/rules.go` (`GA1`, `NV1`, `SC1`, `TL1`, `WT1`, and any added
  since), and each other hook in the repos' tracked `.claude/settings.json` and `~/.claude/settings.json` (the pkill guard).
- **Prose rules:** each bold-led rule or bullet in goinfer's, ken's and tack's `CLAUDE.md`, and in the installed skills
  (`timed-run`, `model-bring-up`, `night-start`, `doc-review`), with the incident it cites, where it cites one.
- **Copies:** a rule that appears in more than one repo's file. tack carries several of goinfer's (staging paths, `main`
  moves under you, staticcheck and its canary, the `GOOS` trap, reading CI, the run budget, the testing tiers).

## 1. Guards with no refusals in 60 days

Gather each machine's refusal log, read-only: `~/.local/state/claude-guard/refusals.jsonl` on the Mac, on nobara
(`ssh nobara cat …`: a file read, which is a light touch under another session's timing lock) and on mbp2. Each line is
JSON with `time`, `rule`, `tool`, `cwd`, `session`, `command`.

| rule | refusals, last 60 days (Mac / nobara / mbp2) | last refusal | wired since | proposal |
|---|---|---|---|---|

- A guard wired for less than 60 days on a machine is **too young to judge** there; say so, do not propose a cut.
- A guard with refusals is doing its work. List its commands once (deduplicated), and flag any refusal that looks false
  (a quoted mention, a search term). A false refusal is a bug in the guard: propose the fix and a test case.
- A guard with none in 60 days on every machine where it is old enough is a **candidate**, not a cut. Give the evidence
  for keeping it anyway (how bad the failure it prevents is: a guard against deleting the timing lock fires rarely by
  design). The recommendation is Francis's to take.
- A machine with no log file: say whether the binary is installed (`ls ~/go/bin/claude-guard`). A missing binary fails
  open, so "no refusals" there means "not running".

## 2. Prose rules whose incident is old and has not recurred

For each prose rule that cites an incident with a date (in its own text, or in the skill that holds its evidence), older
than 60 days:

- Search for a recurrence since that date: `git log --since=<date> -i --grep='<the failure's words>'` in the repo, the
  measurement and task docs written since (`git log --since=<date> --name-only -- docs/measurements docs/tasks`), and the
  sessions' memory index. A keyword hit is a lead, not a recurrence: read the commit or record and decide whether the same
  failure happened again, or only the same word.
- Note whether a script or guard now enforces the rule (`cmd/lint`, `touched.py`, the citation lint, `claude-guard`, a
  test). An enforced rule's prose can shrink to the command that enforces it.

| rule (file, section) | incident date | recurred since? (evidence) | enforced by | proposal |
|---|---|---|---|---|

Proposals, in order of preference: keep (it recurred, or the failure is costly and silent); shrink to the enforcing
command; move its story to the skill that holds the evidence; cut. A rule Francis made ("owner rule", "Francis's word") is
never proposed for a cut on age alone: list it with its date and ask.

## 3. Rules copied across repos that have drifted

For each copy, put the two texts side by side and say whether they still mean the same thing. Drift to look for:

- one copy points at text the other repo has since moved (tack citing "as goinfer's `CLAUDE.md` describes" a section goinfer
  moved to a script or a skill);
- one copy is stricter or looser (thresholds, the PR rule, how a red CI is handled);
- a guard now refuses what one copy still instructs (a hook wired in a repo whose `CLAUDE.md` tells sessions to run the
  refused form);
- one repo has a script for what the other still carries as prose.

| rule | goinfer | ken | tack | same meaning? | proposal |
|---|---|---|---|---|---|

The proposal names which copy is the source and what the other should say (often a one-line pointer to the source, or to
the repo's own script).

## 4. Rules for this review

- Read every file in full before writing a row; quote the rule's text in the record, so a later review can diff it.
- Cite code by declaration and docs by path and heading (`CLAUDE.md` § "Citations"); the record is dated, so pin it if it
  cites lines.
- Nothing changes without Francis's word. A session that finds a broken guard (a false refusal, a bypass) during the
  review files the fix as a normal PR with a red test first, and lists it in the record.

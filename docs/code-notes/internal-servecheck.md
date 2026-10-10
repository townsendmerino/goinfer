# internal/servecheck: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/servecheck`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## check_g12_test.header

Moved from `internal/servecheck/check_g12_test.go` (the comment above `check_g12_test.header`) on 2026-10-09.

```text
audit-2026-09-10 G-12: the tools row passed turn two on any 200, and the stop row passed when the
stop sequence never fired. These fakes drive the FAILING direction the existing ones never did.
```

## check_test.header

Moved from `internal/servecheck/check_test.go` (the comment above `check_test.header`) on 2026-10-09.

```text
A checker that cannot go red is not a checker. These drive the client against servers that
are deliberately wrong in one specific way each, and assert it NOTICES — the same
break-it-first discipline this project applies to its own lints and gates. They need no
model, so they run in CI, which is where the routes a harness uses currently have no cover
at all (ten of the serveapp test files skip without one).
```

## TestStructured_truncatedDigitFails

Moved from `internal/servecheck/check_test.go` (the comment above `TestStructured_truncatedDigitFails`) on 2026-10-09.

```text
TestStructured_truncatedDigitFails pins V-17 (docs/review-2026-09-04.md): the prompt asks for
"366" specifically because that is M-27's shape (StopWhenComplete used to stop at the first
complete document and return a single truncated digit). The old check only confirmed the
output parsed as SOME json.Number — a truncated "3" parses exactly as cleanly as "366" and
would report OK against the very regression this row exists to catch.
```

## TestRun_zeroModelsReportsSkippedNotFullyPassed

Moved from `internal/servecheck/cmd_test.go` (the comment above `TestRun_zeroModelsReportsSkippedNotFullyPassed`) on 2026-10-09.

```text
TestRun_zeroModelsReportsSkippedNotFullyPassed pins V-17 (docs/review-2026-09-04.md): against
a server with zero models loaded, Chat/Structured/Stop/CountTokens never run at all (Run's own
else-if branch appends a single Skip row instead) — but the summary line used to print "all N
checks passed" unconditionally, reading as full coverage when only the models-list row and a
skip actually happened. Same "a SKIP IS NOT A PASS" doctrine this repo already applies to Go
test output (CLAUDE.md).
```

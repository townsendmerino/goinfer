# internal/cliutil: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/cliutil`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestOnOff_accepts

Moved from `internal/cliutil/cliutil_test.go` (the comment above `TestOnOff_accepts`) on 2026-10-09.

```text
TestOnOff_accepts is M-14 (audit-2026-09-10): --fit=off is the spelling --fit's own help text
and tasks/task-fit-to-hardware.md promise, but a plain flag.Bool rejected it with exit 2.
```

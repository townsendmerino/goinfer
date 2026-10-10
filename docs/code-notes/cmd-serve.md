# cmd/serve: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `cmd/serve`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestBackendTagGuardFailsBuild

Moved from `cmd/serve/backendtag_guard_test.go` (the comment above `TestBackendTagGuardFailsBuild`) on 2026-10-09.

```text
TestBackendTagGuardFailsBuild is the gate for audit D-B / field-report F1: since the M-19
submodule split the root cmd/serve builds no backend, so `go build -tags cuda ./cmd/serve`
(the command in every pre-v0.10.0 doc) used to exit 0 and silently produce a CPU binary. The
backendtag_guard_*.go files turn that into a compile error whose text names the submodule
entrypoint to run instead. This test proves BOTH halves for all three backends: the build
fails (non-zero exit) AND stderr carries the exact replacement command.

It runs in the DEFAULT `go test ./...` — no -short, no build tag. A finding about a silent
build earns a gate that always runs. (The guard files carry //go:build tags, so they are
absent from this default-build test binary; the test shells out to `go build -tags …`, a
separate process, to exercise them.)
```

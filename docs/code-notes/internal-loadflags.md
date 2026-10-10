# internal/loadflags: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/loadflags`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestRegister_chatAndServeGetTheSameFlags

Moved from `internal/loadflags/loadflags_test.go` (the comment above `TestRegister_chatAndServeGetTheSameFlags`) on 2026-10-09.

```text
TestRegister_chatAndServeGetTheSameFlags is the drift this package exists to end: chat lacked
--ctx, --stream-weights and --moe-cache-experts/--moe-cache-slots while serve had them, and a
cold-user run reached for --moe-cache-experts in chat. Same names, same defaults; the help may
differ only by serve's per-model-override notes and its serve-only fallback wording.
```

## TestOptions_everyFlagReachesOptions.embedint4

Moved from `internal/loadflags/loadflags_test.go` (the comment above `TestOptions_everyFlagReachesOptions.embedint4`) on 2026-10-09.

```text
Default is true (2026-09-28): the row must probe the opt-out, not the (now-default) bare
flag, or the "default does NOT look like the non-default value" check below is vacuous.
```

## TestCPUExactPrefillDisclosesTheDefaultsTrade.metalfloor

Moved from `internal/loadflags/loadflags_test.go` (the comment above `TestCPUExactPrefillDisclosesTheDefaultsTrade.metalfloor`) on 2026-10-09.

```text
--exact-prefill's help quotes each backend's floor; Metal's is 16 (metal/backend.go metalFastPrefillFloor, A-P02),
which the help used to give as 512 and then 64 (and the removed --metal-fast-prefill's as 256).
```

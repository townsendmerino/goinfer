# internal/pullcmd: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/pullcmd`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestResolveRunRef_everyFormModelAcceptsPullAcceptsToo

Moved from `internal/pullcmd/pull_test.go` (the comment above `TestResolveRunRef_everyFormModelAcceptsPullAcceptsToo`) on 2026-10-09.

```text
R15 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): `pull` and `--model` are meant
to take the same reference forms — a ref copied from one flag to the other should just work.
Before this fix, resolveRunRef (then inlined in Run) called pull.ParseRef directly, which has
no idea what "hf:" means; --model's path (pull.Resolve, pull/resolve.go:Resolve) strips it first.
So every hf:-prefixed form --model accepts (pull.IsRef reports true for it) was refused by
`pull` with a validRepo error naming "hf" as the owner.

Mutation: delete the `refArg = strings.TrimPrefix(refArg, "hf:")` line in resolveRunRef — the
hf:-prefixed cases below go red with exactly that "hf" owner error, while the bare and demo:
cases (which never carried the prefix) stay green, proving the assertion isolates the fix.
```

## TestResolveRunRef_registryShortNameRewrites.digest

Moved from `internal/pullcmd/pull_test.go` (the comment above `TestResolveRunRef_registryShortNameRewrites.digest`) on 2026-10-09.

```text
M-32 (audit-2026-09-10.md): a recommended checkpoint's SHA256/Bytes used to be dropped
on the round trip through ParseRef(c.Ref()), which only ever sets Pin for a "demo:" ref
— so every OTHER registry entry verified against whatever HF's API reports today instead
of the digest this build vouches for.
```

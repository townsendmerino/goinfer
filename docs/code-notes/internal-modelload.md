# internal/modelload: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `internal/modelload`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestLoadGuarded_saysWhenTheLoadIsNotGuarded

Moved from `internal/modelload/loadswapguard_test.go` (the comment above `TestLoadGuarded_saysWhenTheLoadIsNotGuarded`) on 2026-10-09.

```text
A load the tripwire does not cover must say so. The cold-user run on gemma-4-26b-a4b (2026-10-05) loaded a
non-.gguf source with an empty log, and the operator could not tell a guarded, quiet load from an unguarded one.
Through loadGuarded, both ways: the streamed load prints the notice naming the path, and a .gguf direct build
(guard seam stubbed to arm) prints none.
```

## TestTokenizerFromTok_reportsBothErrors

Moved from `internal/modelload/modelload_test.go` (the comment above `TestTokenizerFromTok_reportsBothErrors`) on 2026-10-09.

```text
N-25, now in the one place all three apps read a .giw's tok half: when the bytes are neither a GGUF
nor a tokenizer.json, the error names BOTH attempts. serve's copy used to return only the JSON one,
so a corrupt GGUF-sourced bundle reported "invalid JSON" and pointed at the wrong half of the file.
```

## TestLoad_safetensorsDirGoesThroughSidecar

Moved from `internal/modelload/modelload_test.go` (the comment above `TestLoad_safetensorsDirGoesThroughSidecar`) on 2026-10-09.

```text
TestLoad_safetensorsDirGoesThroughSidecar is G-S18d's load half (docs/tasks/task-multimodal-support-2026-10.md, S18 on
the Mac): where the sidecar default holds, a safetensors directory loads through a sidecar built beside it once; the
second load reuses it without a rebuild; the tokenizer, chat template included, is still the directory's; and
-direct-load keeps the directory. Red before S18: the directory loaded directly.
```

# examples/embed: notes moved out of code comments

History, measurements and open work that used to sit in the comments of `examples/embed`, moved here verbatim by the comment diet
(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and
its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,
except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).

## TestEmbedExample_outputIsNotARepeatedToken

Moved from `examples/embed/main_test.go` (the comment above `TestEmbedExample_outputIsNotARepeatedToken`) on 2026-10-09.

```text
R10 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): a README+pkg.go.dev-only reader's
own ≤40-line embed program compiled and ran on the first try — a genuine success — but printed
"Hello. Hello. Hello. Hello. ..." instead of a coherent reply, because it encoded the raw
prompt directly with no chat template. This example already applies one (chat.Detect); this
test is the gate that would have caught a regression back to that shape: it runs the REAL
binary against a real, tiny, chat-tuned checkpoint and asserts the output is not one token (or
one short phrase) repeated into a loop.
```

## TestEmbedExample_outputIsNotARepeatedToken.fixture

Moved from `examples/embed/main_test.go` (the comment above `TestEmbedExample_outputIsNotARepeatedToken.fixture`) on 2026-10-09.

```text
A fixture whose chat template chat.Detect RECOGNISES (ChatML), so the example's templated path
is the one this drives (audit-2026-09-10 G-13(g)). TinyLlama's Zephyr-style template is not one
Detect knows: the example fed it the raw prompt with or without chat.Detect, and its output was
byte-identical either way, so removing the call could not fail this test.
```

## TestEmbedExample_outputIsNotARepeatedToken.bound

Moved from `examples/embed/main_test.go` (the comment above `TestEmbedExample_outputIsNotARepeatedToken.bound`) on 2026-10-09.

```text
A templated instruct model answers and ENDS its turn; fed the raw prompt it runs on to the
256-token cap. Measured on this fixture: 32 chars with the template, 852 without it. The 400
bound was fixed before the run that tested it.
```

## TestBuildPrompt_rendersOnlyOneLeadingBOS

Moved from `examples/embed/main_test.go` (the comment above `TestBuildPrompt_rendersOnlyOneLeadingBOS`) on 2026-10-09.

```text
TestBuildPrompt_rendersOnlyOneLeadingBOS is M-39 (audit-2026-09-10): buildPrompt must not
prepend a second BOS when the chat template it just rendered already carries the family's own
BOS marker. chatml-tiny.gguf (tokenizer/testdata) is a small fixture whose ChatML template
chat.Detect recognizes AND whose vocab carries a real BOS id (unlike Qwen's real ChatML
checkpoints, whose BOS is -1 — the reason the full-binary integration test above can't exercise
this path at all, per the audit's own note). Encoding the rendered text with addBOS=true (the
pre-fix bug) visibly double-prepends the BOS id; this asserts exactly one.
```

## longestConsecutiveRepeat

Moved from `examples/embed/main_test.go` (the comment above `longestConsecutiveRepeat`) on 2026-10-09.

```text
longestConsecutiveRepeat checks every window size from minLen up to a cap for a substring
that occurs immediately followed by itself two or more times running (the repeating unit's
PERIOD, which a real loop can land on at any width — a fixed handful of guessed widths missed
a real one: "Byeagain.User:Byeagain.Assistant:" repeating at period 34), and returns the
occurrence covering the most total text along with its repeat count.
```

# MEM13-tokenizer, 2026-10-21

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM13, "MEM13-tokenizer gates" (TK-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6, Go 1.27.2; the owner's desktop open. Fixture: the committed
`tokenizer/testdata/qwen2-0.5b-coder-meta.gguf`.

## Before (main `97adc2b1`, plus the test file alone)

| | `LoadGGUF`, median of 5 |
|---|---|
| allocated | 88.58 MiB |
| kept | 37.70 MiB |

TK-G2's bars from these: allocated ≤ 62.0 MiB (0.70 ×), kept ≤ 34.70 MiB (− 3 MiB). TK-G1's ids are pinned from the same
tree in `tokenizer/testdata/mem13_tk_ids.json`.

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

**Instrument fix, made after the branch's first reading and before any verdict.** The first after-reading kept 15.32 MiB,
22 MiB under main's, where the rank map accounts for about 7.5 (the rank-map change alone read 30.20). The cause is the
instrument: aikit's `GGUFFile` has a finalizer, so after one `runtime.GC` it and its boxed metadata survive one more
cycle and read as kept. The test now runs two GCs, and both arms were re-read under it (main through a detached
worktree at `97adc2b1` with the same test file). The bar's rule is unchanged: kept ≤ main's − 3 MiB, now 21.61.

## Readings (branch `mem13-tokenizer-2026-10`)

| gate | main | branch | bar | result |
|---|---|---|---|---|
| TK-G1 identity | — | tokenizer package green; ids equal on all three fixtures; every key equal to aikit's on all three | equal | **PASS** |
| TK-G2 allocated | 88.58 MiB | 24.36 MiB (0.28 ×) | ≤ 62.0 | **PASS** |
| TK-G2 kept (two GCs) | 24.61 MiB | 15.32 MiB (− 9.29) | ≤ 21.61 | **PASS** |
| TK-G3 encode ns/op, new/old | — | median 0.9935 of six pairs (0.9890–0.9991, all below 1) | ≤ 1.02 | **PASS** |

Planted defects, each red: the id key's halves swapped (`TestTK_idsPinned`, `TestMergeSymbols_matchesNaive`); the
typed reader dropping a string array's last element (`TestTK_idsPinned`, `TestTK_typedReaderMatchesAikit`).

TK-G3 ran twice. The first pass (`tkg3-void-run.txt`, 18:33) started every cell with a 1-minute load average of 4.01 to
4.39, above the 4.0 bar, so all twelve cells are void; they are listed, not averaged. The second pass (`tkg3-run.txt`,
`tkg3.sh`, 18:34) waits for the load to fall to 4.0 before each cell: 3.41 to 3.70 at the starts. A B B A × 3,
`-benchtime 2s`, test binaries built from main plus the test file and from the branch, under the timing lock.

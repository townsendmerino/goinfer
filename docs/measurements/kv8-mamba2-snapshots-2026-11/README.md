# KV8's message-boundary and preamble snapshots for Mamba-2 hybrids on CUDA (registered 2026-11-16, before any code; a 16-core amd64 box with 62 GB of RAM and one 8 GB RTX 2070 SUPER)

Plan of record: `docs/tasks/task-kv-architecture-2026-10.md`, the KV8 row; the owner's word to build it, 2026-11-16 (relayed by the coordinator). Prior reading:
`docs/measurements/kv8-owed-2026-11/README.md` (G-M2: on `nemotron-h-4b-instruct-128k` a rewritten last message costs 75 s, 0 rows reused, and no snapshot is taken).

## What is built

- **State in the host snapshot.** `backend/cuda/kvsnap.go` places every recurrent layer's conv window and state (DeltaNet's `dnWin`/`dnState` and Mamba-2's `mbWin`/`mbSSM`, both through `cudaLayer.recurrentOf`) after the K/V rows, and
  restores them to the slot a restore names. `snapshotDecline`'s Mamba-2 refusal and `KVSnapshotRecurrent`'s `r.mamba == nil` go; `KVSnapshotBytes` prices the state.
- **The mark needs no new code in the decoder.** A recurrent model's snapshot points are cuts (`decoder/model.go:Model.residentPrefillSeedMarked`: the prefill is split at the point and the mark called between the two calls), and the
  one-token-at-a-time prefill is such a call. Serve computes `MsgBoundary` and `Boundary` for a model whose snapshots carry its state (`KeepsRecurrentSnapshots`), which now includes Mamba-2.
- **The cut cannot change a Mamba-2 prompt's numbers**, because every token takes the same per-token kernels whichever call it is in; the gate below reads that, it does not assume it.
- **Not built:** Metal, a batched Mamba-2 prefill, Marconi admission.

## Tier 0 (no timing; `nemotron-tiny` on the card, seconds to minutes)

| | what | bar |
|---|---|---|
| **N0** | `backend/cuda/kv8a_checkpoint_test.go:TestKV8Msg_cudaRestoreAtTheBoundaryEqualsCold` run on `nemotron-tiny` as well (passes of 64 and 128 rows): turn 2 rewrites the last message from id +3 and restores at the message boundary's block | **red on main first** (0 snapshots taken, 0 rows reused). Then: turn 2 reuses exactly the rounded boundary's rows, and its reply, every `mbWin` and `mbSSM` and the K/V rows are **bit-identical** to a cold prefill's. Plants, each red: the restore skips the conv window; the snapshot's state dropped |
| **N1** | `TestKV8Preamble_cudaRestoreAtTheSharedPreambleEqualsCold` on `nemotron-tiny` (128-row passes): a second conversation sharing the preamble restores the preamble's block | red on main first; then bit-identical as N0; plant: the conv window not restored, red |
| **N2** | a snapshot of an unbound slot holds that slot's Mamba-2 state (`TestRecurrentSlots_snapshotHoldsTheSlotItNames` on `nemotron-tiny`) | passes; plant: the snapshot reads the bound pair, red |
| **N3** | the store's byte count equals the bytes the snapshot reports, and `KVSnapshotBytes` equals them, on `nemotron-tiny` | exact |
| **N4** | `go run ./cmd/gate identity -backend cuda <main> HEAD`, every family that ran | 0 DIFFERENT (a default load takes no snapshot; the library path sets no boundary) |
| **N5** | the existing Mamba-2 and DeltaNet snapshot tests still pass (`TestKV8*`, `TestB35*`, `TestRecurrentSlots_*`, `TestKVSnap*`) with `-v`, 0 skips of a test that skipped before | all pass |

**Stopping rule:** all six, once; a red first (N0, N1) is the registered start, not a failure. **Void if:** a skip counted as a pass, a bar moved after a reading.

## Tier 1: T2's shape on `nemotron-h-4b-instruct-128k`, served (T-N)

Instrument: `docs/measurements/kv8-msgboundary-2026-11/served_msgboundary.py`, run as `--report-cost-only`, one server per cell, **A B B A**, same session, exclusive timing lock, main's `serve` binary against the change's, `--ctx 12288`, greedy, thinking off,
24 new tokens, `~/models/nemotron-h-4b-instruct-128k` (its CUDA bundle, local NVMe, never `/srv/models`), the 1-minute load and `MemAvailable` recorded at each start, a cell above 8.0 load void and re-run in order. **Fit check first** (`--fit-only` at the change): a refusal is NOT RUN.
Two edits to the harness, before any reading of it here: the message-boundary snapshot is the **second largest** of r1's snapshots (with a preamble snapshot taken, `min` of the first two is the preamble's; with two it is the same as before), and the cost
lines print the first three snapshots.

| | |
|---|---|
| **Question** | Does the message-boundary snapshot serve a Mamba-2 hybrid's rewritten last message, and what does keeping the three snapshots (preamble, message, Keep) cost every cold request? |
| **PASS, worst of the two pairs** | r3's first token, change over main, at most 0.5 (main's r3 is about 75 s); r3 reuses exactly r1's message-boundary snapshot's rows (main 0); all four replies equal main's; r2 reuses what main's does and its first token is at most 1.10 of main's; r4 reuses what main's does; r1's first token at most 1.05 of main's; peak server `VmSwap` at most 512 MB |
| **PARKED** | r3's ratio above 0.5 and at most 0.7, or r1's above 1.05 and at most 1.10, the others holding |
| **FAIL** | r3's ratio above 0.7; any reply that differs; r3 not reusing the boundary's rows; r1 over 1.10; r2 over 1.10 or reusing less; r4 reusing less; `VmSwap` over 512 MB |
| **Do-nothing arm** | main. A FAIL on r1 (the cost every cold request pays) stands even when r3 wins |
| **Reported, not graded** | each snapshot's host MiB and copy ms; the first cold request's total; snapshot bytes held; the 0.8B's figures beside them (84 to 163 MiB, 55 to 98 ms) |
| **Stopping rule** | the four cells, no early stop |
| **Cost** | about 2.5 min per main cell (r1 and r3 are 75 s per-token prefills), about 1.7 min per change cell if r3 drops to about 13 s: **about 9 minutes of cells plus four loads, 10 to 12 minutes in all**, stated again with the start time when it runs; over 20 minutes stops it |
| **Means** | PASS: ship by PR, default on for Mamba-2 models whose snapshots carry state. PARKED: Francis decides. FAIL: the branch stays and the record says what failed. |

**Void if:** a timed cell outside the exclusive lock, the model read from `/srv/models`, a NOT RUN counted as a pass, a bar moved after a reading.

## Readings, 2026-11-16

Build `abb4d2ac0` (the change) against `87a8b0e32` (main), each `CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`. Driver 595.112.07 as anchored. The registration above did not change after its commit.

### Tier 0

| | verdict | reading |
|---|---|---|
| **N0** | PASS, red first | `n0-n1-red.log`: on main `nemotron-tiny` failed at "snapshots do not carry the recurrent state" (both pass widths) while `qwen3next-tiny` passed. `n0-n1-green.log`: with the change, turn 2 reuses exactly the boundary's block and its reply, every recurrent buffer and the K/V rows equal a cold prefill's, on `nemotron-tiny` at 64 and 128 rows. The recurrent comparison reads the buffers the Keep checkpoint covers; `TestKV8Mamba_snapshotRestoreEqualsColdAndPlantsAreRed` reads the live Mamba-2 layers directly and asserts it compared Mamba-2 layers |
| **N1** | PASS, red first | on main the preamble test left `[]` snapshots (want 640, 1472, 2368); with the change it passes on `nemotron-tiny`, bit-identical |
| **Plants** | red as registered | `n2-n3.log`: a snapshot whose conv window was zeroed, and one holding no recurrent state, each differ from a cold prefill (the test fails if either is equal) |
| **N2** | PASS | `n2-n3.log`: `TestRecurrentSlots_snapshotHoldsTheSlotItNames` now runs on `qwen35-tiny`, `olmo_hybrid-tiny` and `nemotron-tiny`; a snapshot of an unbound slot restores that slot's state. Its plant (the snapshot's state overwritten with the bound slot's, which is what a read of the bound pair would hold) changes the logits on all three |
| **N3** | PASS | the snapshot's `Bytes()` equals `KVSnapshotBytes`, and its recurrent layer count equals `recurrentLayers()` |
| **N4** | PASS | `n4-identity.log`: `gate identity -backend cuda 8d8ac2102 abb4d2ac0`: 25 IDENTICAL, 0 DIFFERENT, 19 NOT RUN (a fresh worktree lacks the gitignored fixtures); `nemotron_h`, `olmo_hybrid` and `qwen3_5` ran |
| **N5** | PASS | `n5.log`: 60 top-level passes, 0 fails; the 8 skips are heavy-tier tests (`GOINFER_HEAVY_TESTS`), a diagnostic (`GOINFER_KV8_DIAG`) and a gitignored fixture a fresh worktree lacks, none of them touched by this change |
| **Also** | | a guard that asserted the refusal (`TestNemotronResidentCUDA_guards`) now asserts admission; the cache-state grid's text and `docs/option-state-grid.md` name Mamba-2 |

### Tier 1: T-N, `nemotron-h-4b-instruct-128k`, A B B A (08:10:42 to 08:19:20 PST, 8 min 38 s)

`tn-run1.log`, `served-tn/` (`turns.jsonl`, one `serve-*.log` a cell). Fit check first (`fit/`): 4,322 MiB on the card, a 98.5 MiB device checkpoint, a 6.3 GiB snapshot budget. The 1-minute load was 1.30 to 1.72 and about 61 GB were available at each start; no void cell; peak `VmSwap` 0 MB in all four.

| | main (A1, A2) | change (B1, B2) | change over main, pair 1 / pair 2 | bar |
|---|---|---|---|---|
| **r3, the last message rewritten**: first token | 75.05 s, 75.05 s | 14.53 s, 14.53 s | **0.194 / 0.194** | at most 0.5 |
| r3 rows reused | 0, 0 | **6,016**, 6,016 | | exactly the message boundary's block |
| r2, the next turn: first token | 0.17 s, 0.17 s | 0.18 s, 0.17 s | 1.024 / 1.015 | at most 1.10 |
| r2 and r4 rows reused | 7,430 and 7,431 | 7,430 and 7,431 | equal | not less |
| r1, cold: first token | 75.08 s, 75.05 s | 75.36 s, 75.39 s | 1.004 / 1.004 | at most 1.05 |
| replies, r1 to r4 | | equal to main's, all four, both pairs | | equal |

**The verdict is PASS.** Cost of the snapshots on r1 (reported, not graded; `--report-cost-only`): the preamble's at 2,752 rows **173 MiB host, 134 ms** (133.9 and 134.4), the message boundary's at 6,016 rows **275 MiB, 163 and 192 ms**; together 448 MiB and about 300 to 325 ms of r1's 75 s (0.4%). Host memory held 447 of 6,424 MiB. The 0.8B's were 84 to 163 MiB and 55 to 98 ms.

- **The harness printed FAIL on the first grading, and the bars were met.** The registered rule named the message-boundary snapshot as r1's second largest; r1 took **two** snapshots, not three. A request whose next turn repeats the whole prompt has `Keep` 0
  (`internal/app/serveapp/kv_boundary.go:loadedModel.recurrentKeep`: "the slot's own state serves it"), which is the case for this model's template, so there is no Keep snapshot and the second largest was the preamble's (2,752). The harness then
  reported r3's 6,016 rows as not the boundary's block. The boundary is 6,016 by the route's own count (the estimate is 6,030, and 6,016 is its 64-row block); the harness now picks the snapshot within a block of the estimate, and a `--regrade` option grades a
  recorded `turns.jsonl` without running a cell (`tn-regrade.log`, from the same four cells, nothing re-measured). The first output is `tn-run1.log`'s last line, kept verbatim.
- **r2 and r4 are served by the device checkpoint, as before**, not by a host snapshot: no Keep snapshot is taken here, so the Keep path's cost is unchanged (r2 x1.015 to x1.024 is inside the bar and the run-to-run spread of a 0.17 s figure).
- **Not read:** Claude Code's live traffic on a Mamba-2 hybrid, a Mamba-2 model whose template rewrites the end of the prompt (a Keep snapshot of a Mamba-2 state), Metal.

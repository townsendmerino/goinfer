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

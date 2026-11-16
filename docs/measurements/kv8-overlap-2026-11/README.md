# KV8-overlap: take the snapshot's host buffer off the cold request's first token (registered 2026-11-16 10:45 PST, before any code; a 16-core amd64 box, 62 GB RAM, one 8 GB RTX 2070 SUPER)

Ledger row: `docs/tasks/task-kv-architecture-2026-10.md`, KV8-overlap ("overlap a snapshot's device-to-host copy with the rest of the prefill ... about 116 ms more on a cold request than main's on the 0.8B at 7.4k ids").

## What the row's premise misses (probes, 2026-11-16, throwaway tests, `backend/cuda` on this card)

A snapshot's host cost is three legs, and the copy is the smallest. For a 275 MiB snapshot: **pinned allocation 139 ms, the device-to-host copy 22 ms (12.2 GB/s), the free 49 ms**
(64 MiB: 33, 5, 11 ms; 128 MiB: 65, 10, 22 ms: page-locking costs about 0.5 ms a MiB). Overlapping the copy alone could take 14% of the allocation-plus-copy cost off the first token, so it cannot move it.
Pinning on a second thread while a prefill runs (a raw driver call) does overlap, but it took 905 ms for that 275 MiB and slowed the prefill by 50 ms (2.0%): about two thirds of the serial cost back, with a hint from the decoder to start it and a second thread.
**Ordinary (pageable) memory costs less on every leg:** make 0.5 to 28 ms, device-to-host copy 29 to 50 ms (5 to 9 GB/s), host-to-device 29 ms (9.2 GB/s, against pinned's 22 ms), and nothing to free.

## What is changed (replaces the current path; no second path)

`backend/cuda/kvsnap.go:cudaKVSnapshot` holds its rows in a plain `[]float32` of the same size instead of a page-locked `HostBuffer`: the same copy loops write and read it, there is no pinned allocation or free, and `Release` drops the slice.
Device bytes: unchanged (no device buffer is added). Host bytes: the same bytes held, as ordinary memory instead of page-locked (so the kernel may swap it under pressure, which pinned memory cannot be; the store's budget, a tenth of RAM, and its admission are unchanged and a guard threshold is not touched).
Restore uploads from ordinary memory: about 7 ms more a 275 MiB snapshot, on the restoring request's first token.

## Gates

| | tier | what | bar |
|---|---|---|---|
| **O0** | 0 | every existing snapshot test, bit for bit: `qwen3next-tiny` and `nemotron-tiny` boundary and preamble restores, B35's, the plants, the slot tests, and `TestKV8Real_cutEqualsUninterrupted` on `qwen3.5-0.8b` (heavy) | all pass, 0 new skips; the plants stay red |
| **O1** | 0, **red on main first** | taking a snapshot and dropping it makes **0** page-locked host allocations (`gpudev.Device.PinnedHostBuffers`, a counter) | main: 1 per snapshot, red; change: 0 |
| **O2** | 0 | a snapshot holds exactly `Bytes()` host bytes, and the store's byte count is unchanged | exact |
| **O3** | 0 | `gate identity -backend cuda <main> HEAD` | 0 DIFFERENT |
| **T-O** | 1 | `docs/measurements/kv8-msgboundary-2026-11/served_msgboundary.py`'s conversation on `~/models/qwen3.5-0.8b` (local NVMe), `--ctx 12288`, `--kv-sessions 1`, greedy, 24 new tokens; **5 repetitions of A B B A = 10 pairs**, main against the change, one server per cell, exclusive timing lock, load and `MemAvailable` recorded at each start, a cell above 8.0 load void and re-run in order (`ab_pairs.py`). Fit check first | below |

**T-O rule, per pair (change over main).** *r1* (the cold request, which takes the snapshots): first token. *r3* (the restore): first token. Reported: the snapshot copy ms the server logs, peak `VmHWM` and `VmSwap`.

| | |
|---|---|
| **PASS** (merges under CLAUDE.md's bit-identical speed rule) | every reply equal to main's in all 20 cells; r1 faster in **at least 9 of 10 pairs**; r3's median ratio at most 1.02; peak `VmSwap` at most 512 MB; device bytes unchanged; host bytes held equal |
| **PARKED** (Francis decides) | r1 faster in 7 or 8 of 10, or r3's median ratio above 1.02 and at most 1.05 |
| **FAIL** | r1 faster in fewer than 7, any reply differs, r3's median above 1.05, or `VmSwap` above 512 MB |

**Do-nothing arm:** main. **Stopping rule:** all 10 pairs, no early stop. **Expected size of the effect** (from the row and the probes, not a bar): about 100 ms of r1's 7.4 s, 1.4%; a PASS asks the sign to hold in 9 of 10 pairs, not the size.
**Cost, stated before the run:** a cell is about 30 s (load 6 s, r1 7.4 s, r2 2 s, r3 5 s, r4, settle), 20 cells **about 10 to 14 minutes** plus the fit check; started and expected finish times are written to the status file when it starts.
**Void if:** a timed cell outside the exclusive lock, the model read from `/srv/models`, a skip counted as a pass, a bar moved after a reading. **Means:** PASS: ship by PR. PARKED or FAIL: the branch stays and the record says what failed; the row says what the probes found.

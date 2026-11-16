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

## Readings, 2026-11-16

The registration above did not change after its commit. Main is `23c707269`; the change is `7b15b2f8f` (the second run) and `86d234f14` (the first). Driver 595.112.07 as anchored. `Qwen3.5-0.8b` from local NVMe, `--ctx 12288`, one slot, the exclusive lock, 20 cells a run, no void cell, peak `VmSwap` 0 MB in every cell.

**The first implementation held the rows in Go heap memory** (`86d234f14`; `ab-run1.log`, `served-ab/`): r1 faster in **10 of 10 pairs** (median change over main 0.9683), but the server's **peak resident set was higher, 1,368 to 1,419 MB (median 1,398) against main's 1,217 to 1,278 (median 1,232)**, about 150 MB more with the same 1,014 MiB of snapshots held:
freed heap lingers until a collection, where the pinned free returned its pages at once. The registered clause is "host bytes held equal", and a peak that is a tenth higher is not equal, so that implementation was not graded as a PASS. The rows are now an anonymous mapping returned at `Release` (`backend/cuda/hostrows_linux.go`; the heap where the mapping fails or off Linux).

### Tier 0

| | verdict | reading |
|---|---|---|
| **O0** | PASS | `o0b.log` (the final code; `o0.log` is the heap implementation's): 120 passes, 0 fails, over the boundary, preamble and Keep restores on `qwen3next-tiny` and `nemotron-tiny`, B35's, the slot tests and plants, KV4's demotion and promotion tests, and `TestKV8Real_cutEqualsUninterrupted` on `qwen3.5-0.8b` (bit-identical: PASS). The skips are heavy-tier or diagnostic tests that skip the same way on main |
| **O1** | PASS, red first | in a detached worktree at main with only the counter and the test: "taking a snapshot left 3 page-locked host buffers, 2 before" on both fixtures. With the change the count is unchanged |
| **O2** | PASS | the snapshot holds exactly `Bytes()` host bytes, which equal `KVSnapshotBytes`; a released snapshot holds none and refuses a restore |
| **O3** | PASS | `o3b-identity.log`: `gate identity -backend cuda 23c707269 7b15b2f8f`: 25 IDENTICAL, 0 DIFFERENT, 19 NOT RUN (a fresh worktree lacks the gitignored fixtures); `o3-identity.log` is the heap implementation's, the same |

### Tier 1: T-O, 5 x A B B A, 10 pairs

| | first run (heap), `served-ab/` | **final (mapping), `served-ab2/`**, `ab-run2.log` | bar |
|---|---|---|---|
| r1 (the cold request): first token, main -> change | 7.409 -> 7.183 s | **7.389 -> 7.182 s** | |
| r1 pairs faster | 10 of 10 | **10 of 10** (ratios 0.961 to 0.982, median **0.9734**) | at least 9 of 10 |
| r3 (the restore): first token | 2.435 -> 2.307 s | 2.415 -> 2.316 s; median ratio **0.9624** (0.930 to 0.987) | median at most 1.02 |
| r2, r4 (the Keep restores, which also take a snapshot): first token | 0.325 -> 0.219 s, 0.349 -> 0.221 s | 0.344 -> 0.218 s, 0.349 -> 0.218 s; median ratios 0.636 and 0.622 | reported |
| rows reused (r1 to r4) | equal | equal (0, 7,360, 6,016, 7,360) | not less |
| replies | all equal | **all equal to main's in all 20 cells** | equal |
| snapshots taken / host MiB held a cell | 6 / 1,014 both arms | 6 / 1,014 both arms | equal |
| snapshot copy ms a cell (the server's log) | 663 -> 214 | **663 -> 214** | reported |
| peak server `VmHWM` (median, range) | main 1,232 (1,217 to 1,278); change 1,398 (1,368 to 1,419) | **main 1,254 (1,203 to 1,276); change 1,229 (1,210 to 1,243)** | equal |
| peak `VmSwap` | 0 MB | 0 MB | at most 512 MB |

**The verdict is PASS, on the final implementation.** It merges under CLAUDE.md's bit-identical speed rule: outputs bit-identical (O0 to O3 and every reply), scratch and allocated memory not higher (device bytes: none added; host: the same 1,014 MiB held, now ordinary memory instead of page-locked, and the peak resident set equal),
faster on 10 of 10 timed pairs, and it replaces the pinned allocation rather than adding a path.

- **The gain is where the probe said it would be.** The snapshot legs of a cold cell fell from 663 ms to 214 ms (the pinned allocations and frees, not the copies); r1's first token fell by about 200 ms (2.8%), more than the row's 116 ms because r1 takes three snapshots and the second and third are taken after the cut, on the first token's path. r2, r3 and r4 each take or replace a snapshot too, which is why they are faster by 100 to 130 ms: that is the same allocation and free.
- **The expected size was 1.4% and the measured one is 2.7%; the sign held in every pair of both runs.**
- **What the change costs, stated:** a restore uploads from ordinary memory at about 9.2 GB/s where pinned did 12.2 (about 7 ms a 275 MiB snapshot, inside r3's measured gain); the rows can now be swapped out under memory pressure, which page-locked memory could not (the store's budget, a tenth of RAM, its admission and the swap guard are unchanged); and a mapping that fails falls back to the Go heap, where a pinned allocation failure used to decline the snapshot.
- **KV4's host demotion shares the snapshot type** (`backend/cuda/kvdemote.go` takes `snapshotKV`), so a demoted conversation's rows are ordinary memory too; its tests pass (O0) and it was not timed here.
- **What was not done:** overlapping the copy with the rest of the prefill (the row's literal ask). The probes (above) put it at 14% of the allocation-plus-copy cost, and the allocation, not the copy, was the lever.

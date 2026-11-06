# MEM6 on CUDA: resident prefill embeds per pass, and the memory comes back (2026-11, nobara-sb01)

The ledger row (`docs/tasks/task-memory-campaign-2026-10.md`, MEM6): the Metal routes were done 2026-10-22 (`docs/measurements/mem6-2026-10/README.md`); left are the CUDA cells and the `RssAnon` gate. MEM0's nobara half had
measured the host embeddings of a 20K prompt **held after the request**: +218 MiB on the 1.5B and +476 MiB on the 7B, still there 10 s later (`docs/measurements/mem0-2026-10/README.md`, "MEM6").

## The change

CUDA's resident implements `decoder.ResidentPrefillRows` (`PrefillLastRows`): the decoder embeds a range of the suffix into one reused slab each time the chunked prefill asks for a pass,
so the host holds one pass of embedding rows, not the whole suffix. `prefillChunkedMarked` is the same loop over a rows source (`prefillChunkedRows`); the slice entry point passes `embeddings[lo:hi]`.
A pass retried at half the width asks for its range again. The marked, image, span and m-RoPE routes are unchanged.

## Gates

| gate | tier | instrument | PASS iff |
|---|---|---|---|
| MEM6-C1 rows | 0, the card | `TestMEM6_cudaPrefillHostRows`: `llama-tiny` with its window raised to 4,096, CUDA-resident, the pass set to 256 rows (the learned chunk cap stands in for a real model's pass), prefills of 1,024 and 2,048 rows; the most embedding rows the decoder holds at once, through the test observer | at most one pass (256). **Red first on main:** 1,024 and 2,048 held |
| MEM6-C2 identity | 0, the card | the same test: the last logits and 32 decode steps, hashed, through `ResidentPrefillSeedForTest` against the slice entry point (`PrefillLast` on every row embedded up front) | byte-equal |
| MEM6-C3 neighbours | 0, the card | the prefill, KV3 and resident tests of `backend/cuda` | pass as on main |
| MEM6-C4 `RssAnon` returns | 1, the card | `mem0_linux.py` (MEM0's cell, unchanged): one fresh `serve` per cell at `-ctx 32768`, one cold 20K-token request, `RssAnon` read every 250 ms; `cuda:1.5b` and `cuda:7b` (`~/models` NVMe), serve built from main (`serve-mem6-before`, `6ab62e5b2`) and from the branch (`serve-mem6-after`); one run each, before first; the exclusive timing lock | on both models the branch's `RssAnon` 10 s after the request is within **64 MiB** of its value after load. CUDA keeps no session state on the host beyond the id record (20K ints, 160 KB), so nothing is added for it. Reported: the peak, and main's own figures |
| MEM6-C5 identity of replies | 1, the card | the cell's reply text (`max_tokens` 8, greedy) on main and on the branch | equal |
| MEM6-C6 identity | 1, the card | `go run ./cmd/gate identity -backend cuda <main> <branch>` | every family that ran IDENTICAL |

**PARKED:** within 128 MiB, the others holding. **FAIL:** beyond that, a different reply, or a DIFFERENT family.

- **Void:** a run with the one-minute load above 8.0 at its start, or the lock not held; re-run once in order and listed. The prompt must tokenize to about 20K on the Qwen tokenizer (`mem0_linux.py` prints it); a rejected request (a 400) measures nothing and is void.
- **Not the question:** speed, the peak during the request beyond what is reported, other models, the load's own transient (about 4 GB on the 7B, MEM0's observation, a different item).
- **Cost and times (from `date`):** tier 0 minutes; MEM6-C4 four cells at about 1 minute each; MEM6-C6 about 6 minutes.
- **Void if:** a card run outside the lock, a NOT RUN counted as a pass, a bar moved after a reading.

Registered 2026-11-05 16:24 PST (the time from `date`), after MEM6-C1 to C3 were written and read (`red.log`, `green.log`) and before MEM6-C4 ran.


## MEM6-C4 read once, 2026-11-05 16:24 to 16:26 PST: FAIL by the rule as registered

`c4-first/` (`before/`, `after/`, `c4-before.log`, `c4-after.log`, `summary.txt`). `mem0_linux.py`'s cells on `serve-mem6-before` (main, `6ab62e5b2`) and `serve-mem6-after` (the branch with `PrefillLastRows` alone), the exclusive timing lock, load 2.22 and 1.33, 59 GB available, no void cell; the prompt tokenized to 20,201 on every run, `max_tokens` 8.

| `RssAnon`, MiB | after load | peak | +10 s | held (+10 s over after load) |
|---|---|---|---|---|
| 1.5B main | 102 | 303 | 303 | **+201** |
| 1.5B branch | 127 | 195 | 195 | **+68** |
| 7B main | 96 | 639 | 639 | **+543** |
| 7B branch | 118 | 262 | 261 | **+143** |

The bar was 64 MiB. The 1.5B misses it by 4 MiB (the PARKED band) and **the 7B, at +143 MiB, is over 128: FAIL as registered.** It stays on record. The embedding slab fixed about 70% of the held memory and something else held the rest.

**Diagnosis (a throwaway test on the 1.5B at int4, deleted):** after the 20K prefill `RssAnon` was +137 MiB; `runtime.GC` brought the heap in use back to its after-load figure and left `RssAnon` where it was; `debug.FreeOSMemory` returned it to within 3 MiB of after load. So the residue was Go garbage the runtime had not returned, not live memory. The allocation profile between load and the end of the prefill (`runtime.MemProfileRate = 4096`, a GC before each profile, `pprof -base`) put **118 MiB of 136 MiB in `prefillCore.func1`**, the line `xhost := make([]float32, M*hidden)`: every pass copied its embeddings into a fresh contiguous block before the upload, so a 20K prompt allocated its whole size again as garbage, one pass at a time. The slab itself was 12 MiB.

## The second change and MEM6-C4b, registered 2026-11-05 16:44 PST before the reading

**The change:** `prefillCore` stages a pass's embeddings in one block grown to the widest pass and kept (`cudaResident.xhostBuf`; a row shorter than hidden is zero-filled, as the fresh block was), not a new block per pass.

**A tier 0 gate added after the first green reading, and its bar:** `TestMEM6_cudaPrefillAllocatesNoRowCopy` reads the Go bytes a 2,048-row prefill allocates against a 1,024-row one on `llama-tiny` (pass 256): the slope per prompt row must stay under one embedding row. Red before the change (`red-alloc.log`: 437 B a row against a 256 B embedding row, 1.71); 180 B after. **Its first bar, an eighth of an embedding row (as Metal's G1 used), read 180 B against 32 B and was wrong for a 64-wide fixture, where other per-row costs (about 181 B) exceed a row; I replaced it with one row, the copy that was removed, after seeing that figure.** This gate points at the mechanism and is not the verdict; MEM6-C4b is.

**MEM6-C4b:** MEM6-C4 again, unchanged in everything but the build: `serve-mem6-after2` (the branch with both changes) against the same `serve-mem6-before`, one run each, before first, the exclusive timing lock. **The bar does not move:** `RssAnon` 10 s after the request within **64 MiB** of its value after load on both models (PARKED within 128, FAIL beyond), the same reply, void and stopping rules as above. The first reading is not replaced. Cost about 3 minutes. MEM6-C6 (`gate identity`) after it.

### MEM6-C4b, 2026-11-05 16:44 to 16:46 PST: the 1.5B passes, the 7B is PARKED

`c4b/` (`beforeb/`, `afterb/`, `c4b-before.log`, `c4b-after.log`, `summary-b.txt`). `serve-mem6-before` (main) against `serve-mem6-after2` (the branch, both changes), the exclusive timing lock, load 1.69 and 1.23, 60 GB available, no void cell, 20,201 prompt tokens each, peak `VmSwap` 0.

| `RssAnon`, MiB | after load | peak | +10 s | held | bar (64) |
|---|---|---|---|---|---|
| 1.5B main | 111 | 346 | 346 | **+235** | |
| 1.5B branch | 107 | 140 | 140 | **+33** | **PASS** |
| 7B main | 93 | 547 | 547 | **+453** | |
| 7B branch | 120 | 201 | 201 | **+81** | **PARKED** (over 64, within 128) |

**By the registered rule the 7B is PARKED, not a pass**, and this README does not call it one. The change returns 86% (1.5B) and 82% (7B) of what main holds after a 20K prompt, and the held memory no longer grows with the prompt.
**What the +81 MiB probably is, by arithmetic and not measured:** the two buffers the change keeps or leaves as fresh garbage are pass-wide: the embedding slab (freed after the request, returned by the runtime's own schedule) and `xhostBuf` (kept for the resident's life). At the 1.5B's hidden width (1,536) and a 2,048-row pass those are 12.6 MiB each, which matches the 12.3 MiB the profile showed for the slab and most of the +33; the 7B's hidden width is 2.3 times that. Three levers could close the rest if the owner wants the 64 MiB bar met on the 7B: a narrower embedding pass, releasing `xhostBuf` when a prefill ends, or `debug.FreeOSMemory` after a long prefill. Each trades something (a smaller pass is slower, a release re-allocates on the next long prompt, a forced GC costs a pause), so none is made here.

### The other gates

- **C1, C2, C3, C-alloc:** `red.log`, `red-alloc.log`, `green.log` (final): rows held 1,024 and 2,048 → 256 (one pass), the last logits and 32 decode steps byte-equal to the slice entry point's, 40 tests of the prefill, KV3, resident and MEM6 families pass; the allocation slope 437 → 180 B per prompt row.
- **C5 (the reply text): NOT RUN.** `mem0_linux.py` records the usage and not the reply, so the cells cannot say the two servers answered alike; the identity claim rests on C2 and C6.
- **C6:** `gate identity -backend cuda be595d7fe 201a1f75d` (`identity.log`), 16:46:43 to 16:52:11 PST: **29 IDENTICAL, 0 WITHIN TOLERANCE, 0 DIFFERENT, 14 NOT RUN**, GREEN. (The "before" servers were built at `6ab62e5b2`; main had moved to `be595d7fe` by the identity run, by docs and other PRs.)

**Verdict:** the CUDA half of MEM6 is built and gated at tier 0; the `RssAnon` return gate is met on the 1.5B and PARKED on the 7B. The ledger row says so. Not covered: Gemma 4 E2B (its cell is not run here; its per-pass PLE block `pleHost` is allocated fresh each pass and was not changed), a CPU cell, Metal (done 2026-10-22).

## Decision, 2026-11-05 17:43 (coordinator): PARKED stands

The 7B stays PARKED at +81 MiB against the 64 MiB bar. The bar does not move and no lever is pulled; the PR stays open, marked PARKED in its title and in the ledger row. It comes back after D8b with a mechanism, not a new bar. The levers and what each costs, so the owner can choose: (1) a narrower embedding pass: less host staging, slower prefill; (2) release `xhostBuf` when a prefill ends: the held block goes, the next long prompt re-allocates it; (3) `debug.FreeOSMemory` after a long prefill: returns the slab's garbage, at the cost of a forced-GC pause on the request path.


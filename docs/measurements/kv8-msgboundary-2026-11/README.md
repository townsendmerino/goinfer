# KV8's message-boundary checkpoint on CUDA (registered 2026-11-15 13:09 PST, before any code; nobara-kv8)

Plan of record: `docs/tasks/task-kv-architecture-2026-10.md`, KV8, "Next concrete step on CUDA" (2026-11-02) and its Claude Code note. Evidence for the position: KV0-H's A5
(`docs/measurements/kv0h-hybrid-traffic-2026-10.md`): 2 of 8 Claude Code D requests left the previous prompt 3 ids into a 1,386-token user message, 25 ids before that prompt's
end. Two requests, under that study's bar of three, so this registers whether the position works and what it costs, not that traffic needs it.

## The question

For a recurrent model on CUDA, when the next request keeps every earlier message and rewrites the last one from near its start, does a state kept at that message's start serve
it, and what does keeping it cost every request?

## What is built (one mark more in the cold prefill, nothing new at restore)

- **The position.** serve finds `KVScope.MsgBoundary`: the ids before the prompt's last turn's content, by `prefixBoundaryOf`'s method (render again with the last turn's content
  changed; the segments that render identically are the boundary). 0 for one turn, an unknown template, or a model whose snapshots do not carry its state.
- **The mark.** `MsgBoundary` rounded down to the 512-row pass grid, kept as a recurrent snapshot beside B35's `Keep` one, in the same store, under the same budget and admission
  (`decoder/kvsnap.go:Model.kvBoundaryPlan`, `residentSnapshotRows`). Taken only when it lies above the restore point (or 0), above `kvSnapMinShared`, on a block, and below Keep's.
- **Restore** is today's: the deepest held snapshot the prompt matches through (`covering`). A recurrent snapshot serves only its own end (`kvSnapHolds`), so nothing in the restore changes.
- **Not built:** a checkpoint per message beyond the last, Marconi admission, Metal (no recurrent snapshot there), Mamba-2 (refused on CUDA).

## Gates

| | tier | what | bar |
|---|---|---|---|
| **M0** | 0, no device (hash fake, CI reads it) | turn 2 repeats turn 1 through its last message's start and rewrites the message from id +3; turn 3 extends turn 2 normally. Reply equal to a cold run; rows reused equal the pass-grid-rounded boundary | passes, and **red on main first** (main reuses 0 rows: the plant is main). Plants, all red: MsgBoundary ignored; the point rounded up instead of down; the boundary snapshot replacing Keep's |
| **M1** | 0, no device | `MsgBoundary` from the real Qwen3.5 template with and without tools equals the id count before the last turn's opening marker; 0 for one turn; EncodeSegments prefix checked, not assumed | equal on all cases; a plant that renders the wrong turn is red |
| **M2** | 0, nobara CUDA, tiny | `qwen35-tiny` and `olmo_hybrid-tiny` (nemotron-tiny: refused, asserted refused): the restore at the boundary and the suffix prefill give logits **bit-identical** to a cold prefill of the same prompt, every position past it; conv window included | 0 differing logits. Plants, red: restoring the Keep snapshot's state at the boundary's rows; the conv window skipped |
| **M3** | 0 | the store's budget holds with two recurrent snapshots a request; a budget for one keeps Keep's and says why; counters (`kvSnaps` stats) name each decline | exact bytes; a plant that admits past the budget is red |
| **M4** | 0, nobara | `gate identity -backend cuda` at the change: every family that ran stays IDENTICAL (the library path sets no MsgBoundary) | 0 DIFFERENT |
| **T1** | 1, nobara | served, `~/models/qwen3.5-0.8b`, `-backend cuda`, thinking off, greedy, 24 new tokens, local NVMe. Conversation of about 8,000 ids: system 3,000, history 3,500, last user message 1,400. Requests: r1 cold; r2 the next turn (Keep restore, today's path); r3 rewrites the last message from +3 | see below |

**T1's cells:** one server per cell, built at main (`8aaea899b`) and at the change, **A B B A**, same session, exclusive timing lock, the 1-minute load and `MemAvailable` recorded at each
start, a cell above 8.0 load void and re-run in order. Fit check first: the banner at the change, by the server's own plan; a refusal is NOT RUN.

**T1 decision, worst of the two pairs:**

| | |
|---|---|
| **PASS** | r3's first token, change over main, at most 0.5; r3 reuses exactly the rounded boundary's rows (main: 0); r3's reply equals main's cold reply; r2 reuses what main's r2 reuses and its first token is at most 1.10 of main's; r1's first token at most 1.05 of main's; peak server `VmSwap` at most 512 MB |
| **PARKED** | r3's ratio above 0.5 and at most 0.7, or r1's above 1.05 and at most 1.10, the others holding |
| **FAIL** | r3's ratio above 0.7, a reply that differs, r3 not reusing the boundary's rows, r1 over 1.10, r2 slower than 1.10 or reusing less, or `VmSwap` over 512 MB |

- **The do-nothing arm is main.** A FAIL on the cost row (r1) stands even when r3 wins: the mark runs on every cold request, used or not.
- **Reported, not graded:** snapshot bytes and copy time at the boundary and at Keep, host memory held.
- **Not read:** the 9B (a second size only if the 0.8B resolves, for a reason that the copy's cost changes with the state's size), Claude Code's real traffic, Metal.
- **Void if:** a timed cell outside the lock, the model read from `/srv/models`, a NOT RUN counted as a pass, a bar moved after a reading.
- **Stopping rule:** the four cells, no early stop. **Cost:** fit check about 1 minute; each cell about 1 minute; start and finish times from `date` when it runs.
- **Meanings.** PASS: ship by PR, default on for models whose snapshots carry state. PARKED: Francis decides. FAIL: the branch stays, the record says what failed.

## Readings, 2026-11-15 (nobara-kv8)

Written after the runs, from `date` and the logs named. The registration above did not change after its commit except one stamp (below).

- **The registration's time was corrected.** It was first written as 13:25 PST, a guess; its commit (`77ab5fcc4`) is 13:09:54 PST, and both pages now say 13:09. No gate had run.

### M0 to M3 and M4

| | reading |
|---|---|
| **M0** | `decoder/kv8_msgboundary_test.go`: `TestKV8Msg_aRewrittenLastMessageRestoresTheBoundary` was **red on main** first (1 snapshot taken, not 2). Plants, each red: the point rounded up instead of down (2 tests); Keep's snapshot allowed to replace the message's (2 tests); the snapshot a request restored from dropped twice (the store's byte count goes wrong, 1 test). All pass on the change |
| **M1** | `internal/app/serveapp/kv8_msgboundary_test.go`: the boundary is the ids before the last turn's role and content (the template renders `<|im_start|>` as its own segment, so the boundary sits just before the role word, which this page's registration called "the opening marker"), on all four chat routes, with and without tools; 0 for one turn and for a model whose snapshots do not carry its state. **Not red first:** the serve code was written before its tests; the plant (rendering the first turn's content, not the last's) is red. |
| **M2** | `backend/cuda/kv8a_checkpoint_test.go:TestKV8Msg_cudaRestoreAtTheBoundaryEqualsCold`, `qwen3next-tiny`, 64-row passes, turn 1's snapshots at 1,472 and 2,368: turn 2 reuses exactly 1,472 rows, and its reply, every recurrent buffer and the K/V rows are **bit-identical** to a cold prefill's. Plant (the restore skips the conv window): state and K/V rows differ, red. Log: `m2-cuda.log` |
| **M2, as registered, not met** | the registration named `qwen35-tiny` and `olmo_hybrid-tiny`. Both declare 64 positions: a prompt of the 2,048 ids a snapshot needs runs on the CPU, and a snapshot is whole 64-row blocks, so none fits below their cap. The DeltaNet device mechanism is read on `qwen3next-tiny` instead, and the split of one prefill into two calls on `qwen35-tiny` is `TestB35SplitPrefill`'s. **`olmo_hybrid-tiny` has no CUDA read of this.** Mamba-2 (`nemotron-tiny`) keeps no snapshot: `TestKV8Msg_cudaMamba2KeepsNoBoundarySnapshot` passes |
| **M3** | `TestKV8Msg_theBudgetHoldsWithTwoSnapshotsARequest`: room for both holds both; room for one holds Keep's; room for none holds none; bytes never exceed the budget. Plant (the reserve always succeeds) is red on "room for one" |
| **M4** | `gate identity -backend cuda 8aaea899b HEAD` at `5009ee053`: **GREEN, 25 IDENTICAL, 0 DIFFERENT, 19 NOT RUN** (`identity.log`). The hybrids ran: qwen3_5, qwen3_next, olmo_hybrid, nemotron_h. The NOT RUN count is higher than the last reading's 14 because this fresh worktree lacks the gitignored fixtures (`gemma4`'s four assets, among others), so those cells skip |

### T1: PARKED by the registered rule (13:26 to 13:28 PST)

`served-t1/run.log`, `turns.jsonl`, `fit-check.log`. Main `8aaea899b` against the change `5009ee053`, `~/models/qwen3.5-0.8b` from NVMe, `--ctx 12288`, one slot, A B B A, the exclusive timing lock, the 1-minute load 0.59 to 0.82 and about 61 GB available at each cell's start, no void cell, peak `VmSwap` 0 MB. The fit check read 1,394 MiB on the card and a 6.3 GiB snapshot budget. The prompts are 7,434 to 7,464 ids.

| | main | change | change over main, pair 1 / pair 2 | bar |
|---|---|---|---|---|
| **r3, the last message rewritten**: first token | 7.36 s, 7.38 s | 4.80 s, 4.83 s | **0.652 / 0.654** | at most 0.5 |
| r3 rows reused | 0, 0 | **4,096**, 4,096 (r1's boundary snapshot, exactly) | | exactly the boundary's |
| r3's reply | | equals main's cold reply, both pairs | | equal |
| r2, the next turn (Keep): first token | 2.15 s, 2.17 s | 2.14 s, 2.20 s | 0.994 / 1.013 | at most 1.10 |
| r2 rows reused | 6,144 | 6,144 | | not less |
| r1, cold: first token | 7.39 s, 7.42 s | 7.39 s, 7.50 s | 1.000 / 1.012 | at most 1.05 |

**The verdict is PARKED:** r3's ratio is above 0.5 and at most 0.7, every other clause holds. **Francis decides.**

- **Why only 4,096 of 7,435.** The resident's passes are 2,048 rows wide here, and the mark falls on a pass boundary. The last message starts at about 6,100 ids; rounding down to the grid gives 4,096, so r3 re-prefills 3,339 ids of 7,435 where the boundary alone would leave about 1,330. This is B35's rounding ("a warm turn re-prefills up to about one pass"; its follow-up, a cut at the point itself, removes it), and it takes up to 2,047 rows of this mechanism's win at this width. A cut at the point is a separate slice with its own registration: `TestB35SplitPrefill` says a split whose passes are all 16 rows or more is bit-identical, which is the gate it would have to re-read at the decoder.
- **What it costs.** A second snapshot a cold request: 4,096 rows is 115 MiB of host memory and 97.6 ms, beside Keep's 163 MiB and 98.1 ms (`serve-B1.log`). The request's first token did not move (r1 1.000 and 1.012), so the copy sits behind it. The restore of the 4,096-row snapshot took 11.7 ms.
- **Reported, not graded:** the snapshots held after the three requests are 4,096, 6,144 and 6,144 rows on the change; main holds two of 6,144. r2 on the change takes both of its points (its message boundary and Keep) and replaces the snapshot it restored from; r1's boundary snapshot stays because r2's prompt shares its prefix, and the store keeps it until the budget needs it.
- **Not read:** the 9B, Claude Code's real traffic, Metal, Mamba-2. A real `Keep` grid of 512 rows (B35's e2e) was not the width here.

## Second slice: cut the prefill at the snapshot points, not on the pass grid (registered 2026-11-15 14:06 PST, before any code; nobara-kv8)

The coordinator's decision of 2026-11-15, on T1's PARKED: keep #1007 a draft and build B35's named follow-up on its branch. T1 found 4,096 of a 6,100-id boundary served because a snapshot falls on a 2,048-row pass boundary.

**The question.** Does a snapshot taken at its point itself, rounded down only to a 64-row block (the snapshot's own unit), serve a rewritten last message at the bar, without moving any other request's bits or cost?

**What changes (recurrent models only; attention models keep the grid).** The prefill is cut at each snapshot point, the message boundary's and Keep's, each rounded down to 64 rows: `decoder/resident_prefill_seed.go:Model.residentPrefillSeedMarked` prefills up to a point as a prefill of its own, takes the snapshot, and goes on, in place of waiting for a pass boundary. A point is dropped, or lowered by a block, when the prefill after it would be shorter than the resident's final-pass minimum (`Model.prefillTailMinRows`). **Keep is cut too, not only the message's:** a request that restores from an off-grid point has no pass boundary before its own Keep, so on the grid it would take no Keep snapshot, and the next turn of an edited conversation would prefill whole. A 2,048-row grid hides this on #1007's own T1 prompt only because 4,096 is on it.

**Why it should hold, and what that rests on.** `TestB35SplitPrefill` (qwen35-tiny): a prompt prefilled in two calls, each pass 16 rows or more, equals one call bit for bit. That is read on a tiny fixture; the gates below read it again at the decoder, on `qwen3next-tiny`, and on the real model through T2's replies.

| | tier | what | bar |
|---|---|---|---|
| **C0** | 0, hash fake with 128-row passes | turn 1: 1,200 ids, message start 1,000, Keep 1,196. Snapshots land at 960 and 1,152 (a 64-row block below each, where the grid gives 896 and 1,152). A turn rewriting the message from id 1,003 reuses 960, and its reply equals a cold run | **red on #1007 first** (it reuses 896). Plants, red: the points rounded up; a cut with no tail minimum (a 3-row final pass) |
| **C1** | 0, the same fake | a turn that restored the off-grid 960 takes its own Keep snapshot, and the next turn restores it | held rows named; **red** with Keep left on the grid |
| **C2** | 0, nobara CUDA, `qwen3next-tiny` | 128-row passes, message start 1,500, Keep 2,396: snapshots at 1,472 and 2,368; the restore at 1,472 leaves the reply, every recurrent buffer and the K/V rows bit-identical to a cold prefill | 0 differing words. Plant (grid-rounded points) is red on the reuse |
| **C3** | 0 | M0 to M3 and the B35 tests still pass; `gate identity -backend cuda` against current main: every family that ran IDENTICAL (a default load sets no scope) | 0 DIFFERENT |
| **T2** | 1, nobara | T1's script and conversation (`served_msgboundary.py`), main (current `origin/main`) against the combined change (#1007 plus this slice), `~/models/qwen3.5-0.8b`, A B B A, one server per cell, the exclusive timing lock, a cell above load 8.0 void and re-run once | below |

**T2's bars, worst of the two pairs. PASS needs every row; PARKED is for Francis; FAIL stops the merge.**

| | PASS | PARKED | FAIL |
|---|---|---|---|
| r3 (last message rewritten): first token, change over main | at most 0.5 | above 0.5, at most 0.7 | above 0.7 |
| r3 rows reused | exactly the smaller of r1's two snapshots, and that is a block of 64 within 63 rows of the boundary (not the 2,048 grid's) | | anything else |
| **r1, r2, r3 replies** (all three, not r3 only: the cut moves every cold request's passes) | each equals main's | | any differs |
| r1 first token (cold), change over main | at most 1.05 | above 1.05, at most 1.10 | above 1.10 |
| r2 first token, and rows reused | at most 1.10 of main's; not fewer rows | | slower than 1.10, or fewer rows |
| **the cold request's added cost, read explicitly from `serve-*.log`:** the boundary snapshot's copy time, and its host MiB | at most 150 ms and at most 175 MiB (#1007's T1 measured 97.6 ms and 115 MiB at 4,096 rows; Keep's 98.1 ms and 163 MiB at 6,144) | above either, at most 250 ms and 250 MiB: **the owner weighs it** | above either |
| r4, the turn after r3 (r3's history, its reply, a new question): rows reused | at least main's r4 reuse | | fewer: a restored turn lost its Keep point |
| peak `VmSwap` | at most 512 MB | | above |

- **Reported, not graded:** r4's first token against main's; snapshots held after each request; Keep's copy time and bytes now that its point moved up.
- **Not read:** the 9B, Claude Code's real traffic, Metal, Mamba-2, a pass width other than 2,048.
- **Void if:** a timed cell outside the lock, the model read from `/srv/models`, a NOT RUN counted as a pass, a bar moved after a reading. **Stopping rule:** the four cells, no early stop. **Cost:** about 3 minutes for the cells, start and finish from `date`.
- **The merge.** PASS: #1007 and this slice go up together, and the coordinator is told (the cold cost is the owner's to weigh where it is above #1007's measured 97.6 ms and 115 MiB, which this registration's PASS band allows up to 150 ms and 175 MiB). PARKED or FAIL: both stay drafts.

### Second slice, readings (2026-11-15, nobara-kv8)

| | reading |
|---|---|
| **C0** | `decoder/kv8_msgboundary_test.go:TestKV8Cut_snapshotsLandOnTheirBlocks`: **red on #1007 first** (snapshots of 896 and 1,152 rows where 960 and 1,216 are wanted). Plants, each red: points rounded up (5 tests); no tail minimum (`TestKV8Cut_theFinalPassKeepsItsMinimum`) |
| **C1** | `TestKV8Cut_aRestoredTurnKeepsItsOwnPoint`: red with Keep left on the grid (a plant: on #1007 the 128-row grid happened to contain the restore point, so C1 is not red on #1007) |
| **C2** | `backend/cuda/kv8a_checkpoint_test.go:TestKV8Msg_cudaRestoreAtTheBoundaryEqualsCold`, `qwen3next-tiny`, passes of 64 and of 128 rows: snapshots at 1,472 and 2,368 (the 128-row grid has neither); turn 2 reuses 1,472; the reply, every recurrent buffer and the K/V rows are bit-identical to a cold prefill's. Plant (pass-grid points) is red on the held rows. Log: `c2-cuda.log` |
| **C3** | `gate identity -backend cuda d44de0aa3 HEAD` at `688809f3c`: **GREEN, 25 IDENTICAL, 0 DIFFERENT, 19 NOT RUN** (the same missing ignored fixtures as before), `identity-cut.log`. Parity refresh: manifest unchanged. gofmt, vet (tagged), pinned staticcheck, citation lint, comment census clean |

**T2: PASS** (14:20 to 14:22 PST; `served-t2/run.log`, `turns.jsonl`). Main `d44de0aa3` against the combined change `688809f3c` (#1007 plus this slice), `~/models/qwen3.5-0.8b`, `--ctx 12288`, one slot, A B B A, the exclusive timing lock, no void cell, peak `VmSwap` 0 MB. The 1-minute load at the cells' start was 4.81, 3.51, 2.71, 2.15 (the first two carry the build that had just finished; the void line is 8.0).

| | main | change | change over main, pair 1 / pair 2 | bar |
|---|---|---|---|---|
| **r3 (last message rewritten): first token** | 7.39 s, 7.39 s | 2.41 s, 2.41 s | **0.326 / 0.326** | at most 0.5 |
| r3 rows reused | 0 | **6,016** (r1's boundary snapshot; the boundary is about 6,044) | | the block within 63 rows of the boundary |
| r1, r2, r3, r4 replies | | **equal to main's, all four, both pairs** | | equal |
| r1 (cold): first token | 7.41 s, 7.39 s | 7.41 s, 7.42 s | 1.000 / 1.004 | at most 1.05 |
| r2: first token, rows | 2.16 s, 6,144 | 0.36 s, 7,360 | 0.166 / 0.160 | at most 1.10, not fewer |
| r4 (the turn after r3): rows | 6,144 | 7,360 | | not fewer |
| **the cold request's boundary snapshot** | | 160 MiB host, 99.2 and 99.5 ms | | at most 175 MiB and 150 ms |

- **The cold cost, read explicitly (the owner weighs it).** #1007 on its own measured 115 MiB and 97.6 ms at 4,096 rows; the same snapshot, now at 6,016 rows, is **160 MiB and 99.2 ms**: the rows grew, the time did not. **A cold request now also takes Keep's snapshot at 7,360 rows, 192 MiB and 114.9 ms, where main takes it at 6,144 rows, 163 MiB and 98 ms.** So a cold request copies **352 MiB and about 214 ms of snapshots against main's 163 MiB and 98 ms**, and its first token did not move (1.000, 1.004): the extra copy is inside a 7.4 s prefill, about 1.5%. The registration graded only the boundary snapshot; the Keep snapshot's growth is the larger part of the extra bytes and is reported here for the owner to weigh.
- **A result the registration did not grade:** cutting Keep took r2 from 6,144 to 7,360 rows reused and 2.16 s to 0.36 s (ratio 0.16), a gain for every recurrent chat turn, on the 2,048-row grid that rounded Keep down by 1,286 rows. Its replies equal main's.
- **The cut rests on the split being bit-identical** (`TestB35SplitPrefill`, C2, and these replies). It is read on a tiny DeltaNet fixture and on one real checkpoint's greedy replies, 24 tokens each, not on the logits of a real model.
- **Not read:** the 9B, Claude Code's traffic, a pass width other than 2,048 on a real model, Metal, Mamba-2.

## Third addendum: the 9B and recorded agent traffic (registered 2026-11-15 15:42 PST, before any run; an RTX 2070 SUPER, 8 GB)

Decided 2026-11-15: the reads #1007 left owed. Both are by day, under the exclusive timing lock, A B B A, one server per cell, a cell above load 8.0 void and re-run once. Metal and Mamba-2 stay owed.

**R1, the 9B.** T2's shape (`served_msgboundary.py`, requests r1 to r4) on `~/models/Qwen3.5-9B-Q4_K_M.gguf` from NVMe, `--ctx 8192` (the prompts are 7,434 to 7,465 ids), one slot. Builds: main before #1007 (`a01046150`, the first parent of the merge) against main with it (`a9824eb2d`; the tree after it differs from that only in docs). **Fit check first**, by the server's own banner (`--fit-only`): if the fit guard refuses the 9B with its snapshots on the 8 GB card, the read stops and says so; no threshold moves, no bypass flag.

| | PASS | PARKED | FAIL |
|---|---|---|---|
| r3 (last message rewritten): first token, change over main | at most 0.5 | above 0.5, at most 0.7 | above 0.7 |
| r3 rows reused | exactly the smaller of r1's two snapshots, a 64-row block within 63 rows of the boundary | | anything else |
| r1 to r4 replies | each equals main's | | any differs |
| r1 (cold): first token, change over main | at most 1.05 | above 1.05, at most 1.10 | above 1.10 |
| r2: first token and rows | at most 1.10 of main's; not fewer rows | | slower than 1.10, or fewer rows |
| r4: rows | not fewer than main's | | fewer |
| peak `VmSwap` | at most 512 MB | | above |

- **The cold cost is read explicitly and the 0.8B's bars are not applied to it:** a 9B snapshot is several times larger per row (T2's 160 MiB and 192 MiB were the 0.8B's), so 175 MiB and 150 ms would fail it by arithmetic, not by effect. The 9B reports each cold request's snapshots (rows, host MiB, copy ms), r1's first-token difference in seconds against main's, and how many cold conversations the auto budget holds against main's. **The graded cost is r1's first token**, with T2's bars above, and its ambiguous band (1.05 to 1.10) goes to the owner.
- **Cost:** about 8 minutes (9B loads, cold prefills of 7.4k ids). Start and finish from `date` when it runs.

**R2, KV0-H's recorded requests.** `replay_kv0h.py` replays KV0-H's recorded request bodies, each to its own route as recorded (chat completions, responses, messages): its 10 opencode and Codex sessions (84 requests) and the Claude Code chain of its 32,768-token check (10 requests), in the study's order through one slot, on `~/models/qwen3.5-0.8b` (same vocabulary as the 9B they were recorded against), `--ctx 24576`, greedy, 8 new tokens, main `a01046150` against `a9824eb2d`. The bodies stay outside the repository (KV0-H's policy); this record carries counts and times only. The server is not restarted inside a cell, so a session's first request meets what the previous session left, as in KV0-H.

- **Limit, stated first:** the 0.8B's replies are not the 9B's that the recorded histories carry, so a follow-up never continues from the live state: what serves it is a snapshot, the end-of-prompt checkpoint, or nothing. This reads the restore paths on real agent prompt shapes; it is not the 9B's traffic and not a rate of any one agent's turns.
- **Counted per follow-up (every request after a session's first), both arms:** what served it (the message-boundary snapshot, another snapshot, the end-of-prompt checkpoint, live state, nothing: `snap=`, `ckpt_restore=` and `reused=` of the request line, and the snapshots the previous request took), its rows reused and its first token.

| | PASS | PARKED | FAIL |
|---|---|---|---|
| every request's reply (the generated ids, `--log-token-ids`) | equals main's | | any differs |
| rows reused, per request | not fewer than main's on any | | fewer on any |
| follow-ups served by a snapshot | not fewer than main's | | fewer |
| first token per follow-up, change over main: the median, and the total | at most 1.05 each | above 1.05, at most 1.10 | above 1.10 |
| peak `VmSwap` | at most 512 MB | | above |

- **Reported, not graded:** the count each source serves on each arm; how many follow-ups the boundary snapshot serves that main served from nothing; snapshot rows, host MiB and ms per request; the budget held.
- **Cost:** 94 requests a cell, prompts up to about 19k ids, so about 25 to 45 minutes for the four cells (cold prefills of up to 19k ids at the 0.8B's rate, main's cells longer). Start and finish from `date` when it runs.
- **Void if:** a timed cell outside the lock, a model read from `/srv/models`, a NOT RUN counted as a pass, a bar moved after a reading. **Stopping rule:** the four cells of each read, no early stop; a refused fit check ends R1 and is reported.

**Amendment, 2026-11-15 15:51 PST, before R2's run (R1 had read PASS).** R2's first-token bar read "at most 1.00" for change over main. A smoke of the script on two of the sessions (28 requests, not a reading, nothing recorded from it) showed that bar cannot hold: with the arms serving all 12 follow-ups identically (same sources, same rows, same replies), the median of per-request ratios read 1.032 and 1.008, because the follow-ups' first tokens are 0.1 to 0.9 s. The bar is now **at most 1.05 for the median and for the total** (the cold-request bar T2 used), PARKED above 1.05 and at most 1.10, FAIL above 1.10. No other clause moves; the replies, the rows and the snapshot counts are exact comparisons and keep their bars.

### Third addendum, readings (2026-11-15, an RTX 2070 SUPER, 8 GB)

**R1, the 9B: PASS** (15:43 to 15:49 PST; `served-9b/run.log`, `turns.jsonl`, `fit-check.log`). `a01046150` against `a9824eb2d` (the binary for the second is that merge's code plus these scripts), `Qwen3.5-9B-Q4_K_M.gguf` from NVMe, `--ctx 8192`, one slot, A B B A, the exclusive lock, load 0.33 to 1.16 at the cells' start, no void cell, peak `VmSwap` 0 MB. The fit check passed: 5,640 MiB on the card, a 1 x 8,192 KV plan, the 6.3 GiB snapshot budget.

| | main | change | change over main, pair 1 / pair 2 | bar |
|---|---|---|---|---|
| r3 (last message rewritten): first token | 35.24 s, 35.25 s | 9.57 s, 9.57 s | **0.272 / 0.272** | at most 0.5 |
| r3 rows reused | 0 | 6,016, the boundary's block (the boundary is about 6,044) | | exactly that block |
| r1 (cold): first token | 35.38 s, 35.23 s | 35.44 s, 35.49 s | 1.002 / 1.007 | at most 1.05 |
| r2: first token, rows | 8.63 s, 6,144 | 1.13 s, 7,360 | 0.130 / 0.130 | at most 1.10, not fewer |
| r4: rows | 6,144 | 7,360 | | not fewer |
| r1 to r4 replies | | equal to main's, both pairs | | equal |

- **The cold cost, reported (the 0.8B's 175 MiB and 150 ms bars were not applied):** main's cold request takes one snapshot, **434 MiB and about 281 ms** (6,144 rows). The change's takes two: 426 MiB and 260 ms at the boundary (6,016 rows), 510 MiB and 300 ms at Keep (7,360 rows): **936 MiB and about 560 ms**. In pair 2 r1's first token was 0.26 s slower than main's (pair 1: 0.06 s), about the size of the extra copy: on the 9B the copy shows on the path. On the 6.3 GiB auto budget a cold request's pair is about 6.7 conversations, against main's 14.5 (the figure for weighing the cost).
- Each request that restored a snapshot also took a snapshot of its own (r2: 514 MiB, 305 ms at 7,424 rows), on both arms' rules; the change's r2 and r4 are 0.13 of main's.

**R2, KV0-H's recorded requests: FAIL by the registered rule** (15:51 to 16:05 PST; `replay-kv0h/run.log`, `requests.jsonl`, the four `serve-*.log` and `tokenids-*.jsonl`). 11 sessions, 94 requests a cell, `qwen3.5-0.8b`, `--ctx 24576`, `a01046150` against `a9824eb2d`, A B B A, the exclusive lock, load 0.65 to 1.07, no void cell, `VmSwap` 0 MB. The bars are the registration's with the 15:51 amendment.

| | main | change |
|---|---|---|
| replies (generated ids), 94 requests, both pairs | | **0 differ** |
| follow-ups (83): first token, change over main, median | | 1.023, 1.011 (bar 1.05) |
| follow-ups: total first token | 168.0 s, 168.6 s | 159.7 s, 159.6 s = **0.951, 0.946** (bar 1.05) |
| follow-ups served by: cold / end-of-prompt checkpoint / snapshot / live state | 4 / 33 / 9 / 37 | 5 / 33 / 8 / 37 |
| requests reusing fewer rows than main | | **1, in both pairs** (bar: none) |
| follow-ups served by a snapshot | 9 | **8** (bar: not fewer) |

**Two clauses fail, in both pairs, so the verdict is FAIL and no bar moves.** They are one request: the second request of KV0-H's opencode `edit` session (7,265 ids). Main restores 6,144 rows and starts in 1.8 s; the change reuses 0 rows and starts in 7.1 s (+5.3 s).

- **The cause, from the logs and the code:** that request follows a 564-id one, so it is its session's first long prompt, and what main restored is a snapshot an earlier session left. Main's snapshot sits at 6,144 because Keep rounds down to a 2,048-row pass; the prompt's system-and-tools preamble, which every opencode session shares, is about that long, so the rounded point happened to fall inside it. The change's snapshots sit at their exact points, which are past the shared preamble, and a recurrent snapshot serves only its own end, so none of them is a prefix of this prompt. The coarse grid was acting as a cross-session preamble snapshot nobody had asked it to be. Recurrent models take no snapshot at the system-and-tools boundary (`decoder/kvsnap.go:Model.kvBoundaryPlan` takes only Keep for them), which is the preamble-boundary mark KV8's plan lists and nothing has built.
- **Everything else the cut changed went the other way:** 18 follow-ups restore more rows than main's (the Claude Code chain's: 16,384 against 17,920, 18,048 and others; Codex's 4,096 against 5,888 to 5,952), and the total first-token time over the 83 follow-ups is 5% lower. The net is a gain with one regression the registration forbids.
- **The message-boundary snapshot, #1007's headline, served none of the 83 follow-ups here.** One request of 94 took one (`requests.jsonl`, `took`), because its mark falls below any prefill that restores from a deeper point, and in these chains almost every follow-up does. The gains above are the Keep cut's. The two D-category cases KV0-H's A5 named (3 ids into a new user message) are a boundary-snapshot case, and this replay does not reach them, since the 0.8B's replies are not the recorded 9B's and no follow-up here starts before its predecessor's last message.
- **Cold cost across the 94 requests (per cell):** the change takes 12 snapshots, 4,082 MiB and 2,593 ms, main 6, 1,890 MiB and 1,218 ms.
- **Not read:** Metal, Mamba-2; the 9B on these recorded requests (the replay is on the 0.8B); an agent's live traffic. The replay's limit stands: no follow-up continues from the 0.8B's own reply.

- **What `replay-kv0h/` keeps:** `requests.jsonl` carries each request's counters and times and a SHA-256 of its generated ids, not the ids. The token-id logs the servers wrote (generated ids and prompt hashes of KV0-H's recorded requests) are not kept, following KV0-H's policy that ids and bodies stay outside the repository. The replies' equality was graded on the ids before they were reduced to hashes.

## Fourth addendum: a snapshot at the end of the system-and-tools preamble, for recurrent models (registered 2026-11-15 16:12 PST, before any code; an RTX 2070 SUPER, 8 GB)

Decided 2026-11-15: the replay's one failing request is a regression on main since #1007, so the preamble mark KV8's plan lists is built, in KV8's scope. On a recurrent model the decoder already receives `KVScope.Boundary` (the ids before the first conversation turn, found by serve from the template) and ignores it.

**The question.** Does a snapshot at the end of the shared preamble serve a session's first long request after another session's, without costing a request that does not need it?

**What changes (recurrent models, CUDA).** A cold prefill is also cut at `Boundary` rounded down to a 64-row block, below the message boundary's and Keep's cuts (the same split, `decoder/resident_prefill_seed.go:Model.residentPrefillSeedMarked`), and keeps a snapshot there when none is held at those rows. That snapshot is **kept apart from the rule that replaces a shorter snapshot of the same prefix with a longer one**: a request's own deeper snapshots, and a later session's, must not replace it, and the snapshot a request restored it from is not dropped when the request takes deeper ones. It leaves the store by the budget's least-recently-used rule only. The prefix is shared across sessions, so one snapshot is taken per distinct preamble, by the first cold request that reaches it.

| | tier | what | bar |
|---|---|---|---|
| **P0** | 0, hash fake, 128-row passes | turn 1 (1,236 ids, preamble 700, message start 1,000, Keep 1,230) leaves snapshots at 640, 960 and 1,216; a second conversation that shares the first 700 ids and then differs restores 640 and its reply equals a cold run; after a third request the 640 snapshot is still held; a request that restored 640 does not drop it | **red on main first** (main holds no snapshot at 640, so the second conversation reuses 0). Plants, each red: the preamble snapshot replaced by a deeper one; dropped by the request that restored it |
| **P1** | 0, hash fake | no preamble snapshot when one is held at those rows; none below `kvSnapMinShared`; the store's budget with three snapshots (room for two keeps the two deepest, none takes none); a model whose snapshots do not carry its state takes none | exact rows and bytes |
| **P2** | 0, CUDA, `qwen3next-tiny` | 128-row passes: the restore at the preamble's block leaves the reply, every recurrent buffer and the K/V rows bit-identical to a cold prefill | 0 differing words; plant (the state not restored) red |
| **P3** | 0 | M0 to M3, C0 to C2 and the B35 tests still pass; `gate identity -backend cuda` against current main: every family that ran IDENTICAL | 0 DIFFERENT |
| **T3a** | 1, replay | `replay_kv0h.py` as R2, main (`69b05e1a4`) against the change, the same clauses and bars | PASS needs every R2 clause, the failing request included |
| **T3b** | 1, `served_msgboundary.py` | r1 to r4 as T2 on `qwen3.5-0.8b` (ctx 12,288) and on the 9B (ctx 8,192, `--report-cost-only`), A B B A, bars as T2 and the 9B's registration | every clause as there |

- **The cold cost is read explicitly, per snapshot (rows, host MiB, copy ms), per cold request and per cell,** against main's. A request pays the preamble snapshot only when none is held, so the cost lands once per distinct preamble; the first-token bars (r1 at most 1.05, PARKED to 1.10) grade it, and the ambiguous band goes to the owner. The 0.8B's 175 MiB and 150 ms bars apply to the boundary snapshot as in T2; the preamble's size is reported.
- **Void if:** a timed cell outside the exclusive lock, a NOT RUN counted as a pass, a bar moved after a reading. **Stopping rule:** the four cells of each read, no early stop. **Cost:** replay about 15 minutes, the 0.8B about 3, the 9B about 8; times from `date` when they run.

### Fourth addendum, readings (2026-11-15, an RTX 2070 SUPER, 8 GB)

| | reading |
|---|---|
| **P0, P1** | `decoder/kv8_msgboundary_test.go:TestKV8Preamble_aSecondConversationRestoresTheSharedPreamble` and the other `TestKV8Preamble` tests beside it: **red on main first** (snapshots of 960 and 1,216 where 640, 960 and 1,216 are wanted). Plants, each red: a deeper snapshot replacing the preamble's; the request that restored it dropping it; the covering-search replacement. The budget test (room for three, two, none) and the held-once and no-state cases pass |
| **P2** | `backend/cuda/kv8a_checkpoint_test.go:TestKV8Preamble_cudaRestoreAtTheSharedPreambleEqualsCold`, `qwen3next-tiny`, 128-row passes: turn 1 leaves 640, 1,472 and 2,368 rows; a second conversation sharing 700 ids reuses 640, and its reply, every recurrent buffer and the K/V rows are bit-identical to a cold prefill's. Plant (the conv window not restored) red. `p2-cuda.log` |
| **P3** | M0 to M3, C0 to C2 and the B35 tests pass; `gate identity -backend cuda` against current main: **GREEN, 25 IDENTICAL, 0 DIFFERENT, 19 NOT RUN** (the missing ignored fixtures of this worktree), `identity-preamble.log`. Parity refresh: manifest unchanged. gofmt, tagged vet, staticcheck, citation lint, comment census clean |

**T3a, the replay: PASS** (16:21 to 16:31 PST, `replay-preamble/run.log`, `requests.jsonl`). Main `69b05e1a4` (with #1007) against the change, the same 94 requests, A B B A, the exclusive lock, no swap. The first cell started at load 6.12 (my build had just finished; the void line is 8.0), the others at 0.93 to 1.14.

| | main | change | bar |
|---|---|---|---|
| replies, 94 requests, both pairs | | 0 differ | equal |
| requests reusing fewer rows than main | | **0** | none |
| follow-ups served by a snapshot | 8 | 9 plus **3 served by the preamble snapshot** | not fewer |
| follow-ups served cold | 5 | 1 | |
| follow-ups' total first token | 158.4 s, 159.6 s | 83.8 s, 84.0 s = **0.529, 0.526** | at most 1.05 |
| follow-ups' median per-request ratio | | 1.000, 1.000 | at most 1.05 |

The request that failed R2 (the second of the opencode edit session) is now served from a snapshot. The 0.5 total is not the preamble snapshot alone: against the earlier R2 cells, main here is 158 s and the change 84 s, and the 3 requests the preamble snapshot serves were cold, 6 to 7 s each, on main.

**T3b, the served r1 to r4: FAIL by its registration, and the registration is the flaw.** `served-t3b-08b/` and `served-t3b-9b/`, the same script and bars as T2 and the 9B's, run against `69b05e1a4` as main. Both arms already carry #1007's boundary snapshot, so r3 is served from it on both: **r3's first token is 0.992 to 1.005 of main's on the 0.8B and 0.997 to 0.999 on the 9B, the registered bar was at most 0.5, and the script's "r3 reuses exactly the smaller of r1's two snapshots" clause now names the preamble's snapshot, not the boundary's.** Both clauses compare the change with a build that already has the boundary snapshot, so a change that does not touch r3 cannot meet them. Nothing moves: T3b is recorded FAIL, as written. Its bars were copied from T2 against a main that already has #1007, so it tests r3, which this change does not touch; they could not apply. What the runs do show:

| | 0.8B | 9B |
|---|---|---|
| r3 rows reused | 6,016 both arms, the boundary's block | 6,016 both arms |
| r1 (cold) first token, change over main | 0.995 / 1.002 | 1.013 / 1.011 |
| r2 and r4 first token, change over main | 0.99 to 1.08 (0.33 to 0.36 s, noise) | 1.00 to 1.03 |
| r2, r4 rows reused | 7,360 both arms | 7,360 both arms |
| replies r1 to r4 | equal | equal |
| preamble snapshot (taken by r1, the first cold request of the preamble) | 2,752 rows, 84 MiB, 55 to 82 ms | 2,752 rows, **222 MiB, 140 to 161 ms** |
| boundary / Keep snapshots (as T2 and R1) | 160 MiB, 121 to 126 ms / 192 MiB | 426 MiB, 253 to 274 ms / 510 MiB |

**The cold cost, as asked.**
- **First cold request of a preamble:** three snapshots. On the 0.8B that is 436 MiB and about 300 ms, on the 9B **1,158 MiB and about 740 ms**, against main's two (352 MiB and about 214 ms; 936 MiB and about 560 ms). The preamble's snapshot adds **84 MiB and about 55 to 82 ms (0.8B), 222 MiB and 140 to 161 ms (9B)**.
- **Later cold requests with the preamble held:** the same two snapshots as main's.
- **First token:** r1 on the 9B is 1.1% to 1.3% slower than main's (0.4 s of 35.6 s); on the 0.8B it did not move.
- **Over the replay's 94 requests, per cell:** 14 snapshots, 4,623 MiB and 2,904 ms against main's 12, 4,082 MiB and 2,576 ms (+13% bytes, +13% copy time).
- **The budget:** the preamble snapshot is shared, so its bytes are taken once per distinct preamble and dropped only by the budget's least-recently-used rule.

**What a correct T3b would ask:** r1 to r4 not slower and not fewer rows than a main that has #1007 (no-regression bars), and the r3 gain is T2's and R1's reading against a build without #1007. That is a new registration, made after this reading; it is not run here.

## T3b': the served r1 to r4 as a no-regression read (registered 2026-11-15 16:42 PST, before it runs; an RTX 2070 SUPER, 8 GB)

Decided 2026-11-15, replacing T3b's bars (above, recorded FAIL as written). **The question:** does the preamble snapshot change anything for a request that does not need it? Main `69b05e1a4` (with #1007; the Go tree of `origin/main` is the same) against `a38e9d697`'s code (this branch's, the Go tree unchanged since), `served_msgboundary.py --no-regression`, r1 to r4 as T2, A B B A, one server per cell, the exclusive timing lock, a cell above load 8.0 void and re-run once, on `qwen3.5-0.8b` (`--ctx 12288`) and on the 9B (`--ctx 8192`, fit-checked as R1 was, `--report-cost-only`).

| | PASS | PARKED (the ambiguous band; the owner weighs it) | FAIL |
|---|---|---|---|
| r3 first token, change over main | 0.98 to 1.02 | 0.95 to 0.98, or 1.02 to 1.05 | outside 0.95 to 1.05 |
| r3 rows reused | equal to main's | | different |
| r1 (cold) first token | at most 1.05 | above 1.05, at most 1.10 | above 1.10 |
| r2 and r4 first token | at most 1.05, or at most 0.03 s slower | above that, at most 1.10 or 0.06 s slower | beyond |
| r2 and r4 rows reused | not fewer than main's | | fewer |
| replies r1 to r4 | equal to main's | | any differs |
| peak `VmSwap` | at most 512 MB | | above |

- **The 0.03 s allowance on r2 and r4** is a noise floor, stated here before the run: their first tokens are 0.3 s on the 0.8B (1.1 s on the 9B), and T3b's own cells read 0.33 against 0.35 s (1.079 as a ratio) with nothing different between the arms. A ratio bar alone would grade that noise.
- **Worst of the two pairs, on each model; PASS needs both models.** The cold cost is reported as in T3b: the preamble snapshot's rows, host MiB and copy ms, and r1's first-token difference in seconds.
- **Void if:** a timed cell outside the lock, a NOT RUN counted as a pass, a bar moved after a reading. **Stopping rule:** the four cells per model, no early stop. **Cost:** about 2 minutes on the 0.8B and 6 on the 9B; times from `date` when it runs.

### T3b', readings (2026-11-15, 16:42 to 16:48 PST; `served-t3bp-08b/`, `served-t3bp-9b/`)

Main `69b05e1a4` against the branch's code, A B B A, the exclusive lock, no void cell, `VmSwap` 0 MB, one slot. The 9B's fit check passed as R1's did (5,640 MiB on the card, a 1 x 8,192 KV plan). Replies equal and rows reused equal on every request of both models.

| first token, change over main (pair 1 / pair 2) | 0.8B | 9B | bar |
|---|---|---|---|
| r3 | **1.023** (+0.05 s) / 1.004 | 1.001 / 1.001 | 0.98 to 1.02 |
| r1 (cold) | 0.996 / 0.994 | 1.016 (+0.56 s) / 1.014 (+0.50 s) | at most 1.05 |
| r2 | 1.011 / 1.015 | 1.013 / 1.001 | at most 1.05 or 0.03 s |
| r4 | 0.985 / 1.066 (+0.02 s) | 1.004 / 0.953 | at most 1.05 or 0.03 s |
| r3 rows | 6,016 both arms | 6,016 both arms | equal |

**The verdict is PARKED, by the registered rule: the 9B passes every clause, and on the 0.8B r3 in pair 1 reads 1.023 (2.37 s to 2.42 s), above the 1.02 PASS edge and inside the 0.95 to 1.05 ambiguous band.** The other pair reads 1.004 and the 9B's r3 reads 1.001 in both pairs, with the same rows and the same replies. Nothing moves. The one reading outside the PASS band is a 0.05 s difference on a 2.4 s request that the preamble snapshot does not touch (r3 restores the boundary snapshot on both arms); the registration's allowance for sub-second requests (0.03 s) did not cover it. The decision goes to the owner.

**Decided 2026-11-15 (owner): T3b' is taken as a pass.** The one reading in the ambiguous band is a 0.05 s difference on a request the preamble snapshot does not serve, with the other pair at 1.004 and the 9B at 1.001 in both; the bars above are unchanged. #1014 merged on this.

**The cold cost, as in T3b, run again:** the preamble snapshot is 2,752 rows, **84 MiB and 77 ms on the 0.8B, 222 MiB and 165 ms on the 9B**; the boundary and Keep snapshots are as before (0.8B 160 and 192 MiB, 98 to 142 ms; 9B 426 and 510 MiB, 277 to 321 ms). r1's first token on the 9B is 0.50 to 0.56 s slower than main's (1.4% to 1.6%) and on the 0.8B 0.03 to 0.04 s faster.

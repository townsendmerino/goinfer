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

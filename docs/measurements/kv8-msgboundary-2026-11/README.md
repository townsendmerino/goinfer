# KV8's message-boundary checkpoint on CUDA (registered 2026-11-15 13:25 PST, before any code; nobara-kv8)

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

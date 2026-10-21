# KV9 step 0: what host-block storage would save on the CPU (exploratory, 2026-10-14)

**Exploratory.** One run per cell, by day, on a Mac already under memory pressure. Nothing was timed.

**What D2 asked for.** `docs/tasks/task-kv-architecture-2026-10.md` defers KV9 by D2: the CPU path joins only
through KV2's copies, and moving CPU storage to host blocks (§5.14 (b)) waits on a measured multi-session memory
case. This is that measurement. It covers:

- CPU `serve` on `main` plus KV2's CPU copies;
- four sessions sharing an agent-sized preamble, against one session;
- the 0.5B and the 1.5B.

No storage code was changed.

## Method

**Build and machine.**

- `serve` was built from branch `kv2-2026-10` at `b821bbd9`, which is `main` plus KV2. The branch was used read-only.
- Machine: MacBook Pro, M1 Pro, 16 GB, macOS.
- Run under `-backend cpu`, int4 weights from the CPU `.giw` sidecar serve builds, and the default KV precision on the
  CPU (f32).

**Models.** The bench GGUFs, with KV bytes per position from the published geometry. serve's own fit line agrees:
0.8 GB at 32,768 positions for the 0.5B, 1.8 GB for the 1.5B.

| model | layers × KV heads × head dim | f32 K+V per position |
|---|---|---|
| Qwen2.5-Coder-0.5B q4_k_m | 24 × 2 × 64 | 24,576 B |
| Qwen2.5-Coder-1.5B q4_k_m | 28 × 2 × 128 | 57,344 B |

**Prompts.**

- The preamble is text from this repository at `e42db592`, cut to a character budget. serve counted 5,778 and 20,193
  prompt tokens.
- Each session sends the same system preamble and its own one-line user message, with `max_tokens` 8 at temperature 0.

**Readings.**

- After each session: the server's `footprint` (phys_footprint, which counts compressed pages) and `ps` RSS.
  footprint is the figure used; RSS drops as macOS compresses pages.
- `usage.prefill_reused_tokens` shows what each session reused.
- A guard between sessions stops a cell before it pushes the machine further into swap.

**Harness.** `kv9_mem.py` (this directory) runs the cells, and `kv9_table.py` tabulates its output. The JSON lines of
every run are committed beside them:

- `run1-sequential.jsonl`;
- `run2-ladder8.jsonl`;
- `run3-pair-start.jsonl`;
- the two settle passes, `run-settle-*.jsonl`.

Server logs are on the Mac under `~/goinfer-logs/kv9-2026-10/`. Units below are MiB.

## Finding 1: sessions sent one after another never hold the preamble twice

In runs 1 and 2, the sessions were sent one at a time with `-max-concurrent 1`. Every later session reused 5,761 rows,
not a whole number of 64-row blocks, and the footprint did not move:

- the 0.5B went from 1,030 to 1,057 across eight sessions;
- the 1.5B went from 2,050 to 2,051.

KV2's copy path never ran. `internal/serveapp/sessions.go:pickSession` gives a new conversation the only idle session
holding the preamble, rewound in place, whenever the prompt keeps more of that session than it discards. Each
conversation took over the previous one's session.

A probe in run 3's first attempt shows what that costs a client. Conversation 1 came back with 5,801 prompt tokens and
5,761 reused: its own tail had been discarded.

So on CPU serve as built, separate sessions holding the same preamble arise in two ways:

- **Concurrency:** a busy session is never taken, so concurrent requests each get their own.
- **Long tails:** each conversation's own tail exceeds the preamble.

The second is unreachable for a 20K preamble on a 32K-context model. Below that, interleaved conversations keep the
preamble once and re-prefill each other's tails.

## Finding 2: four sessions, measured

Run 3 started sessions 1 and 2 concurrently with `-max-concurrent 4`: two cold prefills, two sessions. Sessions 3 and
4 then came one at a time.

- Each copied 5,248 rows from an idle session. KV2's `CapHit` leaves the last 530 rows to a fast-class pass of the
  session's own.
- A final probe continuing conversation 1 reused 5,786 of 5,801 (0.5B) and 5,780 of 5,795 (1.5B): it kept its own
  rows.

| cell | 1 session (run 2) | 2 sessions, concurrent cold prefills | 3 sessions | 4 sessions |
|---|---|---|---|---|
| 0.5B, 5,778 tokens | 1,030 | 1,878 | 2,158 | 2,159 |
| 1.5B, 5,778 tokens | 2,050 | 3,533 | 4,068 (swap +417) | stopped by the guard |
| 0.5B, 20,193 tokens | 3,129 (run 1; swap +1,265) | not run | not run | not run |
| 1.5B, 20,193 tokens | not run | not run | not run | not run |

**The 20K cells.**

- The 0.5B's single cold prefill of 20,193 tokens pushed the server to 3,129 MiB and grew swap by 1.27 GB. The guard
  stopped the cell before session 2.
- The 1.5B cell was stopped before its first request.
- Four 20K sessions need at least two concurrent 20K prefills (Finding 1), about 6 GB on the 0.5B alone. That does not
  fit on this Mac by day.

## Finding 3: the KV is a minority of the footprint; the cold prefill's high-water sets it

- **One session of the 0.5B at 6K:** a footprint of 1,030 MiB, of which the session's KV is 135 MiB.
- **At 20K:** 3,129 MiB for 473 MiB of KV.

The rest is what the batched prefill allocated, which Go's heap keeps:

- the footprint did not fall in 60 s of idle after the request (`run-settle-default.jsonl`);
- KV copied afterwards lands in that retained heap without moving it. In run 3's 0.5B cell, session 4's copy of
  123 MiB added 1 MiB.

`GOGC=10` in serve's environment cut the one-session footprint from 1,039 to 684 MiB (`run-settle-gogc10.jsonl`). It
did not make later copies visible either.

## What host-block storage would save

KV9 would store the preamble's whole 64-row blocks once, plus each session's own rows from its first unshared block.
By geometry, at f32, with four sessions:

| cell | KV per session | 4 sessions today | with host blocks | saved | against the measured 4-session footprint |
|---|---|---|---|---|---|
| 0.5B, 5,778 | 135 | 542 | 137 | 405 | 19% of 2,159 |
| 1.5B, 5,778 | 316 | 1,264 | 319 | 945 | not measured at 4 sessions; 3 sessions read 4,068 |
| 0.5B, 20,193 | 473 | 1,893 | 476 | 1,418 | not measured |
| 1.5B, 20,193 | 1,104 | 4,417 | 1,110 | 3,308 | not measured |
| 7B, 19,692 (§5.14's case; 114,688 B per position) | 2,154 | 8,615 | 2,168 | 6,447 | not measured |

Two comparisons put these numbers in context:

- `-kv i8`, available today and lossy, holds each session's KV in about a quarter of f32's bytes. Four sessions at i8
  hold about what one session holds at f32, which is what host blocks would hold at f32.
- KV9 does not touch the cold-prefill high-water (Finding 3). In every measured cell, that high-water is larger than
  the savings.

## Recommendation

**Close KV9 with KV2's copies as enough, and keep D2 as decided.** The memory case D2 asked for does not hold up on
what was measured:

- **The saving is modest where it was measured.** On the 0.5B at 6K, four sessions' KV is 19% of the four-session
  footprint, and the saving is 405 MiB of 2,159.
- **It is large only where the workload is not one the CPU path serves.**
  - The 7B case saves 6.3 GiB by geometry. But a 20K cold prefill on the CPU took 265 s on the 0.5B, and four agents
    on a 7B at 20K is not a CPU workload anyone runs.
  - At 20K, a second session holding the preamble needs concurrent cold prefills, whose own high-water is several
    times the saving (Finding 3).
- **Where memory is the problem, `-kv i8` already gives the same four-session total**, lossy and without code.

Two smaller items came out of the measurement. Each is worth more on the CPU path than KV9:

1. **The CPU cold prefill's memory high-water.** At 20K, the 0.5B's prefill reached 3.1 GiB of footprint for 0.47 GiB
   of KV, Go kept it afterwards, and it pushed this Mac 1.27 GB into swap. Two directions:
   - bound it, with a chunked CPU prefill;
   - return it after a long prefill, as the GPU residency path's `releaseHostMemory` does after a build.

   This is what sets a CPU server's footprint.
2. **The in-place takeover between interleaved conversations (Finding 1).** Two agents taking turns on CPU serve
   discard each other's tails and re-prefill them on every switch: up to the whole conversation past the preamble.

   KV2's copy makes the other trade possible: a fresh session with the preamble copied in, keeping both tails, at one
   preamble's memory per conversation. That is a choice of rule for the owner, and it is the one place host blocks
   would give both. If the owner wants both tails kept at one preamble's memory, that is the case to reopen KV9 on.

Reopen KV9 on a measured workload in which, after item 1 bounds the prefill high-water, several sessions' KV is still
most of a CPU server's memory.

## Not measured

- Four sessions at 20K, on either model.
- The 1.5B's fourth session at 6K: the swap guard stopped it.
- Any Linux box.
- Any precision other than f32.

The machine started every run with 3.9 to 5.4 GB of swap already in use from the desktop session. Footprint is per
process and was read on this server only.

Two CPU sidecars, which serve writes as a cache, are left in `~/models` on the Mac:
`qwen2.5-coder-0.5b-instruct-q4_k_m.int4.e4h.cpu-arm64.v16.giw` (361 MB) and the 1.5B's (1.0 GB).

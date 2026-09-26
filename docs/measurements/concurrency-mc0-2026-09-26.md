# MC0 — the resident-KV thrash is real on Metal; the CPU "control" thrashed too, through an LRU bug, now fixed (2026-09-26)

MC0 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md). The question is whether two
interleaved conversations evict each other's KV, so that every turn re-prefills its whole history.

**Verdict: confirmed, by the pre-registered reading, once the control's own bug was fixed.**
- **Metal** (the GPU-resident path, one KV per model): at 2 clients, every turn after the first reuses **7 tokens**,
  the chat template's shared lead. At 1 client the same turns reuse their full history (170 → 779 tokens). The
  aggregate is **0.88×** the 1-client figure (65.5 → 57.7 tok/s). MC1's premise holds.
- **CPU**, the control that was supposed to show no thrash, thrashed as well. Its session LRU reused 7 tokens per
  turn at 2 clients and ran at **0.69×** its 1-client aggregate. The cause is a bug in `bestExtend`, introduced
  2026-09-23 by `e1c867f6` (L-15/P-18). It is fixed here: with the fix the CPU control reuses its full history at 2
  clients and holds **1.00×** (43.1 → 43.3 tok/s). **The fix alone is 1.58× on 2-client CPU serving (27.4 → 43.3
  tok/s).**

## Setup

- M1 Pro 16 GB, macOS 26.6.2.
- qwen2.5-coder-1.5b-instruct q4_k_m from `~/models`, W7's checkpoint.
- `scripts/bench_w7_plain.py` (extended for MC0 in `36a8db24`: per-turn `usage.prefill_reused_tokens`, `--backend`,
  `--engines`). The workload is 6 plain turns per client, 128 tokens per turn, greedy. There is a fresh server per
  (backend, client count), and each client's first turn carries a nonce.
- goinfer `serve`, `-quant int4`, `-kv-sessions` at its default of 4.
  - Run 1 (`mc0.json`, 14:42–14:52 PDT) used `serve-metal-622b1f9b`: R18-wired HEAD, without the fix.
  - Run 2 (`mc0b.json`, 14:55–15:05 PDT) used the same tree plus the `pickSession` fix.
- Both runs were idle-gated at load1 ≤ 2.0 (1.98 and 1.75 at start).
- One sample per cell. The reuse counts are deterministic, so they need no reps; the aggregates are single samples.
- Raw: [`concurrency-mc0-2026-09-26/`](concurrency-mc0-2026-09-26/) (both JSONs, both logs, both run scripts).

## Results

| backend | build | 1 client | 2 clients | 2 ÷ 1 | reused on turns 2–6 at 2 clients |
|---|---|---|---|---|---|
| Metal | without fix | 64.64 tok/s | 57.32 | 0.887× | 7, 7, 7, 7, 7 (both clients) |
| Metal | with fix | 65.54 | 57.72 | **0.881×** | 7, 7, 7, 7, 7 (both clients) |
| CPU | without fix | 39.91 | 27.42 | **0.687×** | 7, 7, 7, 7, 7 (both clients) |
| CPU | with fix | 43.10 | 43.28 | **1.004×** | 171, 325, 480, 632, 780 (both clients) |

At 1 client, every backend and build reuses the full prior history: 170, 324, 479, 631, 779.

Against MC0's pre-registered reading:
- Metal at 2 clients reuses near zero while the 1-client run reuses the full history: **yes**.
- The CPU control's 2-client aggregate is within 0.85–1.0× of its 1-client figure: **no** in run 1 (0.69×), because the
  control itself was thrashing; **yes** in run 2 (1.00×).

So the thrash is confirmed, and MC1 goes ahead. The control became valid only after MC0 found and fixed its bug. That
is recorded here rather than folded silently into the verdict.

## The CPU bug

`sessionLRU.acquire` (`internal/serveapp/sessions.go`) reuses the session `bestExtend` picks.
- `bestExtend` accepts a candidate whose common prefix with the prompt beats the candidate's "floor": the longest
  prefix it shares with any *other* resident session. That is what stops a shared system-prompt preamble counting as
  reuse.
- With **one** resident session there are no others, so the floor is 0 and a 7-token preamble match qualifies.
- So conversation B's first turn took A's session and rewound it to the 7-token lead. A's next turn took it back the
  same way. The LRU never grew past one session, although `-kv-sessions` was 4, and every turn re-prefilled its
  whole history.

The previous whole-containment rule could not do this: A's full token list is never a prefix of B's prompt. It came in
with `e1c867f6` (2026-09-23, L-15/P-18: longest common prefix instead of whole containment), and it was live on
`main` for three days. The concurrency task doc, filed that same morning, read the CPU path as thrash-free from the
code, which was true of the rule it replaced.

**Fix:** `pickSession`. With room in the LRU for a fresh session, a candidate whose reuse would throw away more of it
than it keeps goes to a fresh session instead:
- A fresh session costs only re-prefilling the shared lead.
- Truncating costs the other conversation its history.
- P-18's partial matches, a stop-string tail or an edited last message, keep most of their session and still reuse.
- With the LRU full, reusing the candidate *is* the eviction, so behaviour there is unchanged.

The tests:
- `TestPickSession_sparePreambleGoesFresh` failed on the old logic: it would have truncated A's 170 tokens to 7.
- `TestPickSession_keepsP18Reuse` pins the P-18 cases.
- The existing `TestBestExtend*` tests are unchanged and pass.

## What this means for the rest of the doc

- **MC1's size of prize on this workload is ~12% at 2 clients (57.7 → ~65 tok/s),** not W7's 40%. W7 measured
  60.08 → 36.14 on 2026-09-19. Metal's prefill has become much faster since, so re-prefilling a ~800-token history
  costs less than it did. That is a cross-session comparison, so it gets no more precision than "much less".
- The prize grows with history length. It is largest for the owner's decision-1 case, parallel agents whose turns
  carry thousands of tokens of tool output, which is where MC1 earns its keep. Re-running MC0's Metal cells with a
  long-context workload before MC1 ships would size that. It is not done here.
- At 4 clients a single slot misses on every turn, as it does at 2. MC1's 4-client gate still has to be measured.

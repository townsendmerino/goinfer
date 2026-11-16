# KV8's owed read: Claude Code's live traffic through the served Messages API (registered 2026-11-16 09:42 PST, before any recording; a 16-core amd64 box, 62 GB RAM, one 8 GB RTX 2070 SUPER)

A read, not a merge bar. Plan of record: the KV8 row in `docs/tasks/task-kv-architecture-2026-10.md` ("Owed: Claude Code's live traffic").

## The question

In a short real Claude Code session against goinfer, does any request **miss a reuse it should get**: reuse 0 rows when an earlier request in the session shared a long prefix with it?

## Instrument

- **Server:** `serve` built at `cc96d256a` (main), `--backend cuda --ctx 32768 --kv-sessions 1 --log-requests --log-token-ids`, `~/models/qwen3.5-0.8b` from local NVMe (never `/srv/models`), served as one model.
  Fit-checked first by the server's own banner. **The 9B was fit-checked and not used:** at 32,768 it loads (7,142 MiB on the card) but keeps no device checkpoint ("50.2 MiB needed beside a 624 MiB reserve, 643 MiB free"), so a reuse read on it would measure that absence.
  The window is 32,768 because Claude Code's first request is about 19.7k tokens (`docs/integrations/claude-code.md`).
- **Client:** Claude Code (`~/.local/bin/claude`, 2.1.322), headless `-p` turns of one session (`--continue`), in a **fresh empty HOME** and a temporary project directory holding a 2-file toy project (`calc.py` with an `add` that subtracts, and its test). The environment
  is the repo's recipe: `ANTHROPIC_BASE_URL` at a local recording proxy that forwards to the server, `ANTHROPIC_AUTH_TOKEN=goinfer` (the placeholder the server ignores, not a credential), every model slot at the served id, `CLAUDE_CODE_ATTRIBUTION_HEADER=0`,
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768`. **No Anthropic key or token is used or reachable** (the HOME has none). Three prompts: read the file and say what `add` does; fix the bug with the smallest edit; run the test and report.
- **Recording:** a local proxy (`cc_record_proxy.py`) saves each `/v1/messages` body, in order, **outside the repository** (`~/goinfer-bench/cc-live-2026-11-16/`). Only counts are committed: per request, the route, the prompt ids, rows reused, which restore served it and its first token.
- **Replay:** every recorded request, in order, through one slot on a cold server (the recorded model name, greedy, 8 new tokens, as KV0-H's replay does: `docs/measurements/kv8-msgboundary-2026-11/replay_kv0h.py`'s `Server`, `replay` and `parse`), **two passes (A A)** under the exclusive timing lock, the 1-minute load and available memory recorded at each start, a pass above 8.0 load void and re-run. A B B A needs a second build to be B, and the coordinator waived main against itself.
  The live pass's own server log is a third reading.

## What is reported, per request

Route, prompt ids, **rows reused**, **what served it** (a host snapshot of N rows, taken by an earlier request as its m-th snapshot; the device checkpoint; the slot's own rows; or none), first token, the snapshots the request took (rows, host MiB, copy ms), and **lcp**: the leading ids it shares with the most similar earlier request.

## Rule (a miss), registered

- **MISS:** a request that reuses 0 rows while an earlier request in the session shares at least 2,048 leading ids with it (`kvSnapMinShared`'s order; a prefix shorter than that is below what the store keeps).
- **SHORT:** a request that reuses fewer rows than `lcp` rounded down to 64 by 1,024 or more while a snapshot or the slot held a deeper point that the request's prefix covers. Reported, not a miss: a recurrent state restores only at a point it was kept.
- **Not a miss:** a request the live session itself changed the preamble of (the first request of a new session, a subagent or title request on another prompt).
- **Void if:** a replay pass outside the exclusive lock, a request not replayed, the model read from `/srv/models`, a pass above the load bar and not re-run.
- **Ambiguous:** a request whose earlier sharer has `lcp` between 1,024 and 2,048: listed, not graded.

| outcome | meaning |
|---|---|
| no MISS in either pass | nothing owed from live traffic on this model; the SHORT list is the next place to look |
| one or more MISS | each is a task: the request, what the earlier request held, and why no snapshot covered it |
| a session of fewer than 10 requests | NOT ENOUGH: a fourth prompt is added once, before any replay, and the registration says so |

**Stopping rule:** one recorded session, two replay passes. **Cost** (stated before the run): recording about 5 to 15 minutes (a 20k-id first request is about 20 s cold on the 0.8B, and tool-calling turns are short), each replay pass about 5 to 10 minutes; **about 25 to 40 minutes of card time in all**, started about 09:50 PST.

## Amendment, 2026-11-16 09:45 PST (before any replay)

The first recording, on `qwen3.5-0.8b` at 32,768, is **NOT ENOUGH** by the table above: 4 requests (11k ids each, 12 tools), and the model answered in text and never called a tool, so no file was read or edited. It is kept (counts only) and not replayed as the session.
The session is recorded again on **`~/models/Qwen3.5-9B-Q4_K_M.gguf` at `--ctx 24576`**, fit-checked at 09:44 by the server's own banner: 6,680 MiB on the card, a **50.2 MiB device checkpoint kept**, 1.5 GB of KV, swap guard armed at its defaults. 24,576 holds this client's requests here (11k ids with 12 tools, `-p` mode)
and its compaction point (the window less a 4,096 output cap) is not reached by a short session. The 9B at 32,768 stays refused for this read (no checkpoint kept). Every other line above stands, including the rule and the outcomes.

## Readings, 2026-11-16

Build `cc96d256a` (main), `qwen3.5-0.8b` for the first recording and `Qwen3.5-9B-Q4_K_M.gguf` (local NVMe) for the session read, Claude Code 2.1.322 in a fresh empty HOME, driver 595.112.07 as anchored. Recorded bodies stay in a private directory outside the repository; what is committed is `replay.log` (the table) and `requests.jsonl` (numbers only).

- **Recording** (09:44 to 09:51 PST, `Qwen3.5-9B` at `--ctx 24576`): three headless turns of one session, 13 `/v1/messages` requests of 11,188 to 13,873 ids with 12 tools. The model read `calc.py`, fixed the bug (`a - b` became `a + b`) and ran the test; the third turn hit its 8-turn cap on a loop, which does not matter here. The first recording on the 0.8B was 4 requests with no tool call (NOT ENOUGH, above).
- **Structure of the session** (by comparing the requests' system blocks, tools and messages, not their text): six conversations that share the system-and-tools preamble (8,896 ids) and differ from their first message on: requests 1 to 3, 4 and 5, 6 to 8, 9 and 10, 11 and 12, and 13. Within a conversation each request continues the last.
- **Replay:** two cold passes (A1 09:52:44, A2 09:57:02) under the exclusive lock, load 0.22 and 1.05, about 61 GB available, peak `VmSwap` 0 MB, no void pass. The recording server's own log is read as pass L (its replies were the real ones; the replay's are 8 tokens).

**The verdict: no MISS in either pass.** Only request 1 reuses 0 rows; every other request reuses at least the preamble.

| # | ids (L) | reused (L / A1 / A2) | served by (L) | first token L / A1 / A2 |
|---|---|---|---|---|
| 1 | 11,188 | 0 / 0 / 0 | none (cold); takes snapshots at 8,896 and 11,136 | 60.7 / 60.8 / 60.9 s |
| 2 | 11,311 | 11,279 / 11,247 / 11,247 | the slot's own rows (L), the device checkpoint (A) | 0.28 / 1.01 / 0.98 s |
| 3 | 12,678 | 11,136 / 11,200 / 11,200 | the snapshot of request 1's second | 10.9 / 11.7 / 11.6 s |
| 4 | 12,141 | 8,896 / 8,960 / 8,960 | the **preamble** snapshot (a new conversation) | 24.6 / 24.7 / 24.7 s |
| 5 | 13,710 | 12,141 / 12,200 / 12,200 | the device checkpoint | 11.6 / 11.7 / 11.8 s |
| 6 | 12,049 | 8,896 / 8,960 / 8,960 | the preamble snapshot | 23.5 / 23.5 / 23.6 s |
| 7 | 12,095 | 12,032 / 12,032 / 12,032 | the snapshot request 6 took | 0.63 / 1.71 / 1.73 s |
| 8 | 13,458 | 12,032 / 12,096 / 12,096 | a snapshot a request in its conversation took | 11.3 / 11.3 / 11.3 s |
| 9 | 12,357 | 8,896 / 8,960 / 8,960 | the preamble snapshot | 25.7 / 25.7 / 25.8 s |
| 10 | 13,873 | 12,357 / 12,419 / 12,419 | the device checkpoint | 11.4 / 11.4 / 11.4 s |
| 11 | 11,923 | 8,896 / 8,960 / 8,960 | the preamble snapshot | 22.6 / 22.6 / 22.7 s |
| 12 | 13,717 | 11,923 / 11,982 / 11,982 | the device checkpoint | 13.2 / 13.3 / 13.3 s |
| 13 | 12,334 | 8,896 / 8,960 / 8,960 | the preamble snapshot | 24.7 / 24.8 / 24.8 s |

(The replay's prompts are 59 ids longer than the live ones, from its shorter recorded replies' re-rendering, so its snapshot rows sit one 64-row block higher; `requests.jsonl` has every request's counts.)

- **Rule:** no MISS; no SHORT (no request reused 1,024 or more rows less than the shared prefix `replay_cc_live.py` can count, which is a lower bound at message granularity: system blocks, tools, whole messages). The five requests that start a conversation share only the preamble with anything earlier, and reuse exactly it.
- **Where the time goes, reported not graded:** the five conversation starts (4, 6, 9, 11, 13) each re-prefill about 3.2k ids past the preamble, 22.6 to 25.8 s each of the session's 241 s of first-token time; nothing earlier shares those ids (their first messages differ). The continuations take 0.3 to 13 s.
- **Snapshots, live pass:** 9 taken (606 to 890 MiB host, 375 to 527 ms each, 4.1 s in all, 1.7% of the first-token time), 3 restored (the preamble's 5 times, request 1's second once, request 6's twice); **6 of 9 were never restored** in this session. The host budget held 6,002 of 6,424 MiB after request 13, so a session of this size on the 9B is within a few snapshots of the budget; what the store evicts first once it is full is not read here.
- **Replay determinism:** A1 and A2 reuse the same rows for every request.
- **Not read:** a longer session (the store's eviction under the budget), a session whose template rewrites the prompt's end, Claude Code against the 0.8B (it answered in text), Metal.

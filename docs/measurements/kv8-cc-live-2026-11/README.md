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

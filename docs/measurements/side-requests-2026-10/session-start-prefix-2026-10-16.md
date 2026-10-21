# What a session's first request shares with the previous session, and whether a one-slot card could keep it (2026-10-16)

Owner item (2026-10-16), measure only: how much of each session's first-request prefill (about 80 s on 7,500 tokens in opencode) is a
prefix shared with the previous session, and whether a one-slot card could keep it given the 448 rollback. **No design here.**

All of it is read from runs already on disk; nothing was loaded for it. Readers:
`TestSideRequests_evictionCounterfactual` (opencode, Chat Completions) and `TestSideRequests_sessionStartPrefix` (one session per
server, any route), both in `internal/serveapp/side_requests_replay_test.go`, skipped without `SIDE_OUT`.

## The cost

A session's first request is a cold prefill of the whole prompt, and on this path (Mellum2.1 int4mix, C′ expert streaming, one GPU
slot) it runs at about **95 ids a second**, the same rate in opencode and in Codex:

| | first prompt | time to first token | share of the session's GPU time |
|---|---|---|---|
| opencode, 6 sessions (two runs of 3 tasks) | 7,505 to 7,518 ids | 81 to 122 s (81 to 83 s with the fast alias, which shortens the title ahead of it) | not measured |
| Codex, 30 sessions (G-ids-3) | 6,102 to 6,116 ids | 63 to 65 s | 4% to 84%, median 39% (median session 162 s over 8 requests) |

## What it shares with the previous session

**Almost all of it.** The prompts of two sessions differ only in the task's own text, near the end.
- **opencode:** the first main request of a session shares **7,492 of its 7,505 to 7,507 ids (99.8%)** with the previous session's slot.
- **Codex, all 870 ordered pairs of the 30 sessions** (whichever ran first): 99.6% to 100%, median 99.8% of a first prompt.

So the prefill of a session start is, to within 13 to 25 ids, work the previous session's slot had already done.

## Could a one-slot card keep it?

**Not under the 448 rollback, in any session measured.** The windowed ring serves a follow-up that diverges up to the rollback before
the slot's newest position (448 at the planned slack, and an expert-cache load gets no grant; `cuda/kvwindow.go`). A new session
diverges from the previous one at the shared prefix `p`, so the previous session's slot can serve it only if that session **ended
within 448 positions of `p`**. Measured, how far past `p` each previous session ended:

| | sessions ended past the shared prefix by | within 448? |
|---|---|---|
| opencode, alias off, session 2 and 3 starts | 800 and 928 ids (slots of 8,292 and 8,420) | no |
| opencode, alias on, session 2 and 3 starts | 1,108 and 1,188 ids (slots of 8,600 and 8,680) | no |
| Codex, 870 ordered pairs | min **760**, median 2,833, max 9,932 ids | **0 of 870** |

Even the shortest session (Codex `search`, 2 requests, ending at 6,862 ids on a 6,102-id prompt) ended 760 past, 312 over. The
shortest opencode session (3 requests, `read`) ended 774 past its own first prompt (7,518 ids, final slot 8,292). A session that does any real work adds more than
448 tokens, so the answer does not depend on the order sessions arrive in.

**Sensitivity to the allowance** (informative: it is not this model's configuration). With a rollback of 960, what the build's grant
can reach on a card without the expert cache (`kvWindowRollbackMax`, 1,024, less one compaction step), **29 of the 870 Codex pairs**
(3.3%) and the opencode transitions at 800 and 928 would fit, the ones at 1,108 and 1,188 not.

## Caveats

- **"Final slot length"** is the last request's prompt plus its generated ids, read from serve's log; a session that compacts
  near its end can be shorter than its longest point. None of the 30 Codex sessions ends within the floor either way.
- **opencode is five transitions**, Codex 870 pairs of 30 sessions that each ran on their own server. The pairs treat any session as
  able to follow any other, which is what a long-running server sees.
- **A rollback measures the sliding layers**: the global layers hold the whole history and would serve the prefix. The window is
  what fails.
- **Not measured:** Claude Code's session starts (its first request is about 20K tokens, 10 times the opencode figure by the I3
  night's measurement); a dense model on a card that gets the grant.

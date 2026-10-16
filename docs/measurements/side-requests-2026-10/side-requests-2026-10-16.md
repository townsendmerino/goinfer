# Side requests and the one GPU KV slot (2026-10-16): measurement, and a design proposal not built

**Owner decision (2026-10-16): the scratch slot is not built.** The finding to keep: the slow main turns after side requests were
cold anyway, at session start and after a compaction, given the 448-position rollback; a kept slot would have served none of them.
What was approved instead is a product change that removes the side requests' own cost: a fast alias that runs them without
thinking (`<model>-fast`, on by default, `-fast-alias=false` turns it off; `connect` points Claude Code's small/fast model and opencode's `small_model` at it).
Its tier-1 reading is the last section. Interactive Claude Code's mid-session side requests: not run; revisit if a later run shows
side requests landing between main turns.

Owner item 3 (2026-10-16): measure by day which requests evict the main conversation's slot, what they cost the next main turn,
and whether Claude Code sends side requests that `connect` now routes to the served model; then propose a design, with the
direction "a short request does not evict a long cached prefix", and report before building. **Nothing here is built.**

## What was measured

### opencode 1.19.0 × Mellum2.1, Chat Completions (the day run of 2026-10-15, 23:05 to 23:16 PDT)

Serve `2a0d11bb`, `--ctx 16384 --moe-cache-experts`, one GPU KV slot, windowed KV (window 1024 + 512 slack), f32 KV, the C′ expert
cache. Tasks read, search, edit, one run each, all passed. 25 requests. Plan: `ids-carried-tier2/plan-cc-opencode.json`; output in
`~/goinfer-bench/ids-cc-2026-10/out` on nobara. Read with `TestSideRequests_evictionCounterfactual`
(`internal/serveapp/side_requests_replay_test.go`, `SIDE_OUT=... SIDE_CONFIG=cc-opencode-mellum`).

**The side requests** (no tools, a different system prompt, sharing at most the first 5 prompt ids with the main conversation):

| # | kind | prompt ids | generated | wall on the GPU | when |
|---|---|---|---|---|---|
| 1, 5, 9 | session-title generator | 588, 575, 577 | 1,341, 831, 2,327 | 26.9, 18.2, 41.5 s | the first request of each session, 0.6 s before the main one |
| 23 | context summarizer | 2,408 | 1,134 | 43.1 s | at a compaction, after 21 main requests |

The titles take 831 to 2,327 generated tokens because the reasoning model thinks before it names the session. The four side
requests are 129.6 s of the run's 591 s, 22%, serialized ahead of main work on a card that runs one generation at a time. That
is their own cost, separate from what they do to the slot.

**What each one did to the next main turn** (serve's own `-log-requests` fields, with the prompts re-rendered offline):

| main request | slot it found | shared with its prompt | rollback | time to first token | reused |
|---|---|---|---|---|---|
| #2 | empty (the server's first turn) | | | 106.9 s | 0 |
| #6 | #4's prompt and reply, 8,292 ids | 7,492 of 7,505 | 800 | 97.5 s | 0 |
| #10 | #8's, 8,420 ids | 7,492 of 7,507 | 928 | 121.8 s | 0 |
| #24 | #22's, 12,585 ids | 7,491 of 8,877 | 5,094 | 95.4 s | 0 |

Warm main follow-ups in the same run took 0.3 to 3.4 s to first token. So the eviction is visible and large: 95 to 122 s against
a few seconds.

**But a kept slot would not have served any of it.** A windowed-KV resident serves a follow-up that diverges up to its rollback
before the slot's end; with the C′ expert cache the build grants no extra (`kvGrantRollback` returns at once for
`cacheExperts`), so the rollback here is the planned 448 (the banner says so: "a follow-up may diverge 448 before the newest").
The shared prefix of each of the three evicted main requests lies 800, 928 and 5,094 positions before its lost slot's end. The
sliding layers no longer hold those rows whether or not a title request came between. **Prompt ids a kept slot would have served:
0 of 3.** #2 had nothing to keep. This is a property of the windowed ring plus the C′ cache on this card, and it holds only while
the rollback is 448: a dense model with the grant (up to 1,024) or a larger window would change the first two rows.

### Claude Code 2.1.296 against the same serve path, by replay (no model)

A stand-in Anthropic Messages server (`anthropic_stub.py` beside this record) records every request and answers at once. Claude
Code ran in print mode with the settings `serve connect claude-code -write` wrote (every model slot set to the served model), with
the matrix's own flags (`--dangerously-skip-permissions --max-turns 40`), in an isolated HOME. Second replay, 3 requests:

| at | request | tools | system prompt |
|---|---|---|---|
| +0.00 s | `generate_session_title`, 3,967 bytes | 0 | "You are naming a coding session so the user can pick it out of a long list of sessions..." |
| +0.08 s | first main request, 78 KB | 21 | the agent prompt |
| +0.13 s | the next main request | 21 | the agent prompt |

- **Claude Code does send a side request, and `connect` routes it to the served model.** Its first replay also logged
  `[claude-code:unrecognized_model] ... "query_source":"generate_session_title"` on stderr: the client does not recognize the model
  name for that request type but sends it regardless.
- **It evicts the same way**, through the same slot manager: a title that arrives while a main conversation holds the slot takes it.
  Here the title arrived 80 ms before the main request, so it ran first and the main request then found a slot holding a title.
  If the order were reversed, the title would take the slot from a freshly prefilled main prompt, and the next main turn
  would be cold (the I3 night's Mellum2.1 run took 464 s for Claude Code's first request). Today that
  depends on 80 ms of arrival order.
- **Not observed:** any other side request in that short task. The client's own source lists these query sources, which would
  all reach the served model: `compact`, `tool_use_summary_generation`, `prompt_suggestion`, `away_summary`, `agent_summary`,
  `extract_memories`, `side_question`, `web_fetch_apply`, `hook_prompt`, `insights`, `rename_generate_name`, `agent_namer`, and others.
  Which of them fire mid-session in interactive use is **not measured**: interactive mode needs a terminal, and was not driven.
- **Subagents** run as long conversations of their own (`CLAUDE_CODE_SUBAGENT_MODEL` is forced to the served model) and would take
  the one slot for their whole run. Not measured. A scratch slot sized for short requests would not hold one.
- **An unexplained difference:** the I3 night's archived Claude Code wire logs (`i3-night-2026-10-13/job-b`) hold 14 POSTs, all with
  21 tools and none a title request, where today's replay with the same settings and flags sends one. The archived run predates the
  all-slots `connect`. Not resolved; it does not change the design question.

## The proposal (not built)

**Direction (owner):** a short request does not evict a long cached prefix; it is served from a scratch area, and the main slot is
kept.

### What it would be

1. **A second KV slot of a smaller capacity.** Slots are uniform today (`kvSlotsFit` gives each one `need` bytes at the same
   `ctxCap`), and on this card only one fits: the C′ expert cache takes the free VRAM left at that point of the build less a flat
   384 MiB margin and any companion reserve (`capSlots`, `reservedBudget`), so nothing is left for a second slot. The scratch slot is a
   slot with its own capacity: `kvBytesForCap(capScratch)`, allocated as the others are, bound by the same `UseKVSlot`.
2. **Eligibility by footprint.** A request is eligible for the scratch slot when its prompt plus its clamped `max_tokens` fits
   `capScratch`. It cannot move mid-generation, so the clamp is the only safe test. The observed title requests ask for 4,096
   output tokens with prompts of 575 to 588, so eligibility needs about 4,700 positions; the summarizer needs 6,500.
3. **A pick that knows capacity** (`kvmgr.pickSlot`). Unchanged first: a request that continues a slot's conversation (the score
   rule) takes that slot. For any other request: among slots it fits, the smallest capacity first, empty before least-recently-used.
   A request that does not fit the scratch slot takes the main slot exactly as today. Dense models with several uniform slots keep
   today's picker: the capacity rule only applies when slots differ.

### Cost in VRAM on the 8 GB card

Mellum2.1: 7 full-attention and 21 sliding layers, `kvDim` 512. The sliding layers cost a fixed 1,536 positions however small the
scratch cap, so there is a floor of about 130 MiB at f32 and the global layers add 28 KiB a position.

| scratch capacity (positions) | 2,048 | 4,096 | 5,120 | 6,144 | 8,192 |
|---|---|---|---|---|---|
| f32 KV (what this run used) | 182 MiB | 238 MiB | **266 MiB** | 294 MiB | 350 MiB |
| f16 KV | 91 MiB | 119 MiB | 133 MiB | 147 MiB | 175 MiB |

For reference the main slot at 16,384 is 574 MiB (the banner's 0.6 GB). The after-load VRAM in this run was 7,358 MiB of 8,192;
what is left is the decode margin and the prefill passes' scratch, which a scratch slot must not eat. So it comes out of the C′
expert cache: about 97 MiB per expert slot (5.8 GB for 57), so **a 5,120-position f32 scratch costs about 2.7 expert slots, 57 → 54
or 55 of 64.** The decode cost of fewer expert slots is not measured here; it is the first tier-1 reading (below). The build would
have to price the scratch slot the way it prices `extraBytes` for a companion, so the cache is sized beside it.

### How it meets KV2 and the windowed ring

- **KV2 (`CopyKV`, the donor choice).** The scratch slot is left out of the donor set: a side request shares nothing with the main
  conversation (at most a 5-id preamble in opencode, a short shared system block in Claude Code), and a copy from a small slot
  cannot cover a long main prompt. `CopyKV` must refuse a range past the destination's capacity (a planted defect for tier 0), and
  `ReusableKVSlot` is already per slot.
- **The windowed ring.** Each slot has its own base (`kvBases[slot]`) and compaction, so a scratch slot larger than window + slack
  compacts as the main one does; the compaction staging buffer (`kvScratch`) is sized to the widest windowed layer and is shared.
  A scratch cap at or under 1,536 positions never compacts but costs the same 168 MiB, so there is no cheap small end.
- **The rollback grant.** `kvGrantRollback` computes its price over every slot's windowed layers; it already returns for an expert
  cache, so nothing changes on this card, and on a dense card the scratch slot's windowed layers join the sum.
- **Recurrent families** keep one slot (`residentSlotCount`); the scratch slot does not apply to them.
- **Concurrency.** On the C′ path one generation runs at a time, so the pick is a plain admission-time choice. MC3's batched
  decode has its own uniform slots and is untouched.

### Tiered gates, to register before any code

- **Tier 0 (seconds, by day):**
  - `pickSlot` table tests with capacities: a request that fits the scratch slot never takes the main one while the scratch slot is
    free or older; a continuation always keeps its slot; a request that does not fit takes the main slot as today; `busy` is honored.
    A planted defect (ignore capacity) turns it red.
  - On the tiny CUDA fixture: a generation in the scratch slot is bit-identical to the same generation in the main slot, across a
    ring compaction; `CopyKV` refuses a range past the destination's capacity; the allocated bytes equal `kvBytesForCap(capScratch)`
    counted as the driver rounds them (2 MiB).
  - The fit plan reports the scratch slot and the shrunk expert cache in the banner (the server's own `fit:` and `KV plan:` lines).
- **Tier 1 (a small real run, 10 minutes or less, by day):**
  - Replay this record's 25 opencode requests (`max_tokens` trimmed) with the scratch slot on and off: every main request that
    followed a side request finds its slot (counted from `-log-requests`).
  - A same-session decode A/B at 57 expert slots against 54 or 55, in one process, for the cost of the VRAM.
- **Tier 2 (night):** only if tier 1 shows a user-visible gain and a number is to be published.

### Recommendation: do not build yet

The measurement says the eviction is real but, on this card's windowed KV with the C′ cache, **not recoverable in any of the
three opencode cases that had a slot to keep**, because the rollback is 448 and the prefixes lie 800 or more back. The scratch slot
would remove a hazard (Claude Code's title-first ordering, a mid-session short request) without recovering any time in the traffic
measured. It would cost about 2.7 of 57 expert slots. What would change this:

1. **Evidence of a mid-session side request** that shares nothing with the main conversation and lands between two main turns of
   one session (interactive Claude Code's `tool_use_summary_generation`, `away_summary`, `side_question`). The cheap test is a short
   interactive session through `connect` against the pinned serve, counting tools-0 requests. Not run.
2. **A rollback larger than the distance:** a dense model with the grant (up to 1,024), or a change to the window.
3. **A client that sends a short request in the middle of a long tool loop** that reuses the main prefix less than a rollback away.

**A cheaper lever the data points at, not built either:** the four side requests were 22% of the run's GPU time, almost all of it
a reasoning model thinking its way to a session title. serve already has `--thinking` and a reasoning budget; a budget for a request
with no tools and a short prompt would cut that. It is a product decision, not a KV one.

## Open

- Interactive Claude Code's mid-session side requests: not measured.
- Subagents' eviction of the main slot: not measured; a scratch slot would not address it.
- The archived Claude Code run's missing title request: unexplained.
- The decode cost of 54 or 55 expert slots against 57: not measured.

## Tier 1: the fast alias on opencode (2026-10-16, by day)

**Amendment, the same day (owner): the alias has no output cap.** Tier 1 ran on `62ba5f07`, whose alias capped output at 1,024
tokens. The owner removed the cap before the merge ("the gain is from thinking off, and a truncated compaction summary is worse than
a slow one"): the alias now takes the request's limit or the main model's default, `-fast-max-tokens` is gone (`-fast-alias`,
default true, replaces it) and `/v1/models` no longer advertises `fast_max_tokens`. **Tier 1 was not re-run for that change**: its
three title requests generated 9, 4 and 4 tokens, so the cap never bound in any of them, and every other request in the run named
the main model. The cap change has its own tier-0 gate (`TestFastAlias_doesNotCapOutput`: a 3,000-token reply through the alias
equals the main model's on all three routes, the request's own limit is honored, and a cap planted back into each route turns it
red). The tier-1 table below is the capped build's.

The same three opencode tasks (read, search, edit, one run each) on Mellum2.1, `--ctx 16384 --moe-cache-experts`, one GPU slot, with
the alias on: serve built at `62ba5f07` (the alias PR), `connect opencode -write` run by the matrix against it, so opencode's
`small_model` named `mellum2.1-thinking-fast`. 06:56 to 07:02 PDT under the timing lock, nobara. Plan:
`ids-carried-tier2/plan-cc-opencode.json` with the serve path and name changed; output in `~/goinfer-bench/fast-alias-2026-10/out`.
Read with `fast_alias_compare.py` (beside this record).

**The "without" arm is the 2026-10-15 run above, not re-run** (an unchanged arm: same model, flags, harness pin and tasks). It is a
different session on a different serve build, three runs against three, so this is **exploratory and cross-session**: the headline
below is large against the sampling noise, and the pass/fail counts are not a comparison. A main-model request is byte-identical with
the alias on and off (tier 0), so the builds differ only on the alias.

| | alias off (2026-10-15) | alias on |
|---|---|---|
| requests | 25 | 20 |
| titles named by | `mellum2.1-thinking` | `mellum2.1-thinking-fast`, 3 of 3 |
| title: generated tokens | 1,341 / 831 / 2,327 | 9 / 4 / 4 |
| title: time on the GPU | 26.9 / 18.2 / 41.5 s (86.6 s) | 6.5 / 5.7 / 5.8 s (18.0 s) |
| summary (context summarizer) | 1 request, 1,134 generated, 43.1 s | **none: no run compacted, so unobserved** |
| each run's first main request, time to first token | 106.9 / 97.5 / 121.8 s (mean 108.7) | 82.6 / 81.2 / 81.4 s (mean 81.7) |
| passed | read, search, edit (3 of 3) | read, search; **edit failed** (2 of 3) |

- **Titles.** Their text is sensible ("Rename sum total everywhere") and carries no reasoning. What remains of a title's 6 s is the
  prefill of its 590-token prompt, which this path runs at about 100 tokens a second.
- **First main request.** It waits for the title that the client sent 0.6 s before it, so it gets back about the title's saved time:
  27 s on average, 25%. The cold prefill of the 7,500-token prompt (about 80 s) is the rest, and is untouched. After the first
  request the runs were warm as before.
- **The edit failure is not the alias's.** The run's last turn was a malformed tool call from the main model (the opening quote of
  `newString` missing, emitted as text). The alias handled only the three titles. The I3 night had this task fail 1 time in 5 on
  the same model (`i3-night-2026-10-13.md`), so one failure in three is within that. Three runs say nothing more.
- **The summary.** No run reached a compaction, so none was observed. **opencode 1.19.0's source answers it** (read from the
  embedded JavaScript of `opencode.exe`): `Provider.getSmallModel`, which reads `small_model`, has two callers, the session-title
  agent (when the `title` agent names no model of its own, and falling back to the main model if there is no small one) and a
  project-copy name generator. The compaction request takes the `compaction` agent's configured model, **else the session's main
  model**; it never calls `getSmallModel`. So the alias never handles a summarizer request, unless a user sets
  `agent.compaction.model` to it, and the 43.1 s summary in the baseline was the main model thinking.
- **Not measured:** Claude Code's small-model requests against the real model. The replay against a stub shows only that its title
  request names the alias.

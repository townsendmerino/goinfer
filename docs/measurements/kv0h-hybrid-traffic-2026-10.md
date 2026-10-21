# KV0-H — the traffic study on a hybrid model: how much of each request would a recurrent-state checkpoint or a parked thread serve (2026-10-20)

Owner's task, 2026-10-20: rerun `docs/measurements/kv0-traffic-2026-10-13.md` (KV0) on a hybrid model, to decide whether parts of KV8
(`docs/tasks/task-kv-architecture-2026-10.md`, "KV8 — Recurrent state pool and checkpoints") move ahead of KV4. KV0 §2.5 says no hybrid was
tested, so it measures nothing about recurrent-state reuse. **Measure only: no checkpoint or parking is built here, and nothing here changes
product code.** Machine: nobara-pc (RTX 2070 SUPER 8 GB, driver 595.91.07). Scripts: KV0's, in `docs/measurements/kv0-traffic-2026-10/`
(`kv0_proxy.py`, `run_sessions.py`, `kv0_run_cell.py`, `analyze.py`), plus `kv0h-2026-10/` beside this record for what is new.

## 1. Registration (written and committed before the first study session)

**What a hybrid does today** (read at main `94a6b9734`): `Model.residentSlotCount` returns 1 when `Model.hasRecurrentState` is true;
`Model.reuseLenOf` reuses only a strict extension of the whole committed sequence; snapshots and parking are refused. So on a hybrid, the
only request that reuses anything is one that is the previous request's whole sequence (prompt and reply) plus more tokens, and only if no
other thread has used the slot since.

**The proxy this replaces.** KV0's opencode data on Qwen3-4B (attention only) read: 34 of 40 same-thread follow-ups extended the whole previous
sequence; 6 stopped short of the end of the previous reply; every thread switch evicts the one slot (Claude Code switched 5 times in 9
requests). This study replaces that proxy with a real hybrid and the hybrid's own server.

**The question.** For real multi-turn harness sessions on a hybrid, per request: is it served by today's rule, or would it be served by (B) a
checkpoint saved where its thread's previous prompt ended, or (C) parking its thread's state while another thread runs, or neither; how many
prompt tokens and how many seconds of prefill is that, and where do the B divergences sit.

### 1.1 Model

A `qwen3_5` or `qwen3_5_moe` checkpoint from `~/models` on this box, never the archive (`CLAUDE.md`, "Models: stored in the archive, benchmarked
from local disk"). It must be GPU-resident and must drive tool calls. Candidates in order: **Qwen3.5-9B Q4_K_M** (`~/models/Qwen3.5-9B-Q4_K_M.gguf`
with its `.giw` sidecar), then **Qwen3.6-35B-A3B** (`--moe-cache-experts`; slower, 15.7 tok/s decode, so a fallback). Qwen3.5-0.8B is below a
tool-driving model. The choice is made by a short pilot like KV0 §1b: opencode and Codex on the `chain` task, which must reach several
requests (KV0's Qwen3-4B reached 9). The pilot's outputs are kept apart and only its request counts are quoted. **The choice is reported before
the study sessions and recorded as an amendment to this file (§1.9).** The server's default thinking mode, int4 weights, the KV the server
picks (a Gated-DeltaNet hybrid stays f32 KV: `decoder/kvresolve.go:CUDAKVF16Decline`), `-kv-sessions 1`. Context per harness is 1.4 x that
harness's first request, as KV0 chose it, read from the pilot's first requests.

### 1.2 Harnesses and tasks

opencode, Codex and Claude Code with KV0's tasks (`fix`, `edit` and `chain`, `matrix/tasks.py` and `kv0_run_cell.py`) and KV0's repetitions
(`plan-kv0-traffic.json`): opencode and Codex each `chain` x2, `fix` x2, `edit` x1 (10 sessions); Claude Code `chain` x1, `fix` x1, one
subagent session (3). The I3 sandbox and runner; the server stays up across a configuration's sessions with its one slot, so a session's first
request meets what the previous session left. **Stopping rule (KV0's):** up to 3 more `chain` sessions for a harness with fewer than 3 sessions
of 5 or more turns, a second subagent session if the first does not spawn one; then stop and report a harness that does not reach it. A session
whose server returned 503 `halted` (the swap guard) is void and re-run.

### 1.3 Per-request ground truth

For each request: (a) the **committed sequence before it** (the previous request's prompt ids plus the ids it generated), (b) the **reuse the
server actually applied**, (c) the **time to first token**. What the server itself records today (`internal/serveapp/reqlog.go`):

| | where | what |
|---|---|---|
| applied reuse | `-log-requests` `reused=` | `Generation.PrefillReused` of the request's first generation, on every route |
| the slot's record before the request | `slot=` | the length of the sequence the bound slot held |
| where the prompt left that record | `div=` | the first position the prompt left it, when it left before its end |
| time to first token | `ttft=` | from the request's start |
| the generated ids | `--log-token-ids` | the ids the generation produced, per generation, with `canon` |
| the prompt | `--log-token-ids` | its **length and SHA-256** (`idsSHA256`), not the ids |

So (b) and (c) are the server's own, and the generated ids in (a) are the server's own. **The prompt ids are not recorded, only their hash.**
The study does not add a record: it renders each recorded request body with the served model's own tokenizer and chat template (KV0's
`internal/serveapp/kv0_tokenize_test.go`, which mirrors each route's rendering) and **accepts a rendering only when its length and SHA-256 equal
the server's logged length and hash for that request**. That is a stronger check than KV0's token-count equality: an accepted request's prompt
ids are the ids the server prefilled, byte for byte. A request that fails the check is excluded from every statistic and listed. **Stop
condition:** if more than 10% of requests of a harness fail the hash check (for instance because carried ids, `ids=<used>/<carried>`, make the
server's prompt differ from a re-rendering), the study stops and I tell Francis before any record is added to the server.
The committed sequence of a thread's previous request is its accepted prompt ids plus the server's logged generated ids. The mapping of a
recorded body to a server log line is by time order and, to settle it, by the hash.

### 1.4 Categories

Threads and the rest of KV0's definitions as in KV0 §1 (a *thread* is the requests of one session whose first 200 tokens are identical; a *turn*
is a request on the session's main thread). Classify each valid request k against **its own thread's previous request p** (committed
sequence C_p = prompt(p) + generated(p)); l = the longest common prefix of prompt(k) and C_p; n = len(prompt(k)).

- **N, new conversation:** k is the first request of its thread in its session. Its shared preamble is reported separately (the longest common
  prefix with any earlier request of another session, KV0's `lcp_other`), as it relates to KV8's preamble-boundary item.
- **A, live:** l = len(C_p) and n > len(C_p) (k strictly extends the committed sequence), and **no request of another thread was served between
  p and k** (server order). Today's rule continues from the live state.
- **B, checkpoint:** len(prompt(p)) <= l < len(C_p) and n > l, and no other thread between: k extends the previous prompt but not its
  generated reply, so a checkpoint saved where the prompt ended would serve it. Recorded: the divergence (message index, role, offset in
  prompt(k): the `where` function of KV0's `analyze.py`) and the tokens re-processed under a checkpoint, n - len(prompt(p)).
- **C, parking:** another thread's request was served between p and k, and k would be A or B against its own thread (sub-labelled C-A, C-B).
  Parking serves it, plus a checkpoint if it is C-B.
- **D, cold:** l < len(prompt(p)): k diverges inside the previous prompt (compaction, edits); only deeper checkpoints would help.

Also noted per harness: whether it sends the model's reasoning back in its history, and what sits at the B divergences.

### 1.5 The do-nothing arm and the projections

Today's measured time to first token on every request is the baseline. For B and C requests the **tokens skipped** are: A-shaped C: len(C_p);
B and C-B: len(prompt(p)); in each case less the reuse today's server applied (`reused=`), floored at 0. **Projected saving (seconds)** = tokens
skipped / the cold-prefill rate measured on that model and box in this study: the median of n / ttft over the study's requests with `reused=0`
and n >= 4,096, in the bands 4,096-8,191, 8,192-16,383 and 16,384 and over (the band of the request's n; a band with fewer than 5 requests
uses the nearest). That ttft includes the first token's decode, so the rate is a slight under-statement. **Every projection is labelled as a
projection**; nothing in the saving columns is a measurement.

### 1.6 Estimate

KV0 took 30-50 minutes on a 4B model; this study runs a 9B hybrid whose cold turns re-process in full today, so I expect **2 to 3 hours** of wall
time for the 13 planned sessions (more if the stopping rule adds sessions), run **by day** (owner's instruction), detached, heartbeat every 30 s,
under the timing lock for its whole run (so it waits for any other timed run). The pilot is separate: about 30 minutes.

### 1.7 The decision rule (as proposed by the owner; unchanged)

All shares are over the **prompt tokens processed today in the study**: the sum over valid requests of (n - reused).
- **Parking is pulled ahead of KV4** if C would skip at least **25%** of those tokens, or at least **5 s median per affected turn**.
- **The end-of-prompt checkpoint is pulled ahead** under the same bars applied to B.
- If both pass, the larger saving goes first.
- If neither passes, KV8 stays where it is, and the reading goes into KV8's section.
If the data does not fit the rule, I say so; I do not re-cut it.

### 1.8 Readings to report

Per harness: counts and token shares for A, B, C, D, N; measured times to first token; projected savings; where the B divergences sit; the
rule applied as written. Exploratory timings are quoted as such. Statistics only are committed (counts, positions, categories): never ids,
bodies or text (KV0's policy; the bodies stay outside the repository).

### 1.9 Amendments

**Amendment A1 (2026-10-20, before the pilot and before any study session; owner's three additions).**
1. **The ground-truth reading is confirmed:** a rendering that matches the server's logged length and SHA-256 identifies the prompt ids exactly, so a
   record exists and nothing is added to the server.
2. **Per-request invariant.** For every request, `slot=` (the length of the sequence the slot held before it, first generation) must equal the
   **verified prompt length plus the number of generated ids** (`--log-token-ids`) of the generation the server served immediately before it
   (server order, last generation of the previous request). Every request that fails it is reported, with the difference, in the readings. A
   failure is not corrected and not dropped: the study reports whether the failures share one offset (for example one stop token not
   committed), and for each category boundary it counts the requests whose class would change at that offset, so a reader can see how much of the
   A/B line rests on an off-by-one. The classification uses the committed sequence as defined in §1.4 (prompt plus generated ids as logged).
3. **Pilot addition: a Claude Code first-request fit check** of the 9B at its KV0-sized context, about 28K (`-ctx 28672`, KV0's Claude Code
   context), on nobara. If the server does not build a resident path at that context, or the first request is declined or not served, **the
   fallback decision (Qwen3.6-35B-A3B) is taken before the study, not during it.**
4. **Swap guard: this run uses the default guard** (the server's own `VmSwap` over its pid and descendants at 512 MB over baseline, with the system
   backstop at 4,096 MB), **not KV0's `swap_guard_mb` 4,096 override.** The session driver sets no `GOINFER_SWAP_GUARD` (the plan has no
   `swap_guard_mb`). If the guard stops a session, it is reported as stopped in the readings and is not dropped; a session whose server answers 503
   `halted` is re-run as §1.2 says, and both the stop and the re-run are listed.
5. **Builds recorded here:** the serve build is main **`94a6b973`** (`~/serve-cuda-94a6b973`, built from main with B36, Option A, B37 and the
   later merges, under the timing lock), and this record's branch is `kv0h-2026-10` (registration at `129fbae0`, driver copy at `bd312720`).

**Amendment A2 (2026-10-20, after the pilot, before any study session).**
1. **Model: Qwen3.5-9B Q4_K_M** (`~/models/Qwen3.5-9B-Q4_K_M.gguf`), `qwen3_5`, the server's default thinking, int4, `-kv-sessions 1`, serve build `94a6b973`.
   The fallback (Qwen3.6-35B-A3B) is not needed. **Pilot counts** (kept apart, 2026-10-20 10:11:07 to 10:31:17 PDT; nothing else from the pilot is
   quoted): opencode `chain` at 20,480: passed, 9 turns, 8 tool calls, 10 requests; Codex `chain` at 20,480: passed, 8 tool calls, 9 requests; Claude
   Code `chain` at 28,672: 8 requests, ended on its own error (finding 2). Hash-accepted renderings: 10 of 10, 9 of 9 and 4 of 8. `slot=` invariant: 25
   requests checked, 0 failures.
2. **Finding: 1.4 x the first request is too small a window for Claude Code with this model.** At `-ctx 28,672` (KV0's Claude Code context, 1.42 x its
   first request: 20,199 tokens measured by `connect`, 17,851 as served here) the prompts ran 17.5k to 19.3k tokens and the context refilled to the limit
   within 3 turns of a compaction, three times in a row; Claude Code stopped: "Autocompact is thrashing: the context refilled to the limit within 3 turns of the
   previous compact, 3 times in a row." The session was `connect`-told a window of 28,672 tokens with an output limit of 6,425, so its usable prompt was about
   22.2k. Noted under I4 in `docs/tasks/task-integration-harness-2026-10.md`.
3. **The hash check's 4 of 8 rejections were carried ids.** Requests 2 to 5 of the Claude Code pilot rendered exactly one token shorter than the server's prompt
   (`ids=1/1 ids_fmt=1`: the server replaced the template's formatting around the reply with the reply's own ids, `internal/serveapp/carried_ids.go`), and the
   offline renderer, holding no key, could not verify the blob. **Fix, a measurement-tool change only (owner-approved):** the study server runs with
   `--ids-key-file` (a fixed 48-byte key, mode 600, outside the repository) and the offline renderer in `internal/serveapp/kv0_tokenize_test.go` takes the
   same key, the served model's name and its quant (`KV0_IDS_KEY_FILE`, `KV0_MODEL_NAME`, `KV0_QUANT`), so a carried blob verifies and the render is the
   server's. The §1.3 stop condition (more than 10% rejected) stays in force for the study. **No server restart inside a session:** the server stays up
   across a configuration's sessions as §1.2 registered; **any restart is recorded in the readings** (a restart would also drop the slot, so a restart is
   listed beside the requests it affects, and the fixed key means ids carried across one still verify, where a per-process key would not).
   A short Claude Code check (about 5 minutes, outputs kept apart) confirms before the study that the four rejected requests now match by hash.
4. **Reading added:** per harness, the number of requests whose carried ids were used **with template formatting replaced** (`ids_fmt`), **without**, carried
   but **not used** (the fallback reason is listed), and **not carried at all**, and the cross-tabulation of those four classes with N, A, B, C and D. Carried
   ids are likely what produces category A on a hybrid; this shows how much.
5. **Contexts.** opencode and Codex stay at **20,480**. **Claude Code: the rule, registered now:** *the largest context up to 49,152 that leaves at least
   512 MiB of the card free after the first Claude Code request*, found in steps of 4,096 by the VRAM the server holds after that request (`nvidia-smi`, in the
   plan's own run); **the number it lands on is recorded here before the study** and the window Claude Code is told is the served one (`connect -write` reads it
   from the server; the session record's `connect` line, "context window N tokens", is checked against the served `--ctx`, and a mismatch voids the session).
   **KV precision: the owner asked for f16 where the qwen3_5 resident path on CUDA supports it. It does not:** `decoder/kvresolve.go:CUDAKVF16Decline` keeps f32 for
   every family outside llama, mistral, qwen2, qwen3 and phi3 and for Gated-DeltaNet hybrids ("this family's attention is not covered"), and the pilot's banner reads
   `KV f32`. **Stopped and told the owner, who accepted f32 (the shipped behaviour of qwen3_5 on CUDA, so the do-nothing arm's real configuration): f16 was declined by `CUDAKVF16Decline`, f32 is used, and the registered rule chooses the context.** The Claude Code check runs at the context the rule lands on, not at 28,672, so one run checks three things: hash matches on the carried-ids requests, the fit with 512 MiB free, and whether autocompact still thrashes. **If it still thrashes at the chosen context, the result is reported before any study session, and the rule's ceiling (49,152) is not exceeded.** `connect`'s output limit is product behaviour and is not changed for the study. (A hybrid keeps K/V only for its full-attention
   layers: 64 KiB per position at f32 on the 9B, against 144 KiB at f16 on Qwen3-4B, so the f32 context the rule lands on is still large.)

**Amendment A3 (2026-10-20, after the check, before any study session).** The rule, applied at f32 KV (`docs/measurements/kv0h-2026-10/calibration.json`):
49,152, 45,056 and 40,960 do not build a resident path; 36,864 builds but leaves **389 MiB** free after the load (under the 512 MiB bar); **32,768** leaves
**647 MiB** free after the load and after the first request (the pilot's recorded 17,851-token Claude Code request, 120.4 s). **Claude Code runs at `-ctx
32768`;** opencode and Codex at 20,480. `connect` told Claude Code "context window 32768 tokens (output limit 8192)", the served window.
**The check at 32,768** (2026-10-20 11:19:55 to 11:40:34 PDT, outputs kept apart): (1) **hash matches: 10 of 10 requests accepted**, every one of them with
carried ids and template formatting replaced (`ids_fmt`), the class that failed 4 of 8 in the pilot, and 0 `slot=` invariant failures; (2) **fit: 647 MiB free
after the first request, 7,404 MiB peak in use over the session (788 MiB free)**; (3) **autocompact still fires: 3 compactions in 12 turns**, 9 of the 10
requests after the first re-prefilled in full (about 2 minutes each), and the task was not completed (`passed: false`, answer cut off at file 3 of 7, no
"thrashing" error this time). **Reported to the owner before any study session; the study waits for the owner's answer.**
**Re-estimated length** (from the pilot's and the check's measured times, not KV0's cost basis): opencode and Codex about 12 s a request in the pilot's mix
(26.8 s cold at 6-7.5k tokens, 2.2 s warm), Claude Code **about 120 s a request** (cold 17.5-19.3k tokens at 150 tok/s: 117-136 s; the check: 10 requests in
1,214 s), over KV0's per-session request counts (opencode 52, Codex 24, Claude Code about 25; 1.5 x those for the 9B). **Central 2.5 hours, range 1.5 to
4.5 hours** (high: all cold, twice the Claude Code requests, up to 3 extra `chain` sessions); Claude Code is about 60% of it. The calibration and check took
25 minutes more and are not in the figure.

**Amendment A4 (2026-10-20, before the study sessions; owner's order change).** The opencode and Codex configuration (`h-qwen35-9b-20k`, `-ctx 20480`, KV0's
tasks and repetitions: 10 sessions, `plan-study-h20.json`) runs **first**. **Claude Code is decided after the reading of the check data** (the 10 requests of the
Claude Code check at 32,768, classified as check data, not study data): if **3 or more of the 9 cold turns are B or C rather than D**, Claude Code runs as registered
after opencode and Codex; if it is **nearly all D**, it is not run on this card, the check is recorded as the Claude Code reading for nobara with the note that a Claude Code
run on a larger card is a follow-up, and that is recorded as an amendment **before the opencode and Codex readings are written**. Nothing else in the registration changes.

**Amendment A5 (2026-10-20, written while the opencode and Codex configuration runs, before any of its readings: the decision A4 asked for).** The Claude Code check at `-ctx 32768`
(**check data, not study data**; `analyze_h.py` on its 10 requests, 10 of 10 hash-accepted, 0 `slot=` invariant failures) classifies as **N 1, A 1, D 8, B 0, C 0**, and every one of
the 10 requests carried ids with template formatting replaced except the first (carried-ids classes: fmt 9, none 1; the one A is an `fmt` request). Of the **9 cold turns** (the first
request is a new conversation, N; 8 follow-ups re-prefilled in full), **B plus C is 0**. The 8 D requests diverge from their thread's previous prompt at 15,700 to 17,951 tokens into prompts
of 17,595 to 19,345: six of them 1.5k to 3.6k tokens before the end of the previous prompt (a rewritten history, user#1 and user#4), two of them 25 tokens before it (user#4 and
user#7, +3 tokens into a 1,386-token user message; a deeper, message-boundary checkpoint would serve those two, and even counting them as B it is 2, under the bar of 3). No other
thread ran in between in the check (one thread), so C cannot appear in it. **By A4's rule ("nearly all D") Claude Code is not run on this card.** The check is recorded as the Claude Code
reading for nobara: on the 8 GB card Qwen3.5-9B with Claude Code is compaction-bound at the largest context that fits (32,768, with an 8,192-token output reserve), 3 compactions in 12
turns, 8 of its 9 follow-ups cold at about 2 minutes each, and the task did not complete. **A Claude Code run on a larger card is a follow-up, and it should exercise interactive-mode background requests (the title request and the small-fast and haiku slots, routed to goinfer), not headless `claude -p` only: headless sends none, so category C cannot appear in it** (a window of about 40k or more, which this card
cannot hold at f32 for this model). The Claude Code half of the decision rule (§1.7) is therefore read from opencode and Codex only, and the readings say so.

**A5 addendum (2026-10-20, owner's question: why one thread, and did anything go to Anthropic?).**
1. **Why the check had one thread where KV0's Claude Code `chain` had two (and six requests off the first).** It is not background traffic and not `connect` switching Claude Code's
   background requests off: *neither run had any.* The wire logs of both show only `POST /v1/messages` with the full 21-tool list and the one served model (KV0: 8 requests; the check:
   10), the check's server log has 10 request lines, all `/v1/messages`, **none with `fast=1`** (the fast alias a Claude Code title or background request would name), and `claude -p`
   (headless) sends no title request. The two "threads" of KV0 were the analyzer's first-200-token rule meeting **the per-conversation attribution hash at token 21**: in KV0's chain
   request 1 and 2 shared it and requests 3 to 8 differed from request 1 at token 21 (first differences at 17,755, then 21 six times). `connect claude-code` has written
   **`CLAUDE_CODE_ATTRIBUTION_HEADER=0`** since `dceaeb4e` (2026-10-13), after KV0's sessions; the check's settings carry it, and in the check the first 200 tokens are identical in all 10
   requests (the first differences from request 1 are 17,851 twice, then 15,700 seven times). So a Claude Code conversation is now one thread by this definition even across compactions,
   which is also why the check had no category C: nothing else ran in between. (`connect` also points `ANTHROPIC_SMALL_FAST_MODEL`, the haiku, sonnet, opus and fable slots and
   `CLAUDE_CODE_SUBAGENT_MODEL` at goinfer; those slots were never used in this headless run.)
2. **Nothing went to Anthropic, and nothing could.** The sandbox is `bwrap --unshare-all`: the network is a private loopback, the only route out is the unix-socket relay to goinfer's
   proxy, and its self-test, run before the check, passed: DNS for `example.com` fails inside it ("Temporary failure in name resolution"), goinfer answers, and the self-test catches
   its planted `share-net` weakness (`sandbox-selftest.txt`). The agent's own output (stdout 89,405 bytes, stderr 84 bytes) has no mention of `anthropic.com`, a name-resolution error or
   a network error. A request toward another host would have failed inside the sandbox, not left the machine.
3. **Statement the readings will carry (registered here, before any reading):** a parking (C) result below the §1.7 bar **applies to opencode and Codex only. It is not evidence against
   parking for Claude Code on a larger card**, where the check could not measure C (one thread, no other thread in between; the window the card holds is compaction-bound).

## 2. Readings (the opencode and Codex configuration; for the owner's review, not merged)

*(Written after amendments A1 to A5, which were committed before any study session and, for A5, before any reading. Statistics only: `study-h20-stats-h.json` beside the scripts; ids, bodies and
text stay outside the repository. Pilot and check data are not in these tables.)*

### 2.1 What ran, and where it falls short of the registration

- **Study:** 2026-10-20 12:23:27 to 12:50:00 PDT (26.5 minutes against the re-estimate of 35 to 70 minutes for this configuration), serve build `94a6b973`, Qwen3.5-9B Q4_K_M, `-ctx 20480`,
  `--kv-sessions 1`, KV f32, the fixed ids key (A2), the default 512 MiB swap guard (A1). One server process for the whole configuration (the registered design; **no restart**, and **no
  swap-guard stop and no 503 `halted`**: no session was void or re-run). The timing lock was held by the study alone (the pause guard armed for 13:30 never fired).
- **Sessions:** 10, KV0's plan: opencode `chain` x2, `fix` x2, `edit` x1, Codex the same. **Requests: 84 (opencode 50, Codex 34), 84 of 84 accepted by the length and SHA-256 check, 0
  excluded; the `slot=` invariant held on all 83 checked requests (0 failures); the server's two logs agree on every request.** opencode's five sessions all passed their task. **Codex's `chain`
  r1 and r2 and `edit` r1 did not pass** (they ended after 3, 4 and 7 requests); `fix` r1 and r2 passed. The stopping rule is met for both harnesses (three sessions of five or more requests
  on two tasks: opencode `chain`, `fix`, `edit`; Codex `fix` x2 and `edit`), so no session was added.
- **Claude Code was not run** (A4, A5): the check at 32,768 is nobara's Claude Code reading, and the Claude Code half of the decision rule is read from opencode and Codex only.

### 2.2 The categories

| harness | requests | N | A | B | C | D | prompt tokens processed today | share of those tokens in N / A / B |
|---|---|---|---|---|---|---|---|---|
| opencode | 50 | 10 | 21 | 19 | 0 | 0 | 198,447 | 19.8% / 3.1% / 77.1% |
| codex | 34 | 5 | 19 | 10 | 0 | 0 | 100,366 | 29.8% / 2.8% / 67.5% |
| **both** | 84 | 15 | 40 | 29 | **0** | **0** | **298,813** | 23.2% / 3.0% / **73.9%** |

(N includes each session's 560 to 613-token title request, which opencode sends first, and the main thread's first request. "Processed today" is the sum of n - reused over the valid requests:
the tokens the server actually prefilled.) **There is no C and no D.** No session ran a second thread in between its main thread's turns, so parking has nothing to serve in these sessions; and
no turn diverged inside the previous prompt, which is what a compaction or an edit of an earlier message would do (none occurred: 0 compactions in all 10 sessions).

**A and B.** The 40 A requests were served from the live state (reused = the committed length) at a median time to first token of **2.2 s (opencode)** and **0.6 s (Codex)**, and processed 3.0% of the
tokens. The 29 B requests are the hybrid's cost today: **median time to first token 37.6 s (opencode, 19 requests) and 27.3 s (Codex, 10)**, re-prefilling the whole prompt each time (reused = 0),
and **73.9% of all the prompt tokens the server processed in the study** (220,740 of 298,813). B turns begin at the main thread's second turn (the previous reply as the template renders it in
history differs from what was generated) and recur through the sessions (for instance opencode `fix` and `edit` turns 7 to 11), interleaved with A.

### 2.3 Where the B divergences sit

All **29 of 29** sit **inside the previous reply's assistant message** (never in the user or tool messages), **15 to 123 tokens into it** (median 26), in messages of 17 to 360
tokens. A checkpoint saved where the previous prompt ended would restore the state before the reply, and the request would then process the reply (as the template renders it) and the new messages:
**71 to 416 tokens re-processed, median 201**, against the full prompt of 6,139 to 9,416 tokens today.
**History and reasoning:** opencode (`/v1/chat/completions`) sends **no reasoning back** (0 of its 40 requests with assistant history carry any), and its template re-renders the reply without its think block; Codex
(`/v1/responses`) sends reasoning items back in **24 of its 34 requests**, whose encrypted blobs carry the server's own ids (A2): those requests are the 24 with carried ids and no formatting replaced.

**Carried ids by category** (A2 item 4): opencode 50 requests with none carried; Codex 24 carried-without-formatting and 10 none. **Codex A: 19 of 19 carried ids; opencode A: 21 of 21 with none**: A does
not need carried ids. Of Codex's 10 B requests, 5 carried ids and 5 did not: carried ids do not by themselves remove B. No request used template formatting replacement (`ids_fmt`) in this configuration.

### 2.4 Time to first token, and the projection (every figure in this subsection is a projection)

Measured today: N median **18.1 s (opencode), 25.2 s (Codex)** (the 6k-7k-token first requests; the 560-613-token title requests answer in 1.7 s); A and B as above. The cold-prefill rate measured in this study
on requests with reused = 0 and n >= 4,096: **216.7 tok/s at 4,096 to 8,191 tokens (30 requests), 206.2 tok/s at 8,192 to 16,383 (9)**; none at 16,384 and over. **Projected saving, B served by an end-of-prompt
checkpoint:** tokens skipped = the previous prompt's length less the reuse today (0): **214,143 tokens, 71.7% of the tokens processed today; projected 702 s over opencode's 19 B turns (median 34.9 s a turn)
and 304 s over Codex's 10 (median 28.5 s); 1,006 s (16.8 minutes) in all, median 34.2 s per affected turn.** The projection uses the band's measured rate, which includes the first token's decode, so it
slightly under-states the saving.

### 2.5 The decision rule, applied as registered (§1.7)

| | share of prompt tokens processed today it would skip (bar 25%) | median projected saving per affected turn (bar 5 s) | affected requests | verdict |
|---|---|---|---|---|
| **B: the end-of-prompt checkpoint** | **71.7%** | **34.2 s** | 29 | **passes both bars** |
| **C: parking** | 0% | none (no C request) | 0 | does not pass: nothing to measure |

**The end-of-prompt checkpoint is pulled ahead of KV4** by the rule as written. **Parking is not, on this reading: there were no C requests.** *Registered in A5 before any reading: a parking result below the
bar applies to opencode and Codex only. It is not evidence against parking for Claude Code on a larger card, where the check could not measure C* (one thread, headless, no other thread in between). If both
had passed, the larger saving would go first; only one passed, so that clause does not apply. The data fit the rule; nothing was re-cut.

### 2.6 What this study does not show

- **Claude Code on a hybrid with interactive-mode background requests** (title, small-fast and haiku slots routed to goinfer): the check was headless and had one thread, and the card holds no window large
  enough to avoid compaction (A5). C is unmeasured there.
- **Threads that interleave**: opencode sends its title request first and never returns to it, and Codex sends one thread. A workload that alternates two live conversations (the other kind of C) is not in these sessions.
- **One model, one card, one set of tasks** (`chain`, `fix`, `edit`), the 9B hybrid at f32 KV; a deeper checkpoint (a message boundary, KV8's note) is not evaluated here beyond the Claude Code check's two near-misses.
- The cold-prefill rate comes from this study's own cold requests (4k to 16k); a 20k-token Claude Code prompt ran at 150 tok/s in the pilot and check.


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
picks (a Gated-DeltaNet hybrid stays f32 KV: `cuda/kvpolicy.go:cudaKVF16Decline`), `-kv-sessions 1`. Context per harness is 1.4 x that
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
   **KV precision: the owner asked for f16 where the qwen3_5 resident path on CUDA supports it. It does not:** `cuda/kvpolicy.go:cudaKVF16Decline` keeps f32 for
   every family outside llama, mistral, qwen2, qwen3 and phi3 and for Gated-DeltaNet hybrids ("this family's attention is not covered"), and the pilot's banner reads
   `KV f32`. **Stopped and told the owner; the application of the context rule waits on the owner's answer.** (A hybrid keeps K/V only for its full-attention
   layers: 64 KiB per position at f32 on the 9B, against 144 KiB at f16 on Qwen3-4B, so the f32 context the rule lands on is still large.)

*(The context Claude Code lands on, and the owner's answer on f32, go here as A3 before the first study session.)*

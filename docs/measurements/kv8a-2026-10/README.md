# KV8a gates — registration (2026-10-20, before any code)

The spec is `docs/tasks/task-kv-architecture-2026-10.md`, "KV8a — An end-of-prompt checkpoint on today's single slot". This turns
its five sketch gates into registered gates. CUDA only; Metal waits for CUDA to pass. Readings stay unmerged for the owner's review.
The rules below do not move after data.

## The owner's decisions (2026-10-20, before any code)

- **The save point is `c = len(prompt)`, as the spec says, with no cut.** Every cold prefill stays shaped as it is today, so cold
  output is unchanged. (A save at Keep, the prompt minus its generation tail, was considered. It would need a pass cut at Keep in
  every cold prefill, the 2–4-row tail then runs on the GEMV path, and every recurrent model's cold output would change.)
- **Gate 2's bar is bit-identity.** On CUDA a prefill split whose passes are all 16 rows or more leaves what one pass leaves, bit for
  bit, wherever the split falls (B35's CUDA split-prefill test, TestB35SplitPrefill, PASS on nobara 2026-10-20; on branch
  `b35-recurrent-checkpoint-2026-10`, PR #112, not on this branch).
  The chunker keeps a prefill's last pass at least `PrefillTailMinRows` long. So a restore at `c` followed by the suffix is expected
  to equal a cold prefill exactly. A suffix shorter than that minimum declines to cold.
- **No bar was on record to inherit.** R-01 phase 0 (commit `44a935eb`) was accepted on
  `decoder/resident_reuse_test.go:TestResidentReuseLen_recurrentExactExtensionOnly`, a unit test of the reuse-length rule; its
  registered hardware gate ("equals the cold prefill of the concatenation token for token") is recorded as unrun.
- **B35 (PR #112) is decided after KV8a passes.** KV8a reuses B35's DeltaNet copy code where it fits, and is built from this branch,
  not on B35. At `c`, KV8a leaves B35's case cold: thinking-off chat, where the next prompt diverges inside the generation tail,
  before `c`. tack runs Qwen3.5 with thinking off. The owner's earlier "KV8a replaces B35" assumed a save at Keep.

## Where the checkpoint lives on CUDA

**Device memory, charged to the load's memory plan; when the plan cannot admit it, KV8a declines and the slot behaves as today.**
No host fallback is built.

- **Why device:** the save runs on every request (gate 4), so its cost is paid on every A turn. A device-to-device copy of the
  state (`gpu.CopyDeviceBatch`) runs at device bandwidth. spec/09 measured snapshot plus restore at 623 µs for the 0.8B
  (`task-recompute-audit.md`, R-01 phase 1 correction). A host copy crosses PCIe on every save, milliseconds each way at
  tens of MiB.
- **Why no host fallback:** a second storage path doubles the restore code and its gates, for a case the 8 GB card does not hit
  at the 9B's size. A load the plan cannot fit declines, which is what the spec requires.
- **Bytes for the 9B: 50.2 MiB** in 48 device buffers, every one of its 24 DeltaNet layers' conv window and state at f32 (gate 1, read
  from the load's own line on nobara 2026-10-20 at `-ctx 20480`, which kept its 20480-position KV plan beside it). §5.7 gives 62.8 MiB
  for Qwen3.6-35B-A3B.

## Gates

| gate | rule | void when | stopping rule | cost basis |
|---|---|---|---|---|
| **1. Bit-exact save and restore** (tier 0, CUDA) | On `qwen3next-tiny` (DeltaNet) and `nemotron-tiny` (Mamba): prefill a prompt; save; run 8 decode steps, which move the live state; restore. Every recurrent layer's state and conv window, downloaded after the restore, equals the download taken right after the save, byte for byte. Plant: a restore that skips the conv window must differ. Then on the 9B (tier 1, one server start), the plan's charge line gives the checkpoint's bytes. | A fixture is missing (a skip counts as a fail); the plant does not differ. | Every layer of both fixtures compared once. | Tiny fixtures: about 1 min on nobara. The 9B bytes: one load, about 15 s (2026-10-18 probe: 12 s). |
| **2. Agreement with a cold prefill: bit-identity** | (a) Tier 0, CUDA, `qwen3next-tiny` and `nemotron-tiny`, B-shaped: turn 1 prefilled and saved at `c`; 8 decode steps move the live state and write rows past `c`; turn 2 (turn 1's prompt, a reply that differs from the generated one 15 tokens in, and a new message) restored at `c` and prefilled from there. Its last-row logits, every recurrent layer's state and conv window, and the attention K/V rows equal a cold prefill of turn 2 on a fresh resident, bit for bit. Two sizes: both turns under the 512-token fast-prefill floor (exact kernels) and both over it (fast kernels), since the rows before `c` keep turn 1's kernel class. Plants: a restore without the conv window, and a restore one position early (`c - 1`); both must differ. (b) Tier 1, the 9B through serve, 20 B-shaped requests built from the pilot's recorded bodies (never the study's), each against a cold request on a fresh server: the served tokens and the text of every logprob, digit for digit; the restore actually used (`prefill_reused_tokens` = `c`) on every warm request, and 0 on every cold one. **Any difference fails.** | (a) a fixture is missing; a plant does not differ. (b) a warm request reports no reuse (then the case did not test the restore: void that request, and void the gate below 15 tested). | (a) each fixture once. (b) the 20 requests once. | (a) seconds. (b) about 20 fresh loads (12 s each) and 40 requests: about 10 min on nobara. |
| **3. The cache-state grid** (tier 0) | `decoder/cachestate.go:cacheStateGrid` gains the save (end of prefill) and restore (at `c`) paths, with a declared cell for every kind of state: attention KV (f32 and f16, windowed), DeltaNet, Mamba, image spans. The recurrent `lcShare` cell's reason changes from "no checkpoint to restore". An undeclared cell fails the test, so the grid stays fail-closed. Plant: drop one declared cell, and the test goes red. | | Every cell checked. | Seconds. |
| **4. The cost on every turn** | (a) Tier 0: the save is ordered before the first chained decode step (`decoder/residency.go:ResidentGreedyChain`, `ResidentSampleChain`), checked through the caller with a recording fake; plant: the save moved after the first step goes red. (b) Tier 1, inside gate 5's run: each request logs the save's own device time; **p95 ≤ 12 ms** (2% of Codex's 0.6 s A median). (c) A turns' median time to first token is reported beside KV0-H's (2.2 s opencode, 0.6 s Codex) as context, not graded as a ratio, since the two runs are different sessions (`CLAUDE.md`, TE3). | The save time is not logged on every request. | (b) every request of gate 5. | Inside gate 5. |
| **5. The measured win** (tier 1, nobara, day run, detached) | KV0-H's opencode and Codex configuration exactly: `plan-study-h20.json` (10 sessions, KV0's tasks and repetitions), Qwen3.5-9B Q4_K_M, `-ctx 20480`, `--kv-sessions 1`, f32 KV, the ids key, the default swap guard, the same KV0-H driver commit (`bf023d40`, corrected below), with the serve build swapped for KV8a's. **Bar: the pooled median time to first token of B turns is at least 17.1 s (half of the projected 34.2 s) below KV0-H's pooled B median.** A turns as gate 4(c). KV0-H's readings are the do-nothing arm. | A session that KV0-H's own rules would void (the length and SHA-256 check, the swap guard, a 503 `halted`); the timing lock shared with another timed run; the 2026-10-19 heavy-load rule at a request's start; a different driver than KV0-H's 595.91.07. | KV0-H's plan once. If fewer than 10 B turns occur, add one `chain` session per harness (KV0-H's own stopping rule), once. | KV0-H took 26.5 min, with B turns cold at 27–38 s; with KV8a, about 15–20 min. Queued at 30, timeout 60. |

**Gate 2(b)'s source, the owner's answer (2026-10-20, before any run):** the pilot holds 27 recorded bodies, so 24 consecutive pairs; only
`/v1/chat/completions` returns logprobs, and only opencode's 9 pairs use it. **Every pilot pair runs on its own route:** opencode (chat
completions) compared on the served tokens and the text of every logprob; Codex (`/v1/responses`) and Claude Code (`/v1/messages`) on the
served token ids and the prompt hash from `--log-token-ids`. Any difference fails. opencode and Codex at `-ctx 20480`, Claude Code at
`-ctx 32768` (KV0-H amendment A3's rule context), each with `--kv-sessions 1`, f32 KV and the ids key.
- **Arms:** one warm server per context: for each pair, turn 1 (the earlier body) then turn 2 (the next), 24 new tokens, not streamed.
  Both turns are greedy. Turn 1 may not write the recorded reply's first token: `logit_bias` -100 on turn 2's token id at turn 1's
  length, read from KV0-H's pilot rendering (`pilot/ids.jsonl`), wherever turn 2's ids extend turn 1's. Turn 2, which carries the
  recorded reply, then diverges at turn 1's end and restores the checkpoint there. A turn 1 that writes the forbidden token voids the
  run: the ban did not reach the sampler. (This supersedes the first amendment's sampled turn 1; see the amendment notes below.)
  Turn 1 must report a cold prefill (`reused=0`), or that server is restarted and turn 1 sent again. The cold arm is one server per context
  with resident reuse off (`GOINFER_NO_RESIDENT_REUSE=1`): every request on it prefills from position 0.
- **Void per request:** a turn 2 that does not report a restore at turn 1's prompt length (a title request's pair, or a strict extension)
  did not test the restore. The gate is void below 15 tested pairs.
- **Cost:** about 40 minutes on nobara (17 pairs at about 65 s, 7 Claude Code pairs at about 170 s, four loads).

**Amended 2026-10-20, after the first run's data, on Francis's word (2026-10-20 17:22 PT: "yes to the 2(b) amendment and yes to merging #114
if the scan is clean"):** turn 1 of each warm pair is sampled (the Arms bullet above). **The first run (2026-10-20 15:23–16:28, nobara) read VOID:**
24 pairs, 5 tested, 0 of them differed, under the floor of 15. In 12 pairs (opencode #3–#8, Codex #2–#7) the server's greedy 24-token
reply to turn 1 equaled the start of the recorded reply, so turn 2 extended the whole held sequence (the request log's `reused` = turn 1 +
24, `div=-`) and continued the live state: the strict-extension path, not the restore. In 6 of the 7 Claude Code pairs turn 2 diverges
before turn 1's end (`div` 25 to 3,606 tokens before it: a rewritten history), which is correctly cold, and opencode #0 is the title
request's pair. The 5 tested pairs restored at turn 1's end (`ckpt_restore` = turn 1's length, `div` 0 to 13 tokens into the reply) and
matched cold token for token; so did all 24 pairs on every comparison. The record is `~/goinfer-bench/kv8a/gate2b-2026-10-20-1523/` on
nobara. The amendment cannot reach the 6 Claude Code pairs or the title pair, so at most 17 pairs can test the restore. **The bar, the
void rule and the floor of 15 are unchanged.**

**The second run, under the amendment (2026-10-20 17:54–18:58, nobara, record `~/goinfer-bench/kv8a/gate2b-2026-10-20-1754/`) read VOID:**
24 pairs, 5 tested (opencode #1, Codex #0–#2, Claude Code #0), 0 of them differed, under the floor of 15; all 24 matched cold on every
comparison. Turn 1 was sampled: its 24 tokens differ from the first (greedy) run's in 5 of the 17 opencode and Codex pairs and in all 7
Claude Code pairs. In the other 12 the sampled reply equals the greedy one token for token, and those 12 again extended the held sequence
(opencode #2–#8, Codex #3–#7): the model is near-certain over a reply's first 24 tokens there, so sampling at temperature 1.0 does not
move turn 2 off the live state. Claude Code #1–#6 diverge before turn 1's end as before. Stopped here, as registered; the pair set and the
floor are not widened.

**Amended again 2026-10-20, after the second run's data, on Francis's word ("1", choosing the first of four options put to him that
evening: forbid turn 1's first token):** turn 1 is greedy with the recorded reply's first token forbidden (the Arms bullet above), on a
serve build that accepts OpenAI's `logit_bias` on `/v1/chat/completions` and `/v1/responses` and the same field on `/v1/messages` (added
for this; the sampler already applied a bias). It supersedes the first amendment's sampled turn 1, which the ban makes unnecessary, and
makes turn 1 reproducible on `/v1/messages`, which takes no seed. Checked before the run on the first run's data: in the 16 opencode and
Codex pairs whose turn 2 extends turn 1, the forbidden token equals that run's greedy first token, so it is the recorded reply's. KV0-H's
rendering of the Claude Code bodies is 109 to 110 tokens shorter than the server's, so its forbidden position there is not the reply's
start; that pair (#0) already diverges at turn 1's end under greedy, and #1 to #6 get no ban. At most 17 pairs can test the restore (16
plus Claude Code #0). **The bar, the void rule and the floor of 15 are unchanged.**

**Corrected 2026-10-20, before any gate 5 data:** gate 5 named KV0-H's driver commit as `2272c3cf9`. The study ran at `bf023d40` (its
START line, 2026-10-20 12:23:27 PDT); `2272c3cf9` predates `plan-study-h20.json`, so the first attempt (2026-10-20 17:24, nobara, record
`~/goinfer-bench/kv8a/gate5-2026-10-20-1724/`) started no session, wrote no statistics and is not a reading; the runner reported rc 1, and now
voids a collection that does not complete. The registration's intent, the same driver commit as KV0-H's study, is unchanged.

**Corrected 2026-10-20, before any run:** gate 2(a) first said "another prompt takes the slot's live state" between the turns. On the one
slot KV8a serves that would overwrite the attention rows before `c` that the restore relies on, so it is not the case KV8a meets; decode
steps moving the live state are. **Scope of the bar:** bit-identity holds where every row before `c` was written by a prefill of the same
kernel class as the cold prefill's (turn 1 cold, both turns on the same side of the fast-prefill floor). Rows an earlier turn wrote in
decode are another kernel class, as they are on today's strict-extension path, and are outside this gate.

## Build estimate

S–M on CUDA, about a day of session time:
- the save after prefill and before the first decode;
- the restore, which copies the checkpoint in, truncates the attention KV to `c`, and prefills from `c`;
- `internal/kvmgr/slots.go:scoreRecurrent` returning `c` (it returns 0 today whenever the request is not a strict extension of the
  committed sequence);
- the plan charge and the decline;
- the grid cells;
- gates 1, 3 and 4(a).

Tier 0 runs on nobara (CUDA) in minutes. Gates 2(b) and 5 then take about 40 minutes of nobara time between them, each detached and
under the timing lock.

## Readings (CUDA, nobara, RTX 2070 SUPER, driver 595.91.07, 2026-10-20)

Graded against the registration above; records stay on nobara under `~/goinfer-bench/kv8a/`.

- **Gates 1 and 2(a): PASS** (night job `kv8a-cuda-gates-2`, 15:07, pin `31dd3f44`): `TestKV8a_checkpointBitExact`,
  `TestKV8a_restoreEqualsColdPrefill` and `TestKV8a_noRoomKeepsNone` pass, with the KV7 and DeltaNet snapshot tests beside them.
- **Gate 5: PASS** (`gate5-2026-10-20-1741/`, 17:41–17:52, KV0-H's driver at `bf023d40`, serve at `8efbd62f`): the B turns' median
  time to first token is **1.32 s over 28 turns, all 28 restored**, against KV0-H's 36.23 s over 29: down 34.91 s, bar 17.1 s. By
  harness: opencode 1.44 s (KV0-H 37.63 s), Codex 1.13 s (KV0-H 27.26 s). 105 of 105 requests accepted, 0 slot-invariant failures.
- **Gate 4(b): PASS** (the same run): the save's own time p95 **1.34 ms** over 105 requests (median 1.02, max 2.19), bar 12 ms.
- **Gate 4(c), context only:** A turns' median 1.42 s over 62 (opencode 2.20 s, Codex 1.20 s) against KV0-H's 1.29 s over 40
  (opencode 2.20 s, Codex 0.62 s). The Codex difference is the sessions' work, not the checkpoint: no A turn re-read its prompt in
  either run, this run's Codex A turns prefilled a median 184 new tokens against 93 (42 turns against 19, in longer conversations,
  7,618 against 6,628 tokens), and the time per new token is 6.72 ms against 6.27 ms. Two runs of different sessions; not a ratio.
- **Gate 2(b): PASS on the third run** (`gate2b-2026-10-20-2248/`, 22:48–23:53, pin `466aa167`, under the second amendment): 24
  pairs, **17 tested, 0 differ**, every pair that can test the restore. Each of the 17 restored at turn 1's end and diverged there
  (`div` = `ckpt_restore` = turn 1's length; restore 0.77–1.31 ms), and all 24 matched cold on every comparison. The run's preflight
  passed `TestLogitBias_reachesTheSamplerOnEveryRoute` at the pin, and no turn 1 wrote its forbidden token. The first two runs read
  VOID (5 tested each; above).

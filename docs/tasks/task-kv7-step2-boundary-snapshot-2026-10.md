# KV7, step 2: a snapshot at the end of the stable prompt, taken during the cold prefill — design only (2026-10-16)

Owner brief (2026-10-16), **design only, report before building:** no hoisting (it changes what the model sees). Take a snapshot at a message boundary
during the cold prefill. Pick the boundary from the rendered template: the end of the stable system and tools section, before the variable trailing system
message and the user task. Restore there, then prefill the rest. Cover the post-compaction case. Say how the windowed layers' ring state is captured at that
mid-prefill point. Estimate times from `docs/measurements/e2e-agents-2026-10-16/claude-code-diagnosis-2026-10-16.md` (PR #77). Gates in the tiered form,
including a rerun with bodies and token ids logged.

This is step two of `docs/tasks/task-kv7-prefix-snapshot-2026-10.md` (§2 names it: "the preamble boundary"). Step one is built and gated
(`docs/measurements/kv7-snapshot-2026-10/kv7-tier1-2026-10-16.md`). Nothing here is built.

## 1. What step one cannot do, in the numbers of the e2e run

Step one takes its snapshot at the end of a cold first prompt, `b = ⌊n/64⌋·64`. A windowed layer's ring then holds only rows `[b-1,471, b)`, so a later
prompt can restore a prefix `q` only if `q ≥ b - 448`: the part of the first prompt after the prefix a later session shares must be at most 448 ids.
Claude Code's is not (PR #77): the first prompt is 18,568 ids; system prompt and tools end at about **16,233**; the user message starts there; and after
the task text a **trailing system message of about 2,000 ids** (environment, agent listing, skill listing, token count, date) follows. A later session
shares about 16.3k ids and then differs (a commit hash in `gitStatus`, then the task); serve took 7 snapshots in that run and restored none; a
request after each Claude Code compaction shares 16.4k ids with the slot and finds it 6,200 positions further on than a 448 rollback serves. All of
them are cold, about 198 s each at 94 ids a second.

## 2. The boundary: where the stable part ends

**The rule: the end of the system-and-tools section, which is where the first conversation turn begins.** Serve renders a prompt as tagged segments
(`chat.Template.RenderToolsSegments`, a `Segment` is `{Text, Special}`) and tokenizes them with `Tokenizer.EncodeSegments`, which documents that every
content segment is a gap between genuine special tokens, so no token merges across a segment boundary. The boundary `p1` is the number of ids the
segments before the first turn's opening marker produce (the system text, the tools block, and any BOS). It is a by-product of the render serve already
does; no second tokenization of the prompt is needed, only of the prefix segments, whose ids are the first `p1` ids of the full prompt by that property.
For Claude Code on the e2e run `p1 ≈ 16,233` ids (rebuilt 16,391, less the 158 the rebuild differs by).

What is *after* `p1` and why it is not in the boundary: the first user message begins with `<system-reminder>` blocks that belong to the client and the
repository (`gitStatus`, `CLAUDE.md`, the attribution block). They are stable within one repository state and change when the repository does, so the
first one is worth 100 to 160 ids at best (the e2e run's divergence at 16,312, 80 ids past `p1`) and wrong whenever the status changed. The boundary
stops where nothing a client or a repository can change comes before it.

**The rule's rounding and the pass it falls in.** `q = ⌊p1 / pass⌋ · pass` where `pass` is the prefill's pass width, 512 rows, so the boundary is a
pass boundary of the prefill that is running anyway (`cudaResident.prefillChunked` cuts a prompt into passes of up to 512 rows at `startPos + 512k`;
the one exception is its last pass, which the 16-row rule may shorten). For `p1 = 16,233`, `q = 15,872`: 361 ids of the stable part are re-prefilled on
each restore, about 3.9 s. **Option for the owner:** cut one extra pass at the 64-row block below `p1` (16,192) to save about 3.9 s a restore at the
cost of a prefill whose pass sequence differs from an unmarked one, which the bit-identity gate below must then prove for that case too (the 16-row
rule and KV0's chunk-start invariance say the rows are the same, `docs/measurements/kv0-invariance-2026-10-13.md`; they were not measured with an
extra cut). Recommended: start with the pass boundary and measure.

**When to take it: instead of step one's snapshot, or beside it?** Instead, decided by the template: take the boundary snapshot when the part after
`p1` is longer than step one can serve, `n - p1 > 448`, and step one's end-of-prompt snapshot otherwise.
- opencode and Codex: `n - p1` is a few dozen ids (the task text and its markers), far under 448; step one's end snapshot restores 7,488 and 6,080 rows and leaves
  17 and 89 ids to prefill (0.2 s and 0.9 s). A boundary snapshot would restore 7,168 and 5,632 rows and leave 337 and 473 ids, 3.6 s and 5.0 s. So
  the end snapshot stays for them.
- Claude Code: `n - p1 = 2,335`, the boundary snapshot.
- One snapshot per prompt, not two: a second would cost another ~550 MiB of pinned host memory and ~400 ms for rows the first already holds.

A client whose first prompt has a long variable tail *and* a stable prefix that a template boundary does not mark (a pasted file in the first user
message) is not covered; it is counted as `unqualified` as now, and the learned alternative (the longest common prefix of two cold prompts as the
boundary, two-touch) is the fallback if that traffic shows up. Not proposed: it makes the second session cold.

## 3. How the ring state is captured at that point

The prefill of a cold prompt is a sequence of passes (`prefillCore`) over positions `[startPos + 512k, ...)`; each pass writes its rows into every
layer's K and V and, for a windowed layer, into the ring with the base compacting as the context passes the window plus slack
(`cuda/kvwindow.go`: window 1,024 + 512 slack = 1,536 positions, rollback 448). After the pass that ends at `q`, the slot holds exactly what a prompt of
`q` ids would leave: the full-attention layers' rows `[0, q)` and each windowed layer's ring with its base at the first row it keeps. **That is the
state `SnapshotKV(slot, q)` already reads** (`cuda/kvsnap.go`): for each windowed layer the rows `[max(base, q-(window-1)-rollback), q)`, through the
slot's shifted view, i.e. `[q-1,471, q)`. Nothing about the copy changes; what changes is *when*: between two passes of one prefill rather than after the
last.

Why this point is sound:
- **Ordering.** `prefillChunked` is a loop on the caller's goroutine; each pass is its own executor job that synchronises the stream. `SnapshotKV` is
  its own executor job, run between two of them, so it reads the rows after the pass that wrote them and before the pass that would move the ring.
- **No decode row can be in it** (property 3 of the design: it is taken before any decode step, as step one's is).
- **Same producer class.** The rows `[0, q)` are written by the passes the prompt's cold prefill runs, which is the class map the snapshot records
  (`kvmgr.ClassMap.ColdClasses(n)` clipped to `q`) and a requester's cold prefill would run (the restore already checks `classesEqual`).
- **The prefill that continues is unaffected**: a D2H read of rows the next pass does not write, taken while no pass runs.

## 4. The mechanism

1. **A mark on the prefill.** A new optional capability next to `Prefiller`: `PrefillMarked(ctx, embeddings, startPos, mark int, at func(pos int) error)`,
   which behaves as `PrefillLast` and, after the pass that ends exactly at `mark` (a pass boundary the backend reports back as it cuts), calls `at(pos)`
   on the caller's goroutine. A mark that is not at a pass boundary of this prefill (the start was a restore or a reuse and the passes fall elsewhere, or
   the prompt is one pass) calls nothing, and the decoder takes step one's snapshot at the end as now. A backend without it ignores the mark.
2. **The decoder.** `residentPrefillSeed` passes the mark when the request carries a boundary (below); `at` runs the existing admission
   (`residentSnapshotPrompt`'s checks: prompt of at least 2,048 ids, cold rows, not already served by a held snapshot within the rollback, budget) with
   `b = q` instead of `b = ⌊n/64⌋·64`, and `SnapshotKV(slot, q)`.
3. **Where the boundary comes from.** `KVScope` (decoder) already carries the namespace on the context; it would also carry `Boundary int`, set by the route
   from the render (`loadedModel.promptForT` returns the prefix segments' id count beside the ids). A direct library caller sets none and gets step one.
4. **Restore picks the snapshot that serves, not the longest match.** `kvSnapStore.best` returns the longest common prefix; with a boundary snapshot and
   an end snapshot of the same family held it would return the longer, unusable one and count the request `unqualified`. It must choose, among the held
   snapshots, the one whose `Serves(q)` is true for the largest `q`.
5. **Everything else is as step one:** the key (namespace, model identity, adapter, KV precision, producer class, chain of block hashes), the budget,
   LRU, the declines (recurrent, windowed f16, adapter, `-kv-share off`), the counters, the log line.

## 5. The three cases

| case | what happens |
|---|---|
| **A session start** (a new session, a slot holding another conversation or a title request) | `best` finds the boundary snapshot, `q = min(lcp, rows) = 15,872`, `Serves(q)` holds (`q` is the snapshot's end), 555 MiB copied back in about 53 ms, the remaining 2.7k ids prefilled. A title request that took the slot just before no longer costs the main request 198 s: it restores. |
| **The request after a compaction** (the history is replaced by a summary) | The prompt keeps the system, the tools and the first reminder and diverges where the summary begins (logged `div` 16,362 and 16,420). `lcp` with the snapshot is at least `q`, so it restores the same rows and prefills the summary and the reminders after it: **restore to the boundary, prefill the summary**, 2.4k and 3.0k ids. |
| **The summariser request** (`div` 17,567, the cause of which PR #77 did not establish) | Also restores to `q`; it prefills 3.8k ids rather than 19.6k. |
| **A warm follow-up** (the next turn of the same session) | Unchanged: the session's own slot serves it by continuation. The snapshot is not consulted (`q <= acq.reuse`). |

## 6. What it buys, from #77's data

Cold prefill runs at 93.7 ids a second on this path (18,568 ids in 198.2 s); a restore is about 53 ms (555 MiB at the 10.9 GB/s step one measured); the
rest is prefilled at the cold rate. The first session start on a server stays cold and takes the snapshot (+0.4 s). Estimates, **not measurements**:

| request (the e2e Claude Code run) | ids | now, to first token | with the boundary at 15,872 | at 16,192 |
|---|---|---|---|---|
| session start 2 (search) | 18,554 | 198.4 s | 28.7 s | 25.3 s |
| session start 3 (edit) | 18,557 | 197.2 s | 28.7 s | 25.3 s |
| after compaction 1 | 18,225 | 193.2 s | 25.2 s | 21.8 s |
| summariser of the second compaction | 19,625 | 208.2 s | 40.1 s | 36.7 s |
| after compaction 2 | 18,920 | 200.5 s | 32.6 s | 29.2 s |
| after compaction 2, the request that was killed | 20,320 | 215.5 s | 47.6 s | 44.1 s |

The seven cold prefills of that run total **1,411 s**; with the boundary, **401 s** (the first stays cold). The edit run alone, which timed out at 1,200 s
with 1,015 s of its time in five cold prefills, would spend 174 s there. Its remaining time is decoding (about 6.1k generated tokens at 34 a second, 180 s),
so the run is expected to finish in well under half its limit; whether it *passes* its task is the model's, not this change's, and is not claimed.

**Costs.** Host memory 555 MiB for the Claude Code snapshot (7 full-attention layers × 15,872 rows + the 21 rings' 1,471 rows, f32), against a 6.3 GiB
budget: one per client family. The take is ~390 ms once per family, before the prefill continues. Restore ~53 ms. No cost to a warm follow-up.

## 7. Failure modes and what they do

- The boundary or the mark is wrong or absent: step one's behaviour. Nothing worse than today.
- The snapshot's rows differ from the cold prefill's: caught by the gates below (bit-identity); a restore that writes nothing, one block short or one
  position off changes the reply, as the step-one fake showed.
- A tools list or system prompt that changes between sessions (a different MCP server set) changes `p1`'s ids: the chain hash no longer matches past
  the change, the restore is shorter or absent, and the request is cold; the new prompt's own cold prefill takes a fresh boundary snapshot, replacing the
  old one (the existing "longer prompt of a held prefix replaces the shorter" rule applies when it is a prefix; otherwise LRU).
- Memory: the budget and `-kv-host-bytes` apply unchanged; a boundary snapshot that does not fit is `declined(budget)`.

## 8. Gates, in the tiered form (to be registered before any code)

**Tier 0, seconds, by day, the tiny windowed CUDA fixtures** (as KV7's `TestKV7_restoreWindowed`, with `kvWindowSlack` lowered so every ring base is
past 0, and a small pass width so a short prompt has several passes):
- **M1, the mid-prefill snapshot is the computed prefix.** Prefill a prompt of `n` ids with a mark at `q` (a pass boundary); the snapshot taken at `q`
  equals, byte for byte over every K and V buffer of every layer including each ring from its base, a separate slot that cold-prefilled exactly the first
  `q` ids; restore it into a slot that held other rows; prefill the rest; the final logits and every K/V row equal the unmarked cold prefill of `n`.
- **M2, taking it does not change the prefill in progress.** The marked prefill's rows and final logits equal the unmarked one's, bit for bit.
- **M3, the boundary from the template.** For the three recorded clients' first requests (opencode, Codex, Claude Code bodies, as
  `TestSideRequests_*` render them): `p1` is the id count of the segments before the first turn; Claude Code's reads within a few ids of the rebuild's
  16,233 and `n - p1 > 448`, so the boundary snapshot is chosen; opencode's and Codex's read `n - p1 <= 448`, so the end snapshot is.
- **M4, the policy (decoder fake):** a boundary snapshot is taken in a cold prefill that passes the mark, not when a held snapshot already serves
  within the rollback; `best` returns the serving snapshot over a longer unusable one; the post-compaction shape (shares `q`, diverges after) restores `q`;
  a request that restored is not counted `unqualified`; a mark off a pass boundary takes step one's snapshot.
- **Planted defects, each must turn a gate red:** (1) the snapshot taken one pass late (the ring advanced); (2) `q` rounded up past the pass boundary;
  (3) the end-of-prompt snapshot taken for a Claude Code shaped prompt (the policy test sees `unqualified`); (4) `best` choosing the longest match; (5)
  the mark honoured on a prefill that began from a restore; (6) the ring base not set on restore (step one's plant, re-run through this path).

**Tier 1, by day, 10 minutes or less, Mellum2.1 (`--ctx 32768`, windowed, one slot):** replay the two recorded Claude Code first requests
(`side-req-2026-10/p3` and `p5`, same working directory) plus a compaction-shaped request built from the e2e transcripts (`cc_rebuild.py`'s `post1B`),
snapshots on against off, the off arm only for the second session as `run-tier1-qwen3-cc.sh` does. Cost: two cold prefills (about 205 s each), one
restored; about 8.5 minutes with two loads. **Rule (registered now):** a restored start at 0.2 of the cold one or better, replies and the first 32
greedy token ids identical on and off (hard), `VmSwap` flat (hard); between 0.2 and 0.5 parked.

**Tier 2, night, the rerun with bodies and token ids:** the e2e run's Claude Code configuration (`e2e-claude-code`, read/search/edit, one rep, serve
with step 2) with `"bodies": true` and `"token_ids": true` in the plan, so the requests are the real ones and not rebuilt. It answers two things the
tier 1 cannot: the true bodies replace the rebuild (the 158-id difference, the divergence of the summariser request, which PR #77 left open), and the
effect is read on the real traffic. Estimate **about 15 minutes** (a cold start, then restores; the off arm is the existing 27-minute run, cited and
labelled cross-session, not re-run). Why tiers 0 and 1 cannot answer it: the compaction and summariser requests only exist in a real agent session.
Pre-registered readings: every Claude Code session start after the first restores (`snap=` present, `ttft` under 0.2 of the cold one), every request after
a compaction does too, `unqualified` is 0; replies are not compared (an agent's sampling is not deterministic), the per-request bodies and ids are kept
and the restored prefix is checked offline against the ids the cold prefill wrote.

## 9. Decisions for the owner

1. **Pass boundary (15,872) or one extra cut at the 64-row block (16,192).** 3.9 s a restore against a pass sequence that differs from an unmarked prefill.
2. **End snapshot *or* boundary snapshot per prompt (proposed), or both.** Both is simpler and costs ~550 MiB and ~400 ms more for the Claude Code family.
3. **The boundary is the first turn's opening marker.** The alternative that also covers the first reminder (80 to 160 ids more, correct only while the
   repository state is unchanged) is not proposed.
4. **Scope of the route change:** `Boundary` set by the Messages, Chat Completions and Responses routes from their render. The MC3 path keeps step one's
   end snapshot (it prefills in chunks under an exclusive section; no Mellum config reaches it: the expert cache forces one slot).

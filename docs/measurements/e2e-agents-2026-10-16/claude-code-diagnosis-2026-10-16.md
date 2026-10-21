# Why Claude Code gets no snapshot restore and has cold follow-ups in the e2e run (2026-10-16)

Written for: the owner. **Diagnosis only, from logs; no GPU, no design.** The run is `results-2026-10-16.md` beside this file (serve `968c50d6`, Mellum2.1, `--ctx 32768`, windowed KV, one slot).

## What the logs hold and what they do not

The run kept no request bodies and no token ids. What it did keep: serve's request lines (`prompt_tokens`, `reused`, `slot`, `div`), the wire summaries, and the agent's own transcripts, in which every injected reminder is stored with its exact text and role (`rendered`, `renderedRole`). `cc_rebuild.py` rebuilds each request from those, onto the skeleton (system prompt and 21 tools) of a recorded Claude Code 2.1.296 first request, and `TestSideRequests_ccDiag` renders them as serve does.

**The rebuild is checked against serve's own numbers, not assumed.** Rebuilt prompts are 18,726, 18,712 and 18,715 ids against 18,568, 18,554 and 18,557 logged: 158 too long in all three, so the skeleton's system prompt is not the run's exactly (the 158 sits before every divergence below). With it removed, the divergence is 16,471 - 158 = 16,313 against 16,312 logged, and the ids after it are 2,255, 2,241, 2,244 against 2,256, 2,242, 2,245. The second post-compaction request diverges from the first at rebuilt 16,578, which is the logged 16,420 + 158 exactly.

**Correction, 2026-10-17: where the 158 came from.** `cc_rebuild.py` wrote its bodies with `json.dump`'s default escaping, so an em dash in a tool description
became the six characters `\u2014`, which serve renders verbatim as tool JSON: **109 of the 158 ids are that.** Written as Claude Code writes them (compact,
characters as they are; this reproduces a recorded request byte for byte), the rebuilt prompts are 18,617, 18,603 and 18,606 ids, **49** over the logged ones,
and the 49 still sit before every divergence. The numbers above stand with 158 read as "the escaped skeleton's offset": the same readings with the
corrected bodies are the divergence at 16,362 (less 49: 16,313 against 16,312 logged), the ids after it unchanged, and the second post-compaction request
at 16,469 (less 49: 16,420, the logged one, exactly). What the other 49 are is not established (a difference in the skeleton's system prompt that none of
the readings depends on). The script is fixed. KV7 step 2's tier 1 first ran its compaction-shaped request from an escaped body and read it cold for that
reason: `docs/measurements/kv7-step2-2026-10-17/`.

## 1. Session starts (requests 2 and 3 of the run: `div=16312`)

The rebuilt prompt, in order (ids as rebuilt, less 158 for logged): system prompt and tools to 16,391; the first user message from 16,391, whose first block is the `gitStatus` reminder; the commit hash at **16,470**; the attribution reminder from 16,531; the task text from 16,666; and then a **trailing system-role message** from about 16,690 to the end, 2,000 ids of the environment, the agent listing, the skill listing, the token count and the date, which Claude Code attaches after the user message.

- **The first differing id is the commit hash in `gitStatus`** (`Recent commits:\n` then `b0d3bec`, `1ecde16`, `9cbea77 fixture`): the harness builds a fresh fixture repository for each run, so its single commit has a new hash each time. In a real project the text is stable until the repository changes, and then it changes with it (the commit list, the status), so this is not only a harness effect.
- **It is not the only reason.** With the hash held equal, the next difference is the task text at 16,666, and the 2,000-id trailing message follows it. The part of a first prompt after the shared prefix is at least 2,050 ids either way.
- **That is the 448 limit.** A snapshot is taken at the end of the first prompt (18,560 rows for the first, as logged), and its ring holds the last 1,471 rows. A later session restoring `q` needs `q >= b - 448`: here `q = 16,256` (the hash's block) against `18,112`, 1,856 short; with the hash equal, `q = 16,448` (the task's block), 1,664 short. Serve took 7 snapshots (614 to 673 MiB, 377 to 427 ms each, one per cold prefill) and restored none; they are all unusable the same way.
- **A prefix that differs between sessions is not the cause**: 16,3xx of 18,5xx ids (88%) are common to every session. The prefix is long and stable; the snapshot just cannot reach back to it.

## 2. The cold follow-ups (requests 6, 7, 8 of the edit run)

The edit run is two compactions (Claude Code's own summaries, at +679 s and +1,159 s). Requests 4 and 5 are warm (reused 18,751 and 18,822); the three that follow a compaction are cold, 193 to 208 s each.

| request | logged `div` | what it is | where it diverges |
|---|---|---|---|
| 6 | 16,362 | the first request after compaction 1 | 16,363 rebuilt (order B): right after the `gitStatus` reminder, where the session had the attribution reminder and the new conversation has the **summary message**. The history is replaced; the prompt keeps system, tools and the first reminder |
| 8 | 16,420 | the first request after compaction 2 | 51 ids into the **second summary** (both summaries open with the same sentence) |
| 7 | 17,567 | the request that produced summary 2 | inside the trailing system message, about 360 ids after its start. **Not established**: its prompt includes the assistant turn's ids and the summariser's instruction, which the transcript cannot rebuild |

Requests 6 and 8 share 16.4k ids with the slot they find; the slots hold 22,524 and 22,627 ids, so the divergence lies about 6,200 positions behind each slot's end, against a rollback of 448. They are cold for the same reason the session starts are.

## Which fix each points to

- **KV7's preamble boundary** fits both findings. The stable region before the first user message's own content is 16.2k of 18.5k ids, and every case above shares at least 16,3xx of it with the previous request: session starts, and the request after each compaction (6 and 8 share 16,362 and 16,420; 7 shares 17,567). A snapshot taken at a block boundary at or below that shared prefix, during the first prefill, would hold the ring rows a restore needs. That is what the snapshot design names as its second step. It would turn about 16.3k of each 18.2k-id cold prefill (about 170 of the 198 s) into a restore.
- **A `connect` setting** is not established. The trailing message is built by Claude Code (the environment, the agent listing, the skill listing, the date); `connect` writes the model and window settings and has no knob for what Claude Code attaches. Even removing the skill listing (about 1,300 ids) leaves the trailing message above 448. Whether Claude Code has a setting that drops all three was not looked up.
- **Serve-side handling other than the boundary** is one candidate that changes what the model reads: rendering the trailing system-role message ahead of the first user message would put its 2,000 stable ids into the prefix, leaving only the task after it (under 448). It moves text the client placed after the user turn, so it is a behaviour question for the owner before it is a design.

## Not measured

Requests 7 and 9 beyond the above; the same on a real project (the effect of a changing `gitStatus`); the number of ids Claude Code's trailing message has on other versions. A rerun with `"bodies": true` and `"token_ids": true` in the plan would replace the rebuild with the real bodies.

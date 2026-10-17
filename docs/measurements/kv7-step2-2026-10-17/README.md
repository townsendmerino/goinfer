# KV7 step 2: the boundary snapshot, built and gated (2026-10-17)

Design: `docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md`. The owner's choices (2026-10-16): pass boundary (no extra cut); one snapshot per prompt
(the boundary one if the tail after it exceeds 448, otherwise step one's end snapshot); the boundary is the first turn's opening marker; MC3 keeps the end
snapshot. nobara-pc, RTX 2070 SUPER, driver 595.91.07, GPU runs under the timing lock. **Tier 1 reads PARKED on its registered rule, see below: not cleared.**

## What was built

- `decoder/kvsnap.go`, `model.go`, `kvscope.go`, `resident_reuse.go`: `KVScope.Boundary`; `ResidentPrefillMarked` (a resident that calls back between two passes);
  `kvBoundaryPlan` (the boundary rounded down to a pass, when the tail after it exceeds the rollback); `kvPrefillMark` (the callback that takes the snapshot);
  `residentSnapshotRows` (the take, now at any row count); `residentSnapshotPrompt` skips the end snapshot once the mark fired or a restore happened;
  restore walks `candidates` (best first by rows servable) and takes the first its window can serve; `covering` also requires that the held snapshot can serve what it shares.
- `cuda/prefill.go`: `PrefillMarkAlign`, `PrefillLastMarked` (the callback after the pass that ends exactly at the mark, on the caller's goroutine).
- `internal/serveapp/kv_boundary.go`: `prefixBoundary`, from the render alone (the first turn's content changed; the segments that render identically are the stable section; their
  ids are checked to be a prefix of the prompt's), set on the five routes' requests and passed to Generate as `KVScope.Boundary`.
- The parity manifest's `deps_hash` refreshed (64 forward goldens green): the decoder edit is non-numeric.

## Tier 0

| gate | result |
|---|---|
| **M1, M2** `TestKV7Mark_midPrefillSnapshot` (the three windowed fixtures, slack 63 so a pass is a block): the snapshot taken between two passes, restored into a slot that held other rows, equals the prefix computed cold (every K/V row, each ring from its base), and restore-and-continue reaches the cold prompt's rows and logits; the marked prefill equals the unmarked one; a mark off a pass boundary fires nothing; a ring base of 0 at the mark voids the reading | **PASS** |
| **M3** `TestPrefixBoundary_endsWhereTheFirstTurnBegins`: one boundary for two sessions of one client, ending at the first turn's opening marker, moving with the system prompt, none without a turn or a template, none for ids that are not the render's | **PASS** |
| **M4** `TestKVSnapshotBoundary_*` (hash fake): a Claude-Code-shaped session takes ONE snapshot at the boundary (896 rows of a 1,000-id section at pass width 128), the second session restores it and its reply is the computed reply; a tail inside the rollback takes the end snapshot and no mark; with an end snapshot and a boundary snapshot of one family held, the one the window serves is restored and the request is not counted unqualified; the post-compaction shape restores the boundary rows; a continuation fires no mark | **PASS** |
| the six planted defects (`run-plants.sh`): (1) taken one pass late; (2) boundary rounded up; (3) boundary ignored (end snapshot for a long tail); (4) restore considers only the longest match; (5) a second snapshot of the same prompt; (6) the ring base not set on restore | **all six RED**, sources restored. (The design's fifth plant, "the mark honoured on a prefill that began from a restore", is not reachable: the guards that prevent it also make the mark unfireable there. Replaced by (5), which is the one-snapshot-per-prompt rule.) |
| regression: `TestKV7*`, `TestKV2*`, `TestKV3*`, `TestWindowedKV*`, `TestPrefill*` on CUDA; `gate identity -backend cuda` against main; the 1,864 digests; `go test ./internal/serveapp`; vet, staticcheck, gofmt | PASS; identity **29 IDENTICAL, 0 different, 14 not run**; digests equal |

## Tier 1: Claude Code session starts on Mellum2.1 (`--ctx 32768`, windowed KV, the expert cache, one slot)

`run-tier1-cc.sh`: two recorded Claude Code first requests of one working directory (p3, p5, with their title requests to the fast alias) and a compaction-shaped request rebuilt from the e2e transcripts.

| | cold (off) | with step 2 (on) | ratio | reply and generated ids on vs off |
|---|---|---|---|---|
| session 2 (p5, 20,242 ids) | **219.5 s** | **47.5 s** (restored 15,872 rows in 47.8 ms; 4,370 ids prefilled) | **0.216** | **equal, equal** |
| compaction-shaped request (17,911 ids) | about 192 s (the first run, from an escaped body, read cold, below) | **21.8 s** (restored 15,872 rows in 48.1 ms; 2,039 ids prefilled) | **about 0.11** | not run (no off arm; 190 s) |
| session 1 (20,234 ids, takes the snapshot) | 216.9 to 217.5 s | 342.9 to 346.9 ms to take 15,872 rows, 555 MiB | | |

**Rule, registered in the design: a restored start at 0.2 of the cold one or better; replies identical; swap flat; between 0.2 and 0.5 parked.**
- **Session 2 reads 0.216: PARKED.** Replies and ids are identical and the server's swap is flat (0 kB on, 2,176 kB off), but the time is over the 0.2 line by 0.016. It is a prompt-shape effect, not a mechanism one: these recorded requests carry a 5,200-character `CLAUDE.md` reminder in the first user message, after the boundary, so 4,370 ids remain to prefill (46.5 s at 94 ids/s). The e2e run's requests, which have no `CLAUDE.md`, leave about 2,700, and the compaction-shaped one, 2,039, reads 0.11. The prediction of the design (21.7 s for the compaction-shaped request) was met to within 0.1 s. This does not change the reading above.
- **Swap.** The compaction-shaped arm's server process peaked at 89 MiB of `VmSwap` (the on and off arms of the comparison: 0 and 2,176 kB). A first attempt at the default 512 MB system-swap guard was refused mid-replay when the kernel (`vm.swappiness` 100) swapped 1.4 GB of the server's own pages during the cold prefill, before the snapshot was taken; it is kept as `out-tier1-attempt1-swapguard-tripped` and the arms were rerun at the 4,096 MB guard the e2e and I3 plans use. Disclosed deviation.
- **The compaction-shaped request's first run** (`out-tier1`, 192.4 s, not restored) used a body that Python's default JSON escaping had altered (an em dash in a tool description became `—`, which serve renders verbatim), so it shared only 2,251 ids with the recorded prompts. `cc_rebuild.py` now writes the form that reproduces a recorded request byte for byte; the corrected body is `out-tier1-cc3`. The same defect is why #77's rebuild was a constant 158 ids long (109 of them): see the correction in `docs/measurements/e2e-agents-2026-10-16/claude-code-diagnosis-2026-10-16.md`.

## Tier 2: the night rerun with bodies and token ids

`pin-kv7s2.sh`, `plan-kv7s2.json`, `run-night.sh`: the e2e run's Claude Code configuration against the pinned build, `"bodies": true` and `"token_ids": true`, then the extractor's table and
`TestSideRequests_snapshotPrefixes` (every restored prefix against the ids an earlier request wrote). Queued; read in the morning. Registered readings: every session start after the first restores
(`snap=` present, time to first token under 0.2 of the cold one), every request after a compaction does too, `unqualified` is 0.

### Result (run 2026-10-17 08:17:33 to 08:32:45 PDT, serve `e432cb40`, after tack's pilot job 1 had finished; load average 0.94 at the start)

Cells 3: read and search pass, edit fails (the model's own: it wrote `-type file`, and its last answer is a tool call as text). Session starts 203.9 s (cold), 29.5 s and 29.4 s: **0.145 and 0.144 of the cold one**,
both `snap=15872`. Eight requests restored the 15,872-row snapshot, none declined (the serve log has no decline line; the server's `unqualified` counter itself was not read, so that third reading rests on the absence
of one). Every restored prefix is the ids an earlier request wrote: `TestSideRequests_snapshotPrefixes` PASS over the twelve requests. The one cold follow-up (#3, 105.9 s, 9,906 ids) is a different agent, not a
post-compaction request: 13 tools and a 2,783-character system prompt against 21 and 6,266, sharing 20 ids with everything before it; it was the first of its kind, took its own boundary snapshot (7,680 rows), and
there was nothing to restore from. That reading of "every session start after the first" (the first of each prompt kind) is mine and is stated here.
The night run's two sections skipped (the pinned worktree had no tokenizer fixture, which is gitignored); they ran after, on the same bodies, and pass: `kv7s2-sections-rerun.md` beside `kv7s2-metrics.md`.
The report-only question: across 8 session starts, 4 pairs share the stable section and 24 do not (another working directory or tools list); of the sharing pairs the leading reminder is byte-identical in 1 of 4
(real sessions p3 and p5: 2,008 ids past the boundary, 21.4 s of prefill). Counts and positions only; the boundary is unchanged.

## Report-only question for tonight's record (owner, 2026-10-17)

Across real sessions, how often is the `CLAUDE.md` reminder after the boundary byte-identical between sessions, and what would a boundary after it save? **No change to the boundary.**
`TestSideRequests_sessionStartShare` (`SIDE_BODIES`) answers it from request bodies; `run-night.sh` runs it over tonight's session starts and the five recorded real sessions and puts the output in `kv7s2-metrics.md`.

**Read today, from the five recorded real sessions** (p1 to p5, Claude Code 2.1.296 from your real HOME, so the first user block is your `CLAUDE.md` with the memory index in it):
- The leading reminder is byte-identical in **all five** (one hash). But only **one pair, p3 and p5, shares the stable section**: the other four ran in other working directories, and Claude Code writes the working directory into the system prompt early, so those pairs agree for just **705 ids** and no boundary helps them (the same finding as the KV7 tier 1 record: "two sessions in different working directories share only 659 ids").
- For that pair the prompts agree for **17,880 ids**: **2,008 ids past the pass boundary** (15,872), which a boundary after the `CLAUDE.md` and attribution reminders would also restore: **21.4 s** of prefill at 93.7 ids/s per session start, out of the 47.5 s the boundary at the first turn leaves. What is left after it is the task text and the trailing system message (about 2,350 ids).
- **Limits, stated:** one same-directory pair is a sample of one. The five were recorded the same day through a stub; the `CLAUDE.md` block includes the auto-memory index, which changes whenever a memory is saved, so across days the reminder will differ more often than it did across these five. Tonight's sessions have no `CLAUDE.md` (the sandbox), so they cannot answer the `CLAUDE.md` part; they show the same question for `gitStatus`, the leading reminder there.

## What is and is not committed (privacy)

Request bodies and the transcripts they were rebuilt from hold a user's own prompts: for the five recorded sessions, the user's `CLAUDE.md` and the memory index in it. This repository is public, so
none of that text is here, in any commit of this branch: the bodies, `out-tier1*/` and the night run's `bodies-*` live under `~/goinfer-bench` on nobara; what is committed is counts, positions, times and
hashes, and the code that computes them. `TestSideRequests_sessionStartShare` and `TestSideRequests_snapshotPrefixes` log only those; `TestSideRequests_ccDiag`, which can decode text around a
divergence, now prints it only with `CCDIAG_TEXT=1` (output made that way is not for a commit). The check on 2026-10-17: every line of the first user block of the five sessions (50 lines of 30 or more
characters), and every 24-character window of them, searched in the branch's diffs against main, the branch tree, and every commit carrying this session's trailer: no match.

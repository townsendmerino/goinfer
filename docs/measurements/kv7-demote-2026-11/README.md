# KV7-rest, first slice on CUDA: an idle conversation's rows demote on release and promote on return (registration, 2026-11-02 20:23 PST)

**Registered before the code** (nobara-sb01; the coordinator's message of 2026-11-02 20:16, KV7-rest per the plan in
`docs/tasks/task-kv-architecture-2026-10.md` §6). Ordinary rigor: tier 0 and tiny fixtures, no timed reading.

## What it builds, and how it differs from the plan

The plan said a block host pool keyed by the prefix index's hashes. **This slice is per slot instead.** KV7-s1's prefix snapshots
already serve a preamble across conversations. The gap nothing covers is the release: when a growing conversation takes an idle one's
blocks (`kvEvictIdle`), the idle conversation's rows are gone and it re-prefills on return (R-04's resident half). So:

- **Demote.** `kvEvictIdle` copies the slot's rows to pinned host memory before releasing them, charged to `Options.KVHostBytes`
  beside the prefix snapshots. No room, or no budget: it releases as today.
- **Promote.** `ReusableKVSlot` and `ReusableKV` answer the prefix while a demoted copy holds it, so the decoder's slot choice finds the
  conversation. Once a request will reuse that slot, `PromoteKV` maps blocks and uploads the rows. A write from position 0, or a copy
  onto the slot, drops the demoted copy and returns its bytes.
- **Counted.** `usage.prefill_reuse.host_tokens` and `promoted_blocks`, which read 0 until now.
- **Limits of this slice:** not on windowed KV (opt-in; its ring rules need their own care) and not on recurrent models (one slot on
  CUDA, so nothing is released). The disk tier and the index-keyed pool remain KV7-rest's.

## Gates (red first)

1. **Bit-exact:** a conversation released and promoted continues with logits bit-identical to the same conversation never released.
2. **The saving is real:** after the release, its continuation reports reuse above zero with `promoted_blocks` equal to the blocks it
   held; without a budget (`KVHostBytes` 0) the same sequence reports reuse 0, as today.
3. **The budget holds:** a budget too small for the rows releases as today and counts the refusal; bytes return when the copy is promoted
   or dropped; a new sequence on the slot drops the copy.
4. **Untouched when off:** identity on the CUDA tiny fixtures from main.

**Plants:** a promote that uploads nothing (gate 1 red); a demote that charges nothing (gate 3 red).

**Card:** tiny fixtures and identity, about 8 minutes, within the night's 2 h.

## Readings, 2026-11-02 (`tiny.log`, `plants.log`, `identity.log`, `e2e.log`)

- **Gate 1, bit-exact: PASS.** A released and promoted slot continues with logits bit-identical to a slot holding the same rows that was never
  released. Plant: a zeroed host copy differs.
- **Gate 2, the saving is real: PASS.** Through the decoder's own generation on the tiny fixture, the returning conversation reuses 15,360
  of 15,616 tokens with 241 blocks promoted. With no host budget, the same sequence reuses 0.
- **Gate 3, the budget holds: PASS.** A budget too small releases as before and counts `demote budget`. Bytes return on promotion, and a new
  sequence on the slot drops the copy. Plant: a demote that charges nothing is red (used 0 against a 62,914,560-byte copy).
- **Gate 4, untouched when off: PASS.** `gate identity -backend cuda aef19fe5f 086ed52dd`: 29 IDENTICAL, 0 different, 14 NOT RUN.
- **A real model, reported, not a timed reading (one run each, `e2e.log`).** qwen2.5-coder-0.5b q4_k_m, `-kv-sessions 2`: conversation A
  (15,331 tokens), then B past its share (31,705), which takes A's blocks, then A returns with 40 more words.

  | `--kv-host-bytes` | A's return | reuse | reply |
  |---|---|---|---|
  | auto | 0.66 s | 15,334 from host memory, 240 blocks promoted | the same |
  | 0 (as before this change) | 2.63 s | 0, a whole re-prefill | the same |

  B's own turn took 9.62 s against 9.09 s, which includes keeping A's 180 MiB in host memory. A's promotion in turn released B, keeping its
  372 MiB. Whether that trade pays on a larger model, under R-04's rule (against the fallback a release takes today), is a registered served
  reading this slice did not run.
- **Card:** about 8 minutes.
- **Not done:** windowed KV, the disk tier, a restart, an index-keyed pool.

## The review round, 2026-11-02 21:22 PST (the coordinator's list, from Gemini's review; `review-red.log`, `review-green.log`, `e2e-review.log`, `identity-review.log`)

Each item was read red on the code as first pushed, except where noted.

| item | red | the fix |
|---|---|---|
| 1. A copy onto a demoted slot | a failed copy left: released false, copy kept false, 257 blocks held, 0 host bytes charged (was 62,914,560) | The slot stays released with its host copy until the copy succeeds; a failure unmaps what it mapped. |
| 1. "The copied rows report as own" | **did not reproduce**: `reused 6144, shared 6144`. `TestKV7_copyOntoDemoted` now pins it. | None needed. |
| 2. A demoted source | the copy released its own source; `promoted 0 blocks` when it promoted 241 | The source counts as busy for the copy. The decoder promotes it before `CopyKV`, whole, and counts it. |
| 2. A promotion is a use | the next release took the slot just promoted | `restoreKV` touches the slot's use. |
| 3. Valid rows | a 1024-row sequence over old blocks kept 15,360 rows; 241 blocks reported for 240 | A slot tracks its written end (`kvValid`). Demotion keeps that much, a returning request's promotion restores only the blocks it reuses (240), and a copy's source comes back whole, since its own conversation may return. |
| 4. The budget | 21,264 of 1,600,000 attempts charged past the budget (2,676 under `-race`); `demoted 1` after a failed copy; a failed promotion left 255 blocks mapped and lost the copy | Room and charge are one step under the lock. `Kept` counts after the copy succeeds. A failed promotion unmaps and keeps the copy. A prefix snapshot's bytes are charged before it is taken, so a demotion cannot slip in between. |
| 5. Evictable | a second release's copy refused while the first held the budget; a snapshot declined `budget` beside a copy | The oldest copy yields to a newer copy, and any copy to a prefix snapshot, before a held snapshot goes (`demote_dropped`). A request told the rows were there when they have since gone gets an error and prefills cold. |
| 5. Off by default | see below | `Options.KVDemote`. |
| 6. The PR's two claims | corrected in the PR text | Copy from a demoted source now has `TestKV7_copyFromDemoted` and `TestKV7_copyFromDemotedIsBitExact` (with a zeroed-copy plant). `host_tokens` did report restores before this slice; only `promoted_blocks` read 0 until now, and `disk_tokens` still does. |

Two tests were not read red, because they pin what already held or what a new option does: `TestKV7_copyOntoDemoted` and `TestKV7_demoteIsOffWithoutItsOption`. `TestKV7_copyFromDemotedIsBitExact` was not read red either; its plant goes red.

**Off by default, and a departure from the wording.** The decision was "`--kv-host-bytes 0` unless set". That flag is also the budget of KV7-s1's prefix snapshots, whose `auto` default was Francis's decision of 2026-10-16, and 0 would turn those off too. So demotion has its own switch, `Options.KVDemote`, which serve sets only when `--kv-host-bytes` names a budget; `auto` keeps snapshots and no released conversation's rows. Flagged to the coordinator in the status file and the PR.

**Reachability, stated plainly.** The failed copy of item 1 and the source release of item 2 need a source with an unmapped block and a full pool. The tests make that hole by hand; the decoder's own traffic was not seen to reach it. The window-compatibility decline the review names cannot reach a demoted slot, because windowed KV does not demote.

**Re-read after the round, on the merged tree (`f5ca174f5`).** Bit-exact against a slot never released, the budget, and the plants: PASS (`review-green.log`, 55 tests, the `loadflags` and `config` lines there were then fixed by regenerating the golden and adding the inventory row). Identity from `origin/main`: 29 IDENTICAL, 0 different, 14 NOT RUN (`identity-review.log`).

**A real model, reported, not a timed reading (one run each, `e2e-review.log`).** qwen2.5-coder-0.5b q4_k_m, `-kv-sessions 2`, the same three turns:

| `--kv-host-bytes` | B, past its share | A's return | reuse | reply |
|---|---|---|---|---|
| auto (default now) | 9.49 s | 2.82 s | 0 | the same |
| 1G | 9.60 s | 0.69 s | 15,334 from host memory, 240 blocks | the same |
| 0 | 9.09 s | 2.62 s | 0 | the same |

**Seen, not this slice's, not chased to the end.** Under `auto` the server took 4 snapshots and restored none of them, with nothing declined (`/admin/status`, last lines of `e2e-review.log`). Reading the code, `ClassMap.CapHit` shrinks a hit until the pass over the rest writes the cold prefill's class, and on this prompt with its 240-token tail it appears to shrink to 0, which `residentRestoreSnapshot` skips without counting. That reading is from the code; I did not print it. No line of `internal/kvmgr` changed here. It bears on what a snapshot is worth on short follow-ups.

**Card, this round:** about 15 minutes, within the night's 2 h: the red and green runs about 2, the final run 9, the `auto` repeats about 2, the push checks to follow.

## The third round, 2026-11-02 23:30 PST (`review3-red.log`, `review3-green.log`, `identity-review3.log`, `push-checks-review3.log`)

| point | reading on the code as pushed | what changed |
|---|---|---|
| 1. A copy onto a demoted slot whose own prefix passes the in-place cap "reports the copied rows as own" | **Did not reproduce.** 3,072 tokens sharing 2,048: the slot answers 2,048 and is bound, and the request reads `reused 6144, shared 6144`. `residentShare` sets `acq.shared` to the whole hit whatever `start` is, so the 4,096 in the review is not what the code computes. | `TestKV7_copyOntoDemotedWithItsOwnPrefix` pins it. No accounting changed. |
| 2. Evicting for what cannot fit | Red twice: slot 3's copy dropped for a reservation that could not fit beside the taker's; the store dropped its snapshot and then refused (`0 snapshots held, 40 bytes used`). | Both check the fit first, counting what cannot be dropped. The fourth round closes what this left open. |
| 3. A partial promotion answering for rows it does not hold | Red: `slot 1 answers 15360 for 15360 rows, holding 1024`. | `kvReleasedLost` also answers lost when no block is mapped at the end of the prefix (full group; windowed KV keeps its own answer). This path runs for every slot, so identity was re-run. |
| 4. Promoting a source | Half red: room for a copy could come from the bound slot's own copy (`reserved true; the bound slot's copy kept: false`). **The other half did not reproduce:** a promoted source past its share, idle again, is released with a new copy when a slot within its share needs blocks, not stopped (`TestKV7_promotedSourceGoesBackToHostMemory`, green on both sides). | The bound slot's copy is never the one dropped. |
| 5. Clearing before mapping, and the docs | Red through both writers: a failed write from position 0 left `released false, 288 blocks held, copy kept false`. The golden's named-budget case and two comments were stale. | `kvEnsureSlot` and `UploadKV` clear the released state after mapping, and a failure unmaps. A golden case reads `KVDemote` true under a named budget. `KVPromotedBlocks` and `promoted_blocks` say what they count. |

The test for point 5 first asked for less than a reclaim could free, so it did not fail at all (recorded in `review3-red.log`); it now asks for the whole table.

## The fourth round, 2026-11-03 01:15 PST (`review4-red.log`, `review4-green.log`, `identity-review4.log`, `push-checks-review4.log`)

| point | reading on the code as pushed | what changed |
|---|---|---|
| 1. A promoted source reclaimed under a full pool | Red: `the source: stopped true, 255 blocks held of 383`. It needs a block the copy must allocate, which the test makes with a hole: the review's other route, a hit that does not end a block, is refused by `CopyKV`. My third-round test that said this did not reproduce never called `CopyKV`. | **A copy from a demoted slot no longer promotes it.** `CopyDemotedKV` writes the rows the copy needs from the host copy straight into `dst`, so the source maps nothing, cannot be reclaimed, and keeps its copy for its own return. A live source is out of a reclaim's reach for the copy (`kvCopySrc`). |
| 2. Answers for rows past what was written | Red, live and demoted: `a live slot answers 15360 for 15360 rows, having written 1000`; `a demoted slot answers 1024 for 1024 rows, having written 1000 in that block`. | `kvReleasedLost` is now `prefix > kvValid`, and `kvValid` holds through a release, so a whole-block copy answers for the rows written. This runs for every slot: a plain continuation reuses the same rows before and after on eight families (`TestKV7_validRowsKeepOrdinaryReuse`, 3,076 of 3,588), and identity was re-run. |
| 3. Holding a stale copy during a write from zero | By reading: the decoder forgets the ids before a write from position 0, so the copy kept on a failed write (the third round's fix) could never be used again. | A write from zero drops the copy first, so its bytes are there for releases the write causes, and clears the released mark after the mapping. A failed write leaves the slot released, unmapped, with no copy. A copy onto a released slot still holds its copy until success, because there the request can still promote it on a decline. |
| 4. Evicting beside bytes in flight | Red: `slot 1's copy was dropped for a reservation that could not fit beside the bytes in flight`. | `KVHostBudget.Fits` counts everything that cannot be dropped, in-flight reservations included. A snapshot's reservation now goes through the same path (`ReserveKVHost`), which closes the case the third round left open. |
| 5. `UploadKV`'s flush error; `st.used` read without the lock | By reading. | The flush error unmaps as the map error does. The log line reads the used bytes through `stats()`. The review's point that layer 0 clears the mark before other layers map stands for hybrid windowed models, but those do not demote: left as it was before this slice. |
| 6. Demotion under `KVShareOff` | Red: `slot 1 answers 15360, copy true`. | Off under `KVShareOff`, as the snapshots are. |

## The fifth round, 2026-11-03 03:16 PST (`review5-red.log`, `review5-green.log`, `identity-review5.log`, `push-checks-review5.log`)

| point | reading on the code as pushed | what changed |
|---|---|---|
| 1. A failed copy onto a demoted slot goes cold | Red, through the decoder's own acquire: `promoted 0 times`, `declined map[error:1 ...]`. The path was the snapshot restore after the copy: it failed on the same pool, and its error path zeroes reuse and forgets the slot. | A released slot keeps nothing of a failed copy or restore, so every such failure is a `*KVCopyDecline`, and the request falls back on the slot's own copy: `promoted 1 times, 32 blocks`. |
| 1b. "Normal traffic" | **Not as stated.** Under full pool a copy can fail only past `dst`'s share, since within it a reclaim serves it. The prompt is then past the share too, so its prefill stops as at the context cap whatever the copy did. The fallback saves work, not the request. | The test reads the acquire for that reason, not the generation. |
| 2. `CopyDemotedKV` ignores `start`; a failed restore on a live slot | Red: `96 blocks written from the copy; want ... 64`. | `start` is passed through: a live `dst` keeps what it holds, and only the rest is written and counted. A live slot's failed restore lowers `kvValid` to the rows it still holds. Blocks left mapped on a live slot stay its own, to be written over. |
| 3. `UploadKV` and `kvValid` | By reading: a later layer's failure left `kvValid` at the first layer's end. | Lowered to `base` on any layer's failure. Not tested: only a hybrid windowed model can reach it. A batched step's `kvEnsureSlot` is left: `kvValid` is a ceiling there, and the decoder's record, which a failed step does not extend, is what a request is scored against. |
| 4. A promotion holds its own bytes while it evicts | True by design, not changed: the copy has to outlive the upload, or a failure loses it. Two conversations taking turns need room for both copies, not the larger. It goes in what the served reading must look at. | None. |
| 5. The continuation test | It could skip and asserted little. | It fails on a load or a decline, and asserts the whole continuation, 3,076 rows. |


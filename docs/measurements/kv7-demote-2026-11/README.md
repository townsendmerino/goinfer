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
| 4. The budget | 21,264 reservations of 1,600,000 charged past the budget (2,676 under `-race`); `demoted 1` after a failed copy; a failed promotion left 255 blocks mapped and lost the copy | Room and charge are one step under the lock. `Kept` counts after the copy succeeds. A failed promotion unmaps and keeps the copy. A prefix snapshot's bytes are charged before it is taken, so a demotion cannot slip in between. |
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

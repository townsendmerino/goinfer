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

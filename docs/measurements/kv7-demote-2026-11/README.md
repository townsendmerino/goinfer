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

# KV7, first slice: a prefix snapshot in host memory — design only (2026-10-16)

Owner brief (2026-10-16), **design only, report before building:** at the end of a session's first prefill, copy the slot's KV for
the prefix later sessions share (the full-attention rows and the sliding-window ring state at that point) to host RAM, keyed like KV2
(namespace, token-id hash). A new request whose prompt starts with that prefix restores it into its slot by device copy and prefills
only the rest. Nothing here is built. The data behind it is `docs/measurements/side-requests-2026-10/session-start-prefix-2026-10-16.md`
(PR #52): a session's first request is 63 to 122 s of cold prefill, 99.8% of it shared with the previous session, and a one-slot
card keeps none of it (the previous session ended 760 to 9,932 positions past the shared prefix, against a 448 rollback).

## Owner decisions (2026-10-16): approved to build

- **Budget:** 10% of physical RAM, capped at 8 GiB, by default; **pinned only when a snapshot is taken**, so nothing is held while
  idle. `-kv-host-bytes` overrides it (`auto`, `0` off, or a byte count); the budget and use are logged at startup and in `/admin/status`.
- **Admission: first touch.** **The preamble marker is not waited for:** first prompts that do not qualify (the part after the shared
  prefix past the rollback) are counted (`unqualified`), and tier 1 reports that count.
- **Gates as designed:** tier 0 bit-identity with all five planted defects; tier 1 the opencode and Codex session-start replay with
  replies identical on and off, and one Claude Code session start on Qwen3-4B if it fits by day.
- **Built on `kv7-snapshot-2026-10`; PR merged by hand on green CI after tier 1.**

## 1. Does it work on today's slot storage, without KV4?

**Yes, as `CopyKV` does, with a host endpoint instead of a second slot.** Everything it needs exists:
- `CopyKV` (`cuda/kvcopy.go`) already copies rows `[start, end)` of every layer in whole 64-row blocks, and for a windowed layer
  copies `[max(src base, end-(window-1)-rollback), end)` and sets the destination's ring base (`kvBases`) to the first row.
  A snapshot is that same set of rows with the destination replaced by a pinned host buffer (`gpu.NewHostBuffer`, already used for the
  logits readback); a restore is the same copy the other way, into the bound slot, with the same base rule.
- It runs as one executor job that synchronises the stream, so it is ordered after the prefill that wrote the rows and before the
  decode that would overwrite them.
- **It works on an expert-cache (C′) load, where KV2 does not:** `TestKV2_inertOnExpertCaching` shows why, there is one slot and so
  no source. A host snapshot needs no second slot. This is the case that matters on the 8 GB card.

**Limits that carry over from `CopyKV`:** a recurrent family declines; **a windowed layer at f16 declines** (the shifted view does not
address it), so Mellum2.1 (windowed, f32) and Qwen3-4B (not windowed, f16 by default) both work and a windowed f16 family would not;
the geometry (`ctxCap`, window, slack, precision) is recorded with the snapshot and a reload drops every snapshot.

**What KV4 would change, and why not wait for it.** On paged storage a snapshot is a list of 64-row blocks per layer group with
refcounts, the ring state needs no special case (window blocks are blocks), and a host tier is KV7's own. This slice stores its
bytes **block by block, per layer, in the chain-hash order KV2's index uses**, so KV4/KV7 adopt the format and drop the slot
copy; nothing built here is thrown away. This is KV7's pinned host tier cut to one prefix, over slot storage: no disk, no
promotion policy beyond "restore on hit".

## 2. What is stored, and when

**Taken at the end of a cold first prefill, before the first decode step**, from the bound slot: for the prompt's `n` ids, the
full blocks `b = ⌊n/64⌋·64`: full-attention rows `[0, b)` of every non-windowed layer, and for each windowed layer the ring rows
`[max(base, b-1,471), b)` (the window less one plus the 448 rollback). It never takes decode rows, because the decode has not run, so
§5.6 property 3 (continuation) does not arise.

**The restriction of a first slice, and what lifts it.** A later session restores `q`, the block-aligned common prefix, with `q ≤ b`.
The sliding layers can serve `q` only from ring rows `[q-1,023, q)`, which the snapshot holds only if `q ≥ b - 448`. So **the part of
the first prompt after the shared prefix must be at most 448 ids**, plus the block rounding. That holds in the data: opencode's task
text is 13 ids past the shared 7,492, Codex's is 33 past 6,072. A first message that is long and differs (a pasted file) does not
qualify: the snapshot would be unusable, so it is not taken, and counted. The fix is to snapshot at the **preamble boundary** during
the prefill (serve already knows where the system message and tools end, §5.7 for KV8), at a chunk boundary below it. It is the second
step, not the first.

**Admission.** First touch, for a prompt of at least 2,048 ids, taken only when no stored snapshot already matches the new prompt within
448 of `b` (then the request restores that one instead, and the snapshot stays). The alternative, two-touch (record the chain hashes
on the first prefill and snapshot on the second that matches), costs the second session a cold prefill and is what a cache with
unrepeated traffic would choose. First touch fits repeated sessions; `snap_taken`, `snap_restored` and `snap_evicted_unused` counters
tell which one the traffic is, and the choice is one line.

## 3. Key and lookup

The key is KV2's: `(namespace, model identity, adapter, KV precision, producer class)` as the root, then the chain of block hashes
over the token ids. A new prompt walks its own chain against each stored snapshot's chain and takes the longest block-aligned match
`q`. **Namespace is in the root** (§5.15): a restore is a cache hit a client can time, so snapshots never cross namespaces, and
`-kv-share off` turns this off with KV2's copies. Equal text with different ids does not match (KV2's accepted rule).

**Caps, the same as KV2's:** `q ≤ ⌊(n-16)/64⌋·64` (at least 16 rows left for the fast-pass minimum), `n ≥ 512` so the requester's cold
class is the snapshot's (`fastPrefillFloor`), and `q` beats the bound slot's own reuse. Where a device donor (KV2) holds as much,
that is taken first: a device copy is faster than the PCIe one.

## 4. Size, restore time and how many to keep

Mellum2.1: 7 full-attention and 21 sliding layers, `kvDim` 512, f32. Qwen3-4B: 36 layers, `kvDim` 1,024, no window. Rows are the
shared prefix rounded down to a block. Copy time at the **10.90 GB/s measured on this board for pinned host-to-device**
(`docs/QUEUE.md`, "the weight copies run at line rate"); 4.7 GB/s if the host buffer were pageable (`r14-drafter-argmax-2026-09-22.md`),
which is why the buffers are pinned.

| session start | model | rows | snapshot | restore, pinned | restore, pageable |
|---|---|---|---|---|---|
| opencode (7,505 ids) | Mellum2.1 f32 | 7,488 | **325 MiB** (205 full + 121 ring) | 31 ms | 73 ms |
| Codex (6,105) | Mellum2.1 f32 | 6,016 | **285 MiB** | 27 ms | 64 ms |
| Claude Code (20,199) | Mellum2.1 f32 | ~19,968 | ~667 MiB | 64 ms | 149 ms |
| opencode | Qwen3-4B f16 / f32 | 7,104 | 999 MiB / 1,998 MiB | 96 / 192 ms | 223 / 446 ms |
| Codex | Qwen3-4B f16 / f32 | 5,824 | 819 MiB / 1,638 MiB | 79 / 158 ms | 183 / 365 ms |
| Claude Code | Qwen3-4B f16 / f32 | 20,160 | 2,835 MiB / 5,670 MiB | 273 / 545 ms | 632 / 1,265 ms |

The Qwen3-4B first-request sizes are `connect`'s measured figures (5,850, 7,165, 20,199). The snapshot's D2H at the end of the first
prefill costs the same milliseconds, once, on the executor thread, before the first token.

**How many.** Host RAM is the budget, and pinned memory cannot page, so it comes from the never-swap live ceiling (§5.8). A flag,
`-kv-host-bytes`, default 2 GiB, holds **about 6 Mellum2.1 snapshots, or 2 Qwen3-4B f16 ones at an opencode or Codex size, and
declines a Claude Code one on Qwen3-4B**. One snapshot per preamble family is what pays: the three clients are three families.
**Eviction:** least recently restored first; a new snapshot that matches an existing family within 448 replaces it only if it is
longer by a block or more, and a snapshot never restored is the first to go. Snapshots are dropped on model unload, adapter change or
any geometry change, and are never written to disk (RAM only: the disk tier is KV7 proper).

## 5. Why cold and warm stay the same

The standard is §5.6 property 2 (sharing invariance), with property 3 not in play. Three things make a restored prefix equal the one
a cold prefill would compute:
1. **Same ids, same rows.** Row `i` depends only on tokens `0..i`, and the key is the chain of the ids themselves: equal chain, equal
   inputs. The bytes are copied, not recomputed.
2. **Same producer class.** The snapshot's rows came from a prefill of `n₁ ≥ 512`, which runs the fast kernels for every row;
   the requester's own cold prefill of `n₂ ≥ 512` would too, and the restore point is a block boundary with at least 16 rows left,
   the chunk-start invariance KV0 measured on CUDA (`docs/measurements/kv0-invariance-2026-10-13.md`).
3. **Prefill-class rows only,** taken before any decode step, so no decode-class row can be shared.

The gate is KV2's G3: **the same reply whether the prefix was restored or computed, in every schedule.**

## 6. What it should buy (estimates, not measurements)

From PR #52's data: the first request runs at **95 ids/s**; restore is 27 to 31 ms on Mellum2.1; the remainder prefills at the same rate.

| session start (Mellum2.1, one slot) | now | with the snapshot | saved |
|---|---|---|---|
| opencode, alias on | 81.2 to 82.6 s to first token (6 s of it the title ahead) | about 6.7 s: title, 31 ms restore, 17 ids (0.2 s) | **about 76 s, 92%** |
| opencode, alias off | 97.5 to 121.8 s | about 19 to 41 s, the title's own time | 78 to 80 s |
| Codex | 63 to 65 s | about 1.2 s: 27 ms restore, 89 ids (0.9 s) | **about 63 s, 98%** |
| Codex, whole session | median 162 s of GPU time | about 100 s | about 38% |
| Claude Code | about 213 s at the same rate (not measured; the I3 night took 464 s) | about 2 s, if 99% is shared | 3.5 minutes or more |

Two limits on that. **The first session on a server is cold:** it creates the snapshot. In G-ids-3 each run had its own server, so
this counts a long-running one, where 29 of 30 session starts qualify. And the snapshot must survive: one per family in 2 GiB.

## 7. Tiered gates (to be registered before any code)

- **Tier 0, seconds, by day, tiny CUDA fixture** (the windowed-KV tests lower `kvWindowSlack` to force a compaction; the KV2 tiny
  tests are the model):
  - **The restore is bit-identical to computing it:** cold prefill of a prompt in one slot, snapshot, forget the slot, restore, then
    compare every layer's K and V rows, the full-attention rows `[0, q)` and each ring's `[q-1,023, q)`, with a cold prefill of the
    same prompt, then prefill the rest and compare logits and a greedy reply.
  - **Planted defects, each red:**
    - **a stale ring:** a snapshot whose ring rows are read after the slot advanced one compaction (the slot's base moved);
    - the destination's ring base not set on restore;
    - a snapshot taken after a decode step;
    - a key without the namespace (a cross-namespace hit);
    - a restore past `b - 448`.
  - **Counters:** the pinned bytes allocated equal the snapshot's bytes and stay under `-kv-host-bytes`; LRU order; the dedup rule;
    declines named (recurrent, windowed f16, `n < 512`, fewer than 16 rows left, geometry changed).
  - **Serve, scripted:** a second session start on a one-slot server reports its reused rows from the snapshot, and its reply equals the cold one.
- **Tier 1, by day, 10 minutes or less, the real model:** replay the first requests of the recorded opencode and Codex sessions
  (bodies exist for both) against Mellum2.1, snapshots on and off, interleaving the two families. Off: four cold starts, about 290 s.
  On: two cold starts, two restored, about 150 s. With two loads, about 9 minutes. Reported per start: time to first token, rows
  restored, the first 32 greedy ids (equal on and off, hard), pinned bytes, and per-process `VmHWM` and `VmSwap` (flat, hard).
  Registered rule: a restored start at 0.2 of the cold one or better, replies identical, swap flat; anything between that and 0.5 is parked.
- **Tier 2:** only for a published number, and not needed for the decision.

## 8. Risks and open decisions

- **The 448 restriction** (§2) excludes sessions that open with a long, different first message. Counted, not hidden; the preamble
  boundary lifts it.
- **Pinned memory against never-swap.** Every snapshot is pinned RAM the box cannot page. The default of 2 GiB is a proposal; it needs the owner's number.
- **Churn.** First touch snapshots every novel long prompt; the counters judge it, and the floor (2,048 ids) and LRU bound it.
- **Latency on the first request:** 30 to 270 ms added once per snapshot taken, against 60 to 200 s saved per restore.
- **Privacy.** A snapshot is a user's prompt in host RAM, as the slot is in VRAM; it is namespaced, dropped on unload, never written to disk.
- **Decisions for the owner:** the budget; first touch against two-touch; whether the first slice waits for the preamble marker.

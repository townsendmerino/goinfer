# The coalescing fix I proposed would fix nothing — and L2 already does most of what the GQA kernel promises

Measured 2026-09-13, nobara-pc. ncu on `attn_batched` at M=1, split-KV OFF, depth 8000, grids
asserted. Prompted by `splitkv-f-depth-invariance-2026-09-13.md`, which found D7's measured DRAM
throughput running 4.6 pp above the roof model's prediction and blamed sector waste.

## 1. The sector-waste hypothesis is REFUTED

| | bytes/sector | sectors/request |
|---|--:|--:|
| D7 Qwen2.5-7B | **66.33 %** | 8.16 |
| mistral-7b | **66.33 %** | 8.15 |

Identical to two decimal places. **A number that is the same on both models cannot explain a
difference between them.** The waste explanation in the depth-invariance record is withdrawn.

It also confirms A1's fix is intact: 66.33% against the **21.96%** `task-prefill-attention.md:68`
measured. That figure was the basis for proposing a coalescing kernel change here — and it came
from the kernel at **M=2048 prefill shape, pre-A1**. `ollama-chase.md:284-296` already says so
plainly: *"A1 killed the coalescing bound (L1TEX 71→38%); the new bound is OCCUPANCY STARVATION."*
The proposed kernel change was aimed at a bound removed a campaign ago.

## 2. What the gap actually is: D7 moves 28.5% more DRAM traffic than its ideal

| model | ideal KV bytes | measured DRAM read | ratio |
|---|--:|--:|--:|
| D7 Qwen2.5-7B | 32.77 MB | 42.09 MB | **1.285** |
| mistral-7b | 65.54 MB | 65.64 MB | **1.002** |

mistral moves essentially exactly its ideal KV bytes. D7 moves 28.5% more, and that excess is the
entire 4.6 pp discrepancy the depth-invariance run found. The roof model is not wrong about
mistral; it is wrong about D7 specifically.

Note this is a DRAM-level result and the 33.67% L1 sector waste does not contradict it: sectors
discarded at L1 were largely served from L2, not re-fetched from memory.

**Structural candidate, untested:** D7 has nH/nKV = **7** query-head blocks sharing each KV head
against mistral's **4**. Perfect sharing would put both at 1.000; mistral achieves it, D7 falls
28.5% short. Imperfect L2 reuse across a wider sharing group is the obvious suspect and is not
established here — no metric in this run distinguishes it from alternatives.

## 3. This weakens the GQA-grouped kernel's traffic argument, and should be fed back

The proposed GQA-grouped flash-decode kernel (one block per (kv head, split)) is justified partly by
"each KV slice is read once instead of nH/nKV times". **Measurement says L2 already achieves that**:
mistral reads its KV 1.002x, not 4x; D7 reads it 1.285x, not 7x. So the traffic win available to
that kernel is **at most 28.5% on D7 and approximately zero on mistral** — not the 4-7x the framing
implies.

Its occupancy argument also needs care: grouping by KV head *reduces* the block count from nH to
nKV (28 → 4 on D7, 32 → 8 on mistral), so the parallelism has to come back from the split
dimension. The kernel is not obviously a win on block count alone; it needs S large enough to
overcome the grouping, and that trade is unmeasured.

None of this kills the kernel — the occupancy floor (12%) is real and split-KV's measured wins are
real. It means the case for it rests on **occupancy from splitting**, not on traffic from grouping,
and a proposal that leads with the traffic argument is leading with the weaker half.

## What is unchanged

The three-point f ordering, the +9.94% D7 forfeits at depth 8000, and the refutation of
`cuda/resident.go:223`'s "already fills the device" all stand. This run was about a proposed fix,
and it says the fix was misaimed.

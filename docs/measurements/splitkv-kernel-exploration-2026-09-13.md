# Exploring the kernel: occupancy is not the bound it was assumed to be

Profiled 2026-09-13, nobara-pc. D7 (Qwen2.5-7B, nH=28 nKV=4 hd=128) at depth 8000, split-KV FORCED
ON — the shipped gate refuses this geometry. ncu on all three `splitkv_*` kernels, per layer per
token. Characterisation run, not a verdict test; expectations were stated before profiling and are
recorded below against what came back.

## Measured

| path | µs | occupancy | MB read | GB/s | vs ideal KV |
|---|--:|--:|--:|--:|--:|
| single-block `attn_batched` (OFF) | 554.4 | 12.61 % | 42.09 | 75.9 | 1.28× |
| `splitkv_scores` | 231.2 | **92.87 %** | 16.45 | 71.2 | 0.50× |
| `splitkv_softmax` | 12.4 | 13.26 % | 0.90 | 72.6 | 0.03× |
| `splitkv_vsum` | 215.8 | **9.09 %** | 17.33 | 80.3 | 0.53× |
| **split total** | **459.4** | — | **34.68** | 75.5 | **1.06×** |

Kernel-level 554.4 → 459.4 µs = **1.21×**, consistent with the 1.099× measured end-to-end (attention
is ~51% of the step at this depth).

## Three things this changes

**1. Splitting REDUCES traffic here, it does not add it.** 42.09 MB → 34.68 MB, **18% less**, from
1.28× ideal down to 1.06×. The single-block path's 28.5% excess — the anomaly
`splitkv-sector-efficiency-2026-09-13.md` could not explain — largely disappears when the work is
tiled over keys. Whatever the excess is, tiling fixes most of it. That is the opposite of the
intuition that partial writes and a combine pass cost extra traffic.

**2. The GQA-grouping argument is now spent.** The proposed kernel's traffic case was "read each KV
slice once instead of nH/nKV times". The split path already sits at **1.06× ideal**. There is ~6%
of traffic left to recover on the geometry with the *worst* sharing ratio (7:1). Grouping by KV head
would also cut the block count from nH to nKV, and `splitkv_scores` shows what block count is worth
here — see below.

**3. Occupancy is not the bound, and this is the finding.** `splitkv_scores` runs at **92.87%
achieved occupancy** and still reaches only **71 GB/s — 16% of the 448 GB/s roof**. Meanwhile
`splitkv_vsum` at **9.09%** occupancy moves bytes *faster* (80.3 GB/s). Ten times the occupancy buys
nothing in throughput. Both kernels sit at 18–20% DRAM. So the decode attention path is not
occupancy-starved in the sense the campaign has been treating it as since A1 — it is latency- or
dependency-bound in a way that filling the machine does not touch.

The pre-stated prediction held on the narrow point (vsum occupancy predicted ~8.75% from
nH·hd/(40·32), measured 9.09%), which is why the surprise is credible: the thread-count model is
right about occupancy and occupancy turns out not to be what matters.

## What this implies for the kernel worth building

The flash-decode V-sum — splitting the V reduction over keys, the non-bit-identical fork — was the
candidate because `splitkv_vsum` is 47% of split-path time at 9% occupancy. **This run undercuts
its rationale too.** If raising `scores` to 93% occupancy did not raise its throughput, there is no
measured reason to expect raising `vsum`'s to do so either. The fork's accuracy objection was
already answered (`reduction-tree-accuracy-2026-09-12.md`: blocked folds beat the sequential one,
1.76–4.93×), but its *performance* rationale now needs its own evidence rather than inheriting the
occupancy story.

**The question to answer before writing any kernel:** what are these kernels actually waiting on?
Both are at ~18% DRAM, low compute, and one is at 93% occupancy. A `WarpStateStats` /
`SchedulerStats` pass naming the dominant stall reason would say whether the ceiling is memory
latency (more in-flight requests would help — favours splitting), instruction dependency (favours
restructuring the inner loop), or something else. That is one profile, and it is the cheapest
remaining question in this line.

## Honest scope

One geometry, one depth, one card. `splitkv_scores`'s 92.87% is a strong point but D7 is the
best-case for it (nH=28 × 63 key tiles = 1764 blocks); a geometry with fewer heads or a shorter
context may still be genuinely occupancy-limited, and the 1.5B's 3.8% vsum figure from the original
campaign was measured at 2048, not 8000.

# Staging q in shared memory: the diagnosis was right, the payoff is 2.7%

Changed and measured 2026-09-13, nobara-pc. `cuda/decode_splitkv.cu` `splitkv_scores` + its launch
in `cuda/resident.go`. Follows `splitkv-stall-profile-2026-09-13.md`, which measured this kernel at
**93.5% lg_throttle**, 203.7 warp-cycles per issued instruction.

## The change

`qh = q + h*hd` depends only on `h = blockIdx.x`, so it is block-invariant — yet all 128 threads
re-loaded all hd/4 `float4` of it from global inside the inner loop. Warp-uniform addresses
broadcast, so it cost almost no bandwidth and was invisible in DRAM, L1TEX and sector metrics; what
it cost was LSU **issue** slots. Now staged into shared memory once per block, with the
`__syncthreads()` placed before the divergent early-return so every thread in the block reaches it.

**Bit-identical by construction** — same values, same d-order, same `__fmaf_rn` sequence; only the
address space changes. Confirmed, not assumed: `TestSplitKV_bitIdentical` and
`TestSplitKV_bitIdentical_gemma3` (hd=256, windowed, winStart>0) both PASS.

## Measured, D7 @8000

| | before | after | change |
|---|--:|--:|--:|
| `scores` lg_throttle | 188.36 cyc | 117.67 cyc | **−38%** |
| `scores` long_scoreboard | 7.61 cyc | 31.89 cyc | +319% |
| `scores` duration | 231.2 µs | 218.6 µs | −5.5% |
| `scores` occupancy | 92.87 % | 81.76 % | −12% |
| `vsum` (untouched) | 215.8 µs | 216.7 µs | +0.4% |
| **attention total** | **459.4 µs** | **447.5 µs** | **1.027×** |

Attention is ~51% of the decode step at this depth, so the end-to-end effect is on the order of
**1.4%** — estimated from the kernel figure, not separately measured.

## Reading it honestly

**The diagnosis was correct and the mechanism moved as predicted**: the targeted stall fell 38%.
**The payoff did not follow proportionally**, because relieving the issue throttle exposed the stall
underneath it — long_scoreboard rose 4x, and `scores` is now memory-latency-bound like `vsum`. A
93.5% stall share is not a 93.5% opportunity; it is the top of a stack.

There is also a real cost in the ledger: occupancy fell 12% (shared memory competes for the same
per-SM budget), which partly offsets the issue-rate gain.

## Whether to keep it

Arguments to keep: it is bit-identical with both gates green, the effect is ~10x the A/A floor
(0.27%) so it is real rather than noise, the change is ~10 lines with its reasoning recorded, and it
leaves `scores` in a state where the *next* lever is legible.

Arguments to revert: ~1.4% end-to-end is close to the bar this campaign has reverted at before —
the V-sum ILP unroll went back for "158 vs 160, no gain" — and it carries a PTX regen and an
occupancy regression for that 1.4%.

**Not decided here.** It is a shipped-behaviour change to a kernel with a bit-identity contract, so
it is Francis's call, and the numbers above are the whole basis for it.

## What it opens

`scores` and `vsum` are now bound by the *same* thing — long_scoreboard, memory latency — where
before they were opposite. That makes one lever apply to both: more memory-level parallelism per
warp (prefetching the next K tile while the current one is consumed, or more in-flight loads). That
is a bigger change than this one and it now has a measured target on both kernels rather than one.

## Provenance

Binary rebuilt from this tree; PTX regenerated with `cuda/build_ptx.sh decode_splitkv` at nvrtc
12.9.86, which is what the checked-in `decode_splitkv.ptx` was already built with — the 12.6.85
freeze applies to `moe.ptx`, `glue.ptx` and the gemv family, not to this module. A no-op regen was
verified byte-identical BEFORE the change, so the PTX diff is the change and nothing else.

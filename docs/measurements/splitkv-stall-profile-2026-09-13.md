# Stall profile: the two split kernels have OPPOSITE bottlenecks, and one has a bit-identical fix

Profiled 2026-09-13, nobara-pc. D7 (Qwen2.5-7B) at depth 8000, split-KV forced on. Explicit
`smsp__average_warps_issue_stalled_*_per_issue_active.ratio` metrics — the WarpStateStats section's
CSV output carries only summaries, so the reasons had to be named individually.

## Measured

| kernel | cyc/issued instr | dominant stall | share |
|---|--:|---|--:|
| `splitkv_scores` | **203.69** | **lg_throttle** (LSU issue queue full) | **93.5 %** |
| `splitkv_vsum` | 8.84 | long_scoreboard (memory latency) | 75.5 % |
| `splitkv_softmax` | 14.46 | long_scoreboard | 61.1 % |

Predicted before the run: a throttle rather than long_scoreboard for `scores`, on the grounds that
93% occupancy should already hide plain memory latency. That held.

## This overturns the inference in the previous record

`splitkv-kernel-exploration-2026-09-13.md` concluded: *"if raising scores to 93% occupancy did not
raise its throughput, there is no measured reason to expect raising vsum's to do so either."*
**That is now refuted, and by its own follow-up.** The two kernels are not bound by the same thing:

- `scores` is **issue-throttled**. More resident warps make it worse — they contend for the same
  LSU slots. Occupancy is not merely useless here, it is the wrong axis entirely.
- `vsum` is **latency-bound** (75.5% long_scoreboard) at 9.09% occupancy with only 8.84 cyc/instr.
  More warps in flight is exactly what it wants.

So the flash-decode V-sum split — splitting the V reduction over keys to raise its warp count — has
its performance rationale **restored**, on evidence, not inherited. Its accuracy objection was
already answered separately (`reduction-tree-accuracy-2026-09-12.md`).

## The cheap fix `scores` actually wants, and it is bit-identical

`cuda/decode_splitkv.cu:28-42`. `qh = q + h*hd` depends only on `h = blockIdx.x`, so it is
**block-invariant** — and every one of the 128 threads re-loads the same 32 `float4` of q from
global inside the inner loop:

    const float* qh = q + (long)h * hd;        // same for the whole block
    for (int j = 0; j < d4; j++) {
        float4 qq = q4[j], kk = k4[j];         // q4[j] identical across all 128 threads
        ...
    }

The hardware broadcasts a warp-uniform address, so the *bandwidth* cost is small — and that is
precisely why this did not show up as a DRAM or L1 figure. But `lg_throttle` counts LSU **issue**,
and **half the inner loop's load instructions are redundant q broadcasts**. hd=128 → 32 K loads that
are genuinely per-thread, plus 32 q loads that are not.

Staging q into shared memory once per block (128 threads, hd=128 floats, one each) removes those 32
loads per thread. It is **bit-identical by construction**: same values, same d-order, same
`__fmaf_rn` sequence — only the address space the q value is read from changes. No new reduction
tree, so no fidelity gate and no golden re-base.

**Size unmeasured.** `scores` is 231.2 µs of the 459.4 µs split total (50%), and lg_throttle is 93.5%
of its stall. Halving LSU instructions does not necessarily halve the stall, and this record does not
claim it will — the mechanism is measured, the payoff is not.

## Where that leaves the kernel question

Two separable pieces of work, in cost order:

1. **Stage q in shared memory in `splitkv_scores`** — small, bit-identical, targets a measured 93.5%
   stall, no fidelity gate. Needs a PTX regen (CLAUDE.md: only reproducible at nvrtc 12.6.85 via the
   documented pip venv; this box is on 12.9).
2. **Flash-decode V-sum** — larger, non-bit-identical, needs the §3.2-style fidelity gate and the
   cross-M identity conditions. Now has a real performance rationale (75.5% long_scoreboard at 9%
   occupancy) rather than an inherited one.

The GQA-grouped variant remains unsupported: the split path already reads 1.06× ideal traffic, so
grouping has ~6% to recover, and it would cut block count from nH to nKV.

## Scope

One geometry, one depth. `scores`'s lg_throttle share may differ where the key-tile count is lower.
The q-broadcast structure is geometry-independent, but its cost as a *share* of the stall is not.

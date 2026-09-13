# Flash-decode V-sum spike: +40-43% on the attention kernel — REOPEN, per the pre-registered rule

Measured 2026-09-13, nobara-pc. D7 (Qwen2.5-7B, nH=28 nKV=4 hd=128) at depth 8000, split-KV forced
on, binary from this tree, driver `595.91.07`. Kill criteria fixed in advance by
`docs/scoping-decode-tree-recanon.md` §6: **<5% kill, 5-15% park, >15% reopen**.

## Result

| S | scores | softmax | vsum_partial | vsum_combine | vsum total | partial occ | **total** | **vs baseline** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 218.9 | 12.4 | 84.3 | 2.5 | 86.8 | 34.45 % | 318.1 | **+40.68 %** |
| 8 | 219.9 | 12.4 | 83.7 | 2.7 | 86.4 | 40.74 % | 318.7 | **+40.41 %** |
| 16 | 218.6 | 12.3 | 79.0 | 3.6 | 82.6 | 47.39 % | **313.5** | **+42.73 %** |

Baseline is the bit-identical split path after q-staging: **447.5 µs**, of which vsum is 216.7 µs.
The V-sum alone goes **216.7 → 82.6-86.8 µs, 2.50-2.62x**. `scores` and `softmax` are untouched and
measure unchanged, which is the internal control: only the kernel that was replaced moved.

**Verdict: REOPEN.** Nearly 3x the threshold, at every S tested.

## The trend is flat, and that is the actionable part

S=4 already captures essentially the whole win. Going to S=16 buys 2 more percentage points while
partial-occupancy climbs 34% → 47%, and the combine grows 2.5 → 3.6 µs as the partials multiply. So
the gain is **not** proportional to warp count — most of it arrives with the first split.

**Operating point: S=4, not the largest S.** Smaller partials buffer, fewer boundaries to pin in the
cross-M tiling table, and 98% of the benefit.

## Prediction failure, recorded because it was confident and repeated

I predicted the low end (5-15%) twice, reasoning from the q-staging result: relieve the dominant
stall, the next one takes over, collect ~5%. **That was wrong by roughly 8x.**

The reasoning failed because it generalised across two kernels with opposite bottlenecks. `scores`
is **issue-throttled** (93.5% lg_throttle) — more warps contend for the same LSU slots and actively
hurt. `vsum` is **latency-bound** (75.5% long_scoreboard at 9% occupancy) — more warps in flight is
exactly the remedy. `splitkv-stall-profile-2026-09-13.md` measured that distinction and stated it;
I then discounted it against the more recent q-staging experience. The lesson is narrow and worth
keeping: *a stall-reason profile is per kernel, and an intuition calibrated on one kernel's bound
does not transfer to another kernel with a different one.*

Second, smaller wrong call: I expected the combine pass to eat the gain. It is **0.8% of total**.

## What this does NOT establish

1. **Fidelity.** The spike is not bit-identical, and no fidelity gate has been run. It needs the
   §3.2-style agreement / flips / KL comparison against an f32 reference that `attn_fused` shipped
   under. `reduction-tree-accuracy-2026-09-12.md` predicts it should pass — a blocked fold measured
   1.76-4.93x closer to f64 than the sequential one — but predicted is not measured, and this
   ordering (perf first, fidelity second) is only defensible because the spike is opt-in and
   unreachable in a stock binary.
2. ~~**End-to-end.** ~46 tok/s against today's 39.4 at depth 8000 is ARITHMETIC from the kernel
   figure and attention's ~51% share, not a served measurement.~~

   **CORRECTED 2026-09-13, and the correction is the interesting part.** A served A/B *was* run the
   same day — two `serve` arms over the same 8000-token prompt, 48 greedy tokens each — and it
   measured **38.79 → 44.83 tok/s, +15.6%**. That number then went into a commit message while this
   section still said end-to-end was unestablished, because **the run's output was never written
   anywhere**: no log, no record, only a terminal. A figure whose only copy is a scrollback is not
   evidence, which is exactly what CLAUDE.md means by "archive the log; do not leave it in `/tmp`" —
   and the failure mode here was worse than losing it, because the unrecorded number kept being
   quoted while the committed document went on contradicting it.

   Re-run on a quiet box to give it a re-readable home, `~/goinfer-logs/vsum-served-ab-20260913-103436.log`:

   | arm | tok/s | continuation |
   |---|---:|---|
   | exact (bit-identical vsum) | 38.60 | "…The problem is…" |
   | spike S=4 | 44.84 | "…The following is…" |

   **+16.2%**, against +15.6% unlogged — consistent, and the kernel figure predicts it to within a
   point. Two caveats that keep this honest: the prompt prefix is cached by the preceding warm
   request, so this is a decode-rate measurement and the "incl prefill" in the harness output is
   misleading; and **both arms emit degenerate text** on this prompt, so the run says nothing about
   quality — the arms differ from the first content word, which is the vacuity check passing and
   nothing more. Quality is the fidelity gate's job (`vsum-split-fidelity-PREREGISTERED.md`).
3. **One geometry, one depth.** D7 only. A geometry where the single-block path already wins
   (phi3-mini class, high KV traffic per key) has no reason to benefit and is untested here.

## Provenance

Three S values, 32 launches each, medians; `ncu -k regex:splitkv --section SpeedOfLight --section
Occupancy`; grid asserted per kernel; every run checked for `DECLINED` in its log, because an
earlier attempt silently profiled the CPU fallback for ten minutes after a 0-byte allocation took
the resident path down. Raw CSVs in the session scratchpad; the sweep log is
`goinfer-logs/vsum-spike-20260913-081214.log`.

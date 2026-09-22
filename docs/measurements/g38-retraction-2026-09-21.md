# RETRACTION: G38's "LM-head GEMV, 10-17x over roofline" finding was a Poll(false) vs Poll(true) bug in my own test, not a real cost

`docs/QUEUE.md` G38 (pushed `db705458`) claimed WebGPU greedy decode's LM-head GEMV alone was 72-82% of the token and ~10-17x over its
own VRAM-bandwidth roofline, "not root-caused" for lack of GPU profiling tools. **Asked to root-cause it. Root-causing it is what found
the bug in the original measurement, not in production code.** RTX 2070 SUPER, driver 595.91.07, Qwen2.5-Coder-1.5B `int8int8`.

## What root-causing it actually found

1. **`TestGEMVNSweep`** (in-process, N=512..200000, one process): a wild, non-monotonic cliff between N=65536 and N=100000, with the
   SAME N reading up to 15x apart depending on what ran before it in the same process. This alone should have been a red flag before
   trusting anything — it pointed at the SWEEP's own methodology (allocating and releasing a differently-sized weight buffer for every
   N in one process), not a property of any single N.
2. Rebuilt the sweep **one N per process** (matching this repo's own separate-process-per-arm convention, for exactly the contamination
   reason above): the cliff vanished entirely. N=151,936 read **874 us at ~267 GB/s** — matching the roofline `docs/QUEUE.md` computes
   in the (retracted) section below, not missing it by 16x.
3. **`TestGEMVLMHeadIsolation`** (new): called the REAL model's own `r.lmHead` buffer directly, standalone, both right after load and
   after 200 real decode steps. Both read 864-887 us — the SAME clean number, regardless of whether anything else was resident. This
   ruled out co-residency / VRAM fragmentation from the rest of the model's buffers as an explanation.
4. With the kernel cleared and the real buffer cleared, the only remaining suspect was the comparison itself
   (`TestDecodeArgmaxHeadroom`'s `Run()` vs `RunNoLogits()` delta). Reading `RunNoLogits`'s own body: it calls
   **`c.device.Poll(false, nil)`** — non-blocking — while every other call site in `gpu/decoderunner.go` (`Run`, the sample paths,
   `argmaxOfLastLogits`) calls **`Poll(true, nil)`**. Timing `RunNoLogits` back-to-back without an explicit blocking poll measures
   CPU-side submission time only; the GPU keeps executing the 28-layer trunk asynchronously in the background while the "measurement"
   has already returned. The original delta was therefore comparing **"wait for everything" against "don't wait at all,"** attributing
   the trunk's own real GPU execution time to the LM head.

## The corrected measurement

Fix: an explicit `c.device.Poll(true, nil)` after `RunNoLogits`, and arms alternated in small blocks (8 rounds x 40 calls each,
positions advancing sequentially) rather than two long back-to-back blocks, so session drift cannot bias one side.

| run | `Run()` | `RunNoLogits()` (fixed) | delta | copy+map alone | implied GEMV |
|---|---:|---:|---:|---:|---:|
| 1 | 11,880 us | 11,313 us | 567 us = 4.8% | 386 us = 3.2% | 181 us = 1.5% |
| 2 | 11,492 us | 10,951 us | 541 us = 4.7% | 234 us = 2.0% | 307 us = 2.7% |
| 3 | 11,788 us | 11,344 us | 444 us = 3.8% | 385 us = 3.3% | 59 us = 0.5% |

**Reproduced across three independent runs at 3.8-4.8%.** The LM-head GEMV's own implied cost is close to zero in every run — the whole
delta is essentially the copy+MapAsync+host-scan, which is what the ORIGINAL (correct, un-retracted) G38 conclusion was actually about:
building a device-argmax kernel for greedy recovers at most ~4-5% of the token, not enough to justify it. **That bottom line survives.
Everything about "the LM-head GEMV is the real cost, 10-17x over roofline, unattributed" does not — it was never there.**

## What this does and does not change

- `docs/QUEUE.md` G38: retraction notice added in place, figures marked not-to-cite. Not deleted — the mechanism of the mistake (a
  non-blocking poll timed as if synchronous) is itself worth keeping on record.
- `docs/tasks/red-october.md` R10: corrected to drop the "LM-head GEMV, 10-17x, not root-caused" framing.
- **The decode side of R10 is now essentially closed for this pass.** G35 (quantize fix) + G36 (attention key-split) are real, shipped,
  2.5x at 1k context. On-device argmax is confirmed low-value (~4-5%, not worth building) for the RIGHT reason now. No further decode
  lever is currently named.
- **R10's prefill investigation is untouched** and is the only genuinely open item left in the brief.
- `gpu/decode_argmax_headroom_test.go`, `gpu/gemv_n_sweep_test.go` (env-var-driven, one N per process), `gpu/gemv_lmhead_isolation_test.go`,
  and `DecodeRunner.LMHeadForTest()` are kept in tree as committed instruments — the sweep and isolation tests are exactly what a future
  "is this kernel actually slow" question should reach for BEFORE trusting a single before/after delta.

## Lesson, stated once so it does not need re-deriving

**An async API's "did I wait for the work" flag is part of what a timing comparison measures, not a detail below the level a benchmark
needs to check.** `RunNoLogits`'s non-blocking poll is correct for its real callers (who do not need synchronous timing) — the bug was
using it inside a timing comparison against a function that DOES block, without checking. The same class of mistake this repo has
already recorded once for concurrency instruments that serialize the property under test (`concurrency-tests-that-serialize` memory) —
here the instrument didn't serialize anything, it just didn't wait, which is the opposite failure mode with the same root cause: not
reading the primitive's own contract before trusting a number it produced.

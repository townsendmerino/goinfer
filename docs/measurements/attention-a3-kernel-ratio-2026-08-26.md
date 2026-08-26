# A3 kernel ratio — is an f32 attention path worth its divergence flag?

**Answer: yes.** The acc64 attention kernels are **8.14× slower than f32** at long-context prefill
shapes, which is ~2.2× worse than the "~3.7×" the code comments assume. At the 70% attention share
measured in the K=8192 profile, that is **~2.59× end-to-end prefill**. Queue entry **G23**.

**Box:** M1 Pro, dense qwen2.5-coder-1.5b shapes, goinfer `53e1c6d`. Instrument:
`decoder/g23_attnkernel_test.go` (`GOINFER_G23=1`), best-of-5.

## The number

Shapes are what G20's tiling actually calls at an 8k prompt: `kt=256, hd=128, nKeys=8192`.

| | acc64 (direct strided read) | f32 + the gathers acc64 skips | ratio |
|---|---|---|---|
| QK | 57.2 ms | 10.7 ms | **5.32×** |
| AV | 142.6 ms | 13.8 ms | **10.33×** |
| combined | 199.8 ms | 24.5 ms | **8.14×** |

Amdahl on the profile's 70% attention share ⇒ **2.59×** end-to-end long-context prefill.

## What makes this number trustworthy — three things that were nearly wrong

**1. The arms compute the same thing, and it is asserted, not assumed.** QK agreement:
**cosine 1.000000000, maxAbs 3.34e-06**. The test *fails* below 0.999999 rather than reporting a
ratio between two different computations.

**2. Parallelism is equalized — the first run was not, and it lied by 2×.** `MatmulBT` fans out via
`parallelCols`; the acc64 kernels are plain serial loops. The first pass compared serial acc64
against parallel f32 and reported **17.6×**. That was disbelieved *because* it exceeded the
documented 3.7× so implausibly, and the cause was found by reading both kernels. With `MatmulBT`
forced serial the honest ratio is 8.14×.

**3. The f32 arm pays the work acc64 avoids.** `MatmulQK/AVAcc64` read K/V directly by stride,
"skipping a kh gather entirely" and "skipping a vt gather+transpose". Both gathers are INSIDE the
timed f32 arm. Omitting them would have flattered f32 by work the real path cannot avoid.

## The residual mystery, stated rather than used

The kernel comments say acc64 is "~3.7× slower than f32". Measured here: **8.14×**. Not a
contradiction to wave away — the likeliest explanation is that 3.7× was taken at DECODE shapes,
and acc64's strided direct read degrades at `nKeys=8192` in a way A1's move (c) fixed for decode
and nobody re-measured at prefill depth. **The 2.59× estimate does not depend on resolving this**
(it uses the measured ratio, not the documented one), but anyone quoting 8.14× as a general acc64
penalty should measure their own shapes first.

## Separately measured, deliberately NOT multiplied in

f32 additionally parallelizes *within* a matmul: **4.01×** on QK alone. This is **not additive**
with the ratio above, because the real prefill path already fans acc64 out across heads (G16/G20's
worker pool). Reported so the two effects are not silently conflated into a headline.

## What this does and does not license

It clears the bar fixed **before** the measurement was taken (G23: "near 3.7× the flag's surface is
earned, near 1.3× it is not"). It does **not** license shipping f32 attention as a default: A1's
guarantees — spec-decode verify == sequential greedy, and decode == prefill — hold only for acc64,
which is why A3 is an off-by-default, documented divergence in the `--metal-fast-prefill` mould.

**Build constraint carried forward:** the f32 branch is single-threaded by construction (its
per-kv-group `kh`/`vt` gather is shared mutable state). A3 must gather once per kv-group and then
fan the group's query heads across workers reading it, or a kernel 8× faster still loses to
parallel acc64.

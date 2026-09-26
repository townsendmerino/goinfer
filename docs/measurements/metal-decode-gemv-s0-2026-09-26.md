# Metal decode's fixed cost — the int4 GEMVs are bound by per-weight work, not bandwidth (S0 for R18, 2026-09-26)

**Question.** After R17, Metal decode at 3900 keys is 0.87× Ollama on the 1.5B and 0.81× on the 7B, and at depth 128
it is 0.86× / 0.87×. Against MLX it is **0.66× on the 1.5B and 0.57× on the 7B at depth 128**
(`benchmarks.md`: MLX 109.8 / 37.9 tok/s). The gap is now the same size at short and long context, so it is a fixed
per-token cost. S0 of decode (`metal-decode-decomp-2026-09-25.md`) put that cost in the GEMVs: 10.3 ms of a 12.9 ms
1.5B token and 41.7 of a 45.5 ms 7B token. **Why are they slow: bandwidth, or arithmetic?**

**Answer: per-weight work, not memory traffic.** On real shapes, each int4 GEMV runs 1.5–2.0× slower than a twin that
reads exactly the same weight bytes in the same pattern. The twin drops the per-weight unpack and multiply-accumulate,
and also the per-weight activation reads from threadgroup memory. The int8 LM head, a GEMV on the same machine, runs at ~90% of the
streaming-read ceiling. The down projection has a second, separate problem: its coal kernel's access pattern tops
out at 108–127 GB/s even without the arithmetic.

## Method

`TestMetalGEMVS0` (`metal/gemv_s0_test.go`) times each decode GEMV category standalone at its real (N, K):
- the production kernel;
- a **loads-only twin** that reads the same device memory with the same thread mapping (weight words, f16 group
  scales, and the coal kernel's per-word activation bytes) but folds it with an XOR instead of unpacking and
  multiplying;
- the int8 LM head kernel (`gemv_w8a8_coal`);
- a plain coalesced streaming read of 512 MB.

Weights are random and rotated over ≥ 256 MB of copies, so no dispatch finds its matrix in the system-level cache,
as in decode, where every layer's matrix is distinct. The time per dispatch comes from a command buffer of
back-to-back dispatches (GPU timestamps), median of 7. Bytes include the scales: 0.5625 bytes/weight for int4 (nibbles
plus one f16 per 32), and N·K + 4N for the int8 head.

M1 Pro 16 GB, macOS 26.6.2, goinfer `17efc946` + this test file, 2026-09-26 ~08:10 local. load1 was 2.33 at the
start, just above the 2.0 gate; the in-sequence cross-check below agrees regardless. Raw:
[`run1.log`](metal-decode-gemv-s0-2026-09-26/run1.log).

## Result

Streaming-read ceiling: **178–182 GB/s** (159 at 64 K threads).

| model | category | kernel | MB | production | GB/s | loads-only twin | GB/s | production ÷ twin |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1.5B | qkv | `gemv_w4a8_sa` | 1.8 | 0.025 ms | 70 | 0.017 ms | 107 | 1.53× |
| 1.5B | o | `gemv_w4a8_sa` | 1.3 | 0.021 | 64 | 0.014 | 97 | 1.51× |
| 1.5B | gate/up | `gemv_w4a8_sa` | 15.5 | 0.172 | 90 | 0.088 | 176 | **1.96×** |
| 1.5B | down | `gemv_w4a8_resid` (coal) | 7.7 | 0.115 | 68 | 0.072 | 108 | 1.60× |
| 1.5B | LM head | `gemv_w8a8_coal` (int8) | 234.0 | 1.456 | **161** | — | — | — |
| 7B | qkv | `gemv_w4a8_sa` | 9.3 | 0.093 | 100 | 0.061 | 152 | 1.52× |
| 7B | o | `gemv_w4a8_sa` | 7.2 | 0.076 | 96 | 0.050 | 144 | 1.51× |
| 7B | gate/up | `gemv_w4a8_sa` | 76.4 | 0.690 | 111 | 0.407 | 187 | **1.69×** |
| 7B | down | `gemv_w4a8_resid` (coal) | 38.2 | 0.515 | 74 | 0.300 | 127 | **1.71×** |
| 7B | LM head | `gemv_w8a8_coal` (int8) | 545.6 | 3.339 | **163** | — | — | — |

**The standalone bench reproduces in-sequence costs.** Per-layer time × 28 layers against S0's no-op-method category
times:

| | gate/up | down | qkv | o | LM head |
|---|---|---|---|---|---|
| 1.5B, ms (this bench / S0) | 4.82 / 4.66 | 3.22 / 3.06 | 0.70 / 0.69 | 0.59 / 0.55 | 1.46 / 1.33 |
| 7B, ms (this bench / S0) | 19.3 / 19.42 | 14.4 / 14.19 | 2.60 / 2.72 | 2.13 / 2.16 | 3.34 / 3.21 |

All within ~5%, so this is a faithful prototyping bench, not a microbenchmark that omits the real pressure.

## Reading

- **The per-weight work is the cost.** Apple GPUs have no 4- or 8-bit dot-product instruction. Each weight costs a
  shift, a mask, a subtract of 8, an int multiply and an add (`UNP8` / `W4A8_BODY`), plus a read of its activation from
  threadgroup memory. The int8 head does half as much per byte, and it runs at the bandwidth bound.
- **This is not the dispatch-count problem an older note claimed.** S0 measured the pure dispatch floor at 0.65–1.1 ms
  per token. The GEMVs are 80–92% of a depth-128 token, and arithmetic is 35–50% of their time.
- **The down projection (coal) is doubly slow.** Its loads-only twin reaches only 108–127 GB/s, because every word
  re-reads 8 activation bytes and a scale from device memory, where the SA family stages activations once in
  threadgroup memory.
- **Size of the lever.** If the int4 GEMVs ran at their loads-only rates:
  - 7B: GEMVs 38.5 → 22.9 ms per token, and the token ~46 → ~31 ms (32 tok/s). That is ~1.3× Ollama's 24.9 and
    ~0.86× MLX's 37.9.
  - 1.5B: GEMVs 9.0 → 5.4 ms, and the token 13.8 → ~10.2 ms (~98 tok/s). That is ~1.16× Ollama and ~0.89× MLX.

  Fixing down's access pattern as well is worth another ~2 ms on the 7B.
- **A faster kernel can be bit-identical.** Each group's (and each coal word's) integer sum is exact, and today's
  kernels then accumulate `float(sum) × scale` in a fixed lane order. Any kernel that produces the same integer sums
  and keeps that order is bit-identical. For example, subtracting 8·Σa once per group removes the per-weight
  subtract, and computing the products in f32 via masks and pre-scaled activations (MLX's `qmv` form) replaces the
  shifts. Every product and partial sum there is an integer below 2²⁴, so it is exact in f32. That makes the design
  space bit-identical by construction: no fidelity gate, only a bit-identity check.

## Prior art this has to answer (read before designing)

- **July's cgo-free spike** (`docs/completed/task-metal-cgofree-spike.md`) called the int4 GEMV **int-MAC-bound**:
  a batch-k GEMM gave T_k ≈ k·T_1 with unpack only ~20% of per-weight cost. It concluded that ~71–77 tok/s is
  "near the practical ceiling for cgo-free W4A8 decode on this GPU".
  - Its "Stage C" (fold −8·Σa once per group, broadcast activations) was *predicted* to underdeliver. It was never
    built.
  - Its "Stage B" (tile repack) was 164 → 118 µs **in isolation but zero end to end**, and not bit-identical. So
    anything here must be graded in sequence and end to end, never in isolation. This bench's in-sequence agreement
    (above) is necessary, not sufficient.
- **R1's W4F16 lane** (`w4f16-decode-speed-2026-09-21.md`) replaced the int multiply with an f16/f32 FMA but kept one
  row per simdgroup, the per-weight shift/mask/subtract/convert and the per-weight threadgroup activation read. It
  gained **2–4%**, so the int multiply alone is not the wall.
- **MLX runs the same model at 109.8 tok/s on this machine**, 1.5× goinfer, so the ceiling is not the GPU's. Its
  4-bit `qmv` keeps activations **in registers**, loaded once and reused across **several rows per simdgroup**. It
  pre-scales them so the weights are **masked in place, never shifted**, and it does f32 FMAs, with the offset in
  one per-group correction.

None of Stage A/B/C or R1 tested that combination: register-resident activations, several rows per simdgroup, and a
shift-free f32 FMA. It can be bit-identical here if each lane keeps owning whole groups in today's order. That is the
thesis of R18.

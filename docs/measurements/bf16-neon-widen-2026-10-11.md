# The NEON bf16 widen on an M1 Pro, 2026-10-11

**Status: exploratory.** Taken by day at load average 2.0 to 2.5 with the timing lock held, and the two kernel arms ran as consecutive blocks, not interleaved: this repo's rule for a graded kernel number is an interleaved in-process A/B on an idle box (CLAUDE.md, "Run budget"), and the Mac's by-day rule forbids quoting a timed Mac number as a result. The direction is not in doubt (the kernel ranges do not overlap, and the served rounds agree to 2%); the sizes are readings, not results. A graded re-read would be an interleaved run at night.

**Machine and versions.** MacBook Pro, M1 Pro, 16 GB, darwin/arm64, 2026-10-11 21:22 to 21:27 PDT. aikit main at 3a30656 (v1.70.0, the NEON widen, the gpu pin bump); goinfer main at 45f3d8e7. The arm64 NEON widen is `linalg/widen_bf16_arm64.s` (`cvtBF16ToF32x8NEON`, SHLL/SHLL2, sixteen per iteration); before it arm64 widened bf16 with the scalar loop of v1.70.0.

## 1. Kernel: `go test -run '^$' -bench WidenBF16 -count=10 ./linalg` (a 32 x 4096 tile)

| benchmark | median ns/op (range) | MB/s |
|---|---|---|
| `BenchmarkWidenBF16` (NEON) | 7,264 (6,506 to 8,263) | 36,099 |
| `BenchmarkWidenBF16_scalarLoop` | 56,377 (56,162 to 57,364) | 4,650 |

Scalar over NEON, medians: **7.76 times.** The ranges do not overlap.

## 2. Served: `~/models/qwen2.5-0.5b-instruct` (bf16 safetensors), `serve --backend cpu`

One prompt, temperature 0, 128 tokens, one warm-up then one timed request per server start, three rounds with the arm order rotated. Decode tok/s is first to last streamed delta, client side. Resident MB is `ps` RSS after the timed request.

| arm | decode tok/s (3 rounds) | median | resident MB (median) |
|---|---|---|---|
| `--quant f32`, aikit main | 12.80 12.91 12.93 | 12.91 | 4,412 |
| `--quant bf16`, aikit main (NEON) | 35.95 36.37 36.76 | 36.37 | 1,562 |
| `--quant bf16`, aikit v1.70.0 | 19.43 19.51 19.56 | 19.51 | 1,589 |

- bf16 NEON over bf16 v1.70.0: **1.86 times** (per round 1.85, 1.86, 1.88).
- bf16 NEON over f32: **2.82 times** (per round 2.81, 2.81, 2.85), at 0.35 of f32's resident memory.
- Greedy reply byte-equal across all three arms and all nine runs.

## What this does and does not say

- **Says:** on this Mac the scalar widen was most of the cost of `--quant bf16` decode on arm64; the NEON one removes it, and bf16 then decodes faster than f32 (the f32 arm reads four times the bytes of the bf16 arm's weights, which is the likely reason, not a measured one).
- **Does not say:** how it compares with int4 (not run), prefill (not run), a larger model, or any other core. amd64 has had its AVX2 widen since v1.70.0 and was not part of this run. One prompt, three rounds.
- **Correctness** is separate and proven on every architecture: `TestWidenBF16_exhaustive` (all 65,536 patterns from offsets 0 to 7) and the bit-identity of the 16-bit `WeightMat` against f32 (`TestWeightMat_half16_bitIdenticalToF32`).

# CPU decode QK: a 16-key acc64 kernel on amd64 (bit-identical speed change), 2026-11-23

Written 13:43 PST, before the timed run. Why: perf finding 12's kernel split (`docs/measurements/cpu-kv-f16-2026-11-23/README.md` § "Kernel
split") read amd64's acc64 QK kernel at 16.0 GB/s against AV's 29.9 GB/s on the same bytes (Phi-3's layer at depth 3,900, 16 workers): one
accumulator chain per 4-key block makes it FMA-latency-bound. The change, in aikit's `linalg`, interleaves four 4-key blocks as four
accumulators; each key is still one f64 lane over ascending d, so every score is bit-identical (`TestQkAcc64AVX16_matchesAVX4AndGo`).

**Rule:** the four-condition rule (CLAUDE.md § "Measurement discipline"): bit-identical outputs; scratch and allocated memory not higher
(the kernel holds its state in registers and allocates nothing); faster on at least 9 of 10 timed pairs; it replaces code (the 4-key kernel
now takes only the N%16 remainder).

**Timed run:** `run_ab.py`, `BenchmarkDecodeAtDepth/grouped_on` on the CPU, f32 KV, A = the decoder test binary on aikit's release, B =
the same goinfer commit built on the change. Depth is reached by `GOINFER_BENCH_FILL=1` (seeded random K/V rows; decode attention's work
does not depend on the cached values), 32 timed tokens a cell, one process a cell, A B B A × 5 blocks = 10 pairs per model and depth.
- **Decision cells:** depth 3,900 on Qwen2.5-Coder-0.5B (int4) and Phi-3 mini (q4k): faster (B ÷ A > 1) on at least 9 of 10 pairs each.
- **Report cells:** depth 128 on both, where attention is a small share: reported, median B ÷ A ≥ 0.99 expected.
- Each cell waits for load under 1.0; a cell starting above 8.0, or with swap growing during it, is void and re-run in place.

## Result, read 2026-11-23 17:26 PST: FAILS the four-condition rule (Phi-3 slower on every pair)

80 cells, 16:33–17:08 PST under the exclusive lock, A = `dec-A-v1.78.0-0854630a6`, B = `dec-B-qk16-0854630a6`; start load 0.67–0.99 on every
cell, no swap, no void cells. Cells `ab/cells.tsv`, log `ab/run.log`, graded by `grade.py`. All cells ran after the 15:16 PST crash
and the chain's re-launch.

| model, depth | A tok/s | B tok/s | median B ÷ A | pairs faster |
|---|---|---|---|---|
| 0.5B, 3,900 | 22.06 | 24.51 | **1.106** | 10/10 |
| 0.5B, 128 | 55.45 | 55.45 | 1.000 | 4/10 |
| Phi-3 mini, 3,900 | 3.105 | 2.945 | **0.948** | **0/10** |
| Phi-3 mini, 128 | 9.082 | 8.986 | 0.990 | 0/10 |

**Verdict: not shipped as built.** The 0.5B gains 10.6% at depth, Phi-3 loses 5.2%. **Candidate mechanism, not yet verified:** Phi-3's key
rows are 3,072 floats (12 KiB) apart, a multiple of 4 KiB, so the sixteen rows the new kernel reads at once fall in the same L1 sets
(Zen 2: 8-way), where the 4-key kernel's four did not; the 0.5B's rows are 512 B apart. `BenchmarkQkAcc64Stride` (aikit) times both
kernels at Phi-3's head shape at that stride and padded by 16 floats, to test it.

## Mechanism check, read 2026-11-23 17:51 PST: L1 set aliasing at a 4 KiB-multiple key stride

`BenchmarkQkAcc64Stride` (aikit `linalg/attn_acc64_stride_amd64_test.go`), one thread, one head of 96 dims × 3,900 keys, 200 calls a
sample, 5 samples a cell run in sequence (direction, not size; log `stride.log`), 17:46 PST under the exclusive lock:

| key row stride | 4-key kernel, median | 16-key kernel, median | 16-key ÷ 4-key |
|---|---|---|---|
| 3,072 floats (12 KiB, Phi-3's kvDim) | 193 µs | 308 µs | 1.6× slower |
| 3,088 floats (padded by 64 B) | 191 µs | 125 µs | 1.5× faster |
| 128 floats | 225 µs | 102 µs | 2.2× faster |

At 3,072 every 16-key sample (262–325 µs) is slower than every 4-key sample (124–249 µs); a 64 B pad reverses it. The 16-key kernel's
sixteen live rows, all at one offset mod 4 KiB, contend for one 8-way L1 set. What to do with the kernel is with the coordinator.

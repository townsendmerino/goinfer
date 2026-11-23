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

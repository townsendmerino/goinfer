# CPU f16 KV read in place: registration, 2026-11-23 13:59 PST, before any timed run

**Question.** With CPU decode reading an f16 cache's rows in place through aikit's `MatmulQKAcc64F16` / `MatmulAVAcc64F16` (no
whole-history widen into scratch), is `--kv f16` faster than `auto`'s f32 on the CPU? Perf finding 12 read the old f16 read at
0.729× f32 at depth 3,900 on the 0.5B (`docs/measurements/cpu-kv-f16-2026-11-23/README.md`).

**Outputs:** bit-identical to the f16 read it replaces (`TestKVF16_decodeReadsInPlace` through `Model.forward`; the aikit kernels'
`TestMatmulQKAcc64F16_matchesF32OnWidened` / `TestMatmulAVAcc64F16_matchesF32OnWidened`). So the 0.5B's f16 fidelity PASS at K = 3,900
(same record, § "Fidelity") carries unchanged. **Phi-3 mini's** f16 fidelity has not been measured: `TestCPUKVF16GateVsReference` cell
`P3M:3900` runs before any claim for Phi-3 (about 2.5 h, by day).

**Speed.** `run_ab.py` in `docs/measurements/cpu-qk-multichain-2026-11-23/` with one binary for both arms (built on the change and on the
multi-chain QK), A = f32 KV (`GOINFER_BENCH_KV` unset), B = f16 (`GOINFER_BENCH_KV=f16`), `GOINFER_BENCH_FILL=1`, 32 timed tokens a cell,
A B B A × 5 blocks = 10 pairs per model and depth, Qwen2.5-Coder-0.5B (int4) and Phi-3 mini (q4k), depths 128 and 3,900, under the
exclusive lock; the void rule as there.
- **Bar** (the lossless opt-in bar, CLAUDE.md § "Measurement discipline"): at depth 3,900, median B ÷ A ≥ 1.05 with at least 9 of 10
  pairs over 1, on each model. AMBIGUOUS → PARKED: median 1.03–1.05.
- **Reported, no bar:** depth 128 (attention is a small share there), and host bytes: each arm's K/V bytes at depth (f16 is half of f32
  by construction; read from the cache by the fidelity gate) and peak RSS.
- **Stopping rule:** fixed N, no early stop.

**What each outcome means.** PASS on a model: `--kv f16` on the CPU is a faster lossless-within-the-gate opt-in for that model, stated
with its host bytes; making it `auto`'s CPU pick is Francis's call afterwards. PARKED or under: `--kv f16` stays a memory option only, the
in-place read still ships (it replaces a slower read of the same bytes, bit-identically).

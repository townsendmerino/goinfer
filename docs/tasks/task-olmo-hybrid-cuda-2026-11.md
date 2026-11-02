# Olmo Hybrid on CUDA: NegEigval and ONormEps in CUDA's DeltaNet (2026-11)

**Registered 2026-11-02 06:56 PST, before any code (nobara-sb01).** Francis's direction of 06:31, relayed by the coordinator: implement
`linear_allow_neg_eigval` (NegEigval) and the output norm's own epsilon (ONormEps) in CUDA's Gated-DeltaNet, gate it against the CPU,
then claim both in `program.CUDA`, so Olmo Hybrid runs resident on CUDA again and `--kv i8` loads. This work has its own card budget,
separate from OP4 on CUDA's.

## Why

OP4cu-c1 (`docs/tasks/task-forward-architecture-2026-10.md` §4.9a) made `program.CUDA` the CUDA build's admission table. CUDA's DeltaNet
read neither setting, so Olmo Hybrid has declined to the CPU since then. Before c1 it ran resident and computed the family differently from
the CPU:
- **NegEigval:** the CPU doubles the write gate, `beta = 2·sigmoid(b)`, giving the range [0, 2).
- **ONormEps:** the CPU's gated output RMSNorm uses 1e-5, while the release's `rms_norm_eps` is 1e-6.

Its only CUDA tests were a smoke test and a prefill-vs-decode check, and neither compared against the CPU.

## What changes

- **NegEigval.** `delta_gates` and `delta_gates_rows` (`cuda/deltanet.cu`) take a `betaScale` argument, 1 or 2, and multiply the sigmoid by
  it. Multiplying by 1 is exact and multiplying by 2 is exact, so every other DeltaNet family's output is unchanged bit for bit.
- **ONormEps.** The `delta_gnorm` and `delta_gnorm_rows` launches pass the layer's own epsilon instead of the model's `NormEps`.
- **Where the values come from.** Both are read at build from each layer's `OpDeltaNet` parameters in the program, not from the family.
- **The PTX.** `deltanet.ptx` is regenerated with `build_ptx.sh` and the NVRTC it was built with (12.9.86). The unchanged source was
  first rebuilt from that toolchain byte-identical on 2026-11-02.
- **The claims.** `program.CUDA` sets `DeltaNegEigval` and `DeltaONormEps`.
- **Named outcome change, the inverse of c1's:** `olmo_hybrid` goes back to resident on CUDA in the admission and decision censuses and in
  identity's cells. `--kv i8` with it loads.

## Gates

- **O-1, the kernel chain (card, seconds).** `cuda/kernels` `TestDeltaNetKernels_cpuParity` gains a second geometry, Olmo Hybrid-7B's
  (nk = nv = 30, hk = 96, hv = 192). It runs with `NegEigval` and `ONormEps` 1e-5 beside an `eps` of 1e-6. Its value scale is small
  enough that the output norm's mean square sits within 10x of the epsilon; the test logs the measured figure. The bars are unchanged:
  each stage's cosine at least 0.999999, and maxAbs/rms at most 1e-3. The Qwen geometry keeps its run, with `betaScale` 1.
  - **Red first:** the Olmo case run with `betaScale` 1 must be red on `delta_gates`, and with the epsilon at 1e-6 must be red on
    `delta_gnorm`.
- **O-2, whole model, tiny (card).** `olmo_hybrid-tiny`, CUDA resident against the CPU at the same quant (int8int8), 128 tokens, at the
  bar the other DeltaNet families hold in `TestQwen35ResidentParityCUDA`: worst cosine at least 0.95, drift (token 1 minus worst) at
  most 0.02, and the replay after `Reset` at a self-cosine of at least 0.9999999. The build must be resident (non-vacuity).
  - **Plants:** each setting dropped at the build. The NegEigval plant must be red at this bar.
  - **The ONormEps plant is expected NOT to be red here.** On a random-weight fixture the norm's mean square is orders above 1e-5. Its
    result is logged either way, and O-1 carries that setting.
- **O-3, whole model, real (card).** `~/models/olmo-hybrid-7b` (local NVMe, not the archive) at int4 on both sides, at O-2's bar.
  - **Fit check first:** the server's own plan at int4 and `--kv i8`, read from its banner. A refusal makes O-3 NOT RUN, named, and the
    decline is reported. It is not a pass.
  - **Plants:** both are run and logged. Neither is required red, because a real checkpoint's sensitivity is what this run measures.
- **O-4, the family's other CUDA tests.**
  - `TestCUDADeltaNetPrefill_matchesPerToken` must PASS on `olmo_hybrid-tiny`; it skips that fixture today.
  - The Olmo smoke tests must pass.
  - `olmo_hybrid-tiny` must load resident with `--kv i8` and with `--kv f16`.
- **O-5, U1.** `gate identity -backend cuda <main base> <head>`. Every family that ran stays IDENTICAL, with `qwen3_5`, `qwen3_5_moe` and
  `qwen3_next` among them. `olmo_hybrid`'s cells may change, because their backend changes: they are listed, and every other cell keeps
  its outcome.
- **O-6, the censuses.** The admission, decision and SB2 censuses are regenerated, and their diffs touch only `olmo_hybrid` lines.
  `walkNamedDeclines` loses `olmo_hybrid-tiny`.
- **O-7, U8:** the push record.

**Void if:**
- a card run outside the timing lock;
- the real checkpoint read from `/srv/models`;
- a NOT RUN counted as a pass;
- a bar moved after a reading.

**Card time, estimated:** O-1 about 1 min, O-2 about 1, O-3 about 7 (one CPU arm, three CUDA loads), O-4 about 1, O-5 about 6, O-6
about 2, O-7 about 3. About 21 minutes, under the 30 that would be flagged.

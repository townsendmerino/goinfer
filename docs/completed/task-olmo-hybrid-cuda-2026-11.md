# Olmo Hybrid on CUDA: NegEigval and ONormEps in CUDA's DeltaNet (2026-11)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.

> Closed 2026-11-04: CUDA's Gated DeltaNet applies Olmo Hybrid's NegEigval and ONormEps and is gated against the CPU (#604, merged). The tree says so in program.CUDA and changelog.d/2026-11-01-olmo-hybrid-cuda-cpu.md. Not owned here: real-checkpoint WebGPU reading (see docs/tasks/task-olmo-hybrid-metal-webgpu-2026-11.md, archived alongside).

**Registered 2026-11-02 06:56 PST, before any code (nobara-sb01).** Francis's direction of 06:31, relayed by the coordinator: implement
`linear_allow_neg_eigval` (NegEigval) and the output norm's own epsilon (ONormEps) in CUDA's Gated-DeltaNet, gate it against the CPU,
then claim both in `program.CUDA`, so Olmo Hybrid runs resident on CUDA again and `--kv i8` loads. This work has its own card budget,
separate from OP4 on CUDA's. *(Corrected in the readings, O-4, and in #604's review: what loads is the CPU-side KV cache at i8
(`KVQuant`, which serve's `--kv` also maps onto). CUDA's resident KV cache (`KVPrecision`) still refuses i8 and f16 for this
family at load, on main as here.)*

## Why

OP4cu-c1 (`docs/tasks/task-forward-architecture-2026-10.md` §4.9a) made `program.CUDA` the CUDA build's admission table. CUDA's DeltaNet
read neither setting, so Olmo Hybrid has declined to the CPU since then. Before c1 it ran resident and computed the family differently from
the CPU:
- **NegEigval:** the CPU doubles the write gate, `beta = 2·sigmoid(b)`, giving the range [0, 2).
- **ONormEps:** the CPU's gated output RMSNorm uses 1e-5, while the release's `rms_norm_eps` is 1e-6.

Its only CUDA tests were a smoke test and a prefill-vs-decode check, and neither compared against the CPU.

## What changes

- **NegEigval.** `delta_gates` and `delta_gates_rows` (`backend/cuda/deltanet.cu`) take a `betaScale` argument, 1 or 2, and multiply the sigmoid by
  it. Multiplying by 1 is exact and multiplying by 2 is exact, so every other DeltaNet family's output is unchanged bit for bit.
- **ONormEps.** The `delta_gnorm` and `delta_gnorm_rows` launches pass the layer's own epsilon instead of the model's `NormEps`.
- **Where the values come from.** Both are read at build from each layer's `OpDeltaNet` parameters in the program, not from the family.
- **The PTX.** `deltanet.ptx` is regenerated with `build_ptx.sh` and the NVRTC it was built with (12.9.86). The unchanged source was
  first rebuilt from that toolchain byte-identical on 2026-11-02.
- **The claims.** `program.CUDA` sets `DeltaNegEigval` and `DeltaONormEps`.
- **Named outcome change, the inverse of c1's:** `olmo_hybrid` goes back to resident on CUDA in the admission and decision censuses and in
  identity's cells. `--kv i8` with it loads. *(As corrected above: the CPU-side `KVQuant` i8 loads resident; the resident
  `KVPrecision` i8 and f16 are refused at load.)*

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

## Readings, 2026-11-02 (nobara-sb01, branch `olmo-cuda-dn-2026-11-02` from main `d9ff47903`; registration `888e86e16`, code `374f1654e`)

Logs are in `docs/measurements/olmo-hybrid-cuda-2026-11/`. Two jobs ran under the shared lock: 07:02:26 to 07:08:37 PST, and 07:08:52 to
07:10:58. The load average was at most 5.6 at a job's start, and `kwin_wayland` was the only other CUDA context.

- **O-1: PASS** (`o1-kernel-chain.log`).
  - At Olmo Hybrid-7B's geometry every stage holds the bars: worst cosine 1.000000000, and maxAbs/rms at most 2.1e-5.
  - The output norm's input mean square is 8.2e-6 (median over steps), within 10x of the epsilon.
  - Both plants are red. With NegEigval dropped, `delta_gates` has cosine 0.944. With ONormEps dropped, `delta_gnorm` has
    maxAbs/rms 22.8.
  - The Qwen geometry is unchanged at `betaScale` 1.
  - **Setup before the recorded run.** A first draft scaled the value rows down to bring the norm's input near the epsilon. It
    overshot, giving a mean square of 7e-10, and the test's own check went red. Unscaled, the input sits at 8e-6, so the scaling was
    removed. That run is not a reading.
- **O-2: PASS** (`o2-o4-tiny.log`).
  - `olmo_hybrid-tiny` at int8int8: worst cosine 0.999527, drift 0.0003, argmax 121/128, replay 1.000000000.
  - With NegEigval dropped: red, 0.889 with drift 0.109.
  - **With ONormEps dropped: red, 0.620 with drift 0.362, against the registered expectation that it would not be.** The tiny
    fixture's norm input is small enough for the epsilon to matter. The registration logged this result either way.
  - The control, `TestQwen35ResidentParityCUDA`, passes: 0.999919 and 0.991248.
- **O-3: PASS** (`o3-real.log`).
  - `~/models/olmo-hybrid-7b` at int4, both sides: worst cosine 0.984266 (token 1 0.988546, drift 0.0043), argmax 106/128,
    replay 1.000000000.
  - Both plants are red, though neither was required. With NegEigval dropped the worst cosine is 0.416 (drift 0.538). With ONormEps
    dropped it is 0.119 (drift 0.624, argmax 36/128).
  - Before c1, CUDA ran this family resident with both settings missing.
  - **Fit check: two slips of mine, corrected before the run.**
    - The registered check passed `--kv i8` and built `./cmd/serve` with `-tags cuda`, which since v0.10.0 builds no backend
      (`o3-fit-check.log`: BUILD FAILED, so O-3 was first NOT RUN).
    - The corrected check builds `./cuda/cmd/serve` and plans the load O-3 makes, at the default KV (`o3-fit-check-corrected.log`).
      Its plan: resident int4, context 9984, KV f32.
    - **Side effect: that serve run transcoded `~/models/olmo-hybrid-7b.int4.e4h.cuda.v17.giw` (4.2 GB, serve's one-time cache)
      onto a disk that is 99% full. Left in place for Francis to keep or remove.**
- **O-4: the registered clause FAILED. The corrected check PASSES.**
  - The registration read c1's "`--kv i8` refused" as the resident KV cache (`KVPrecision`). The census cells c1 changed set
    `KVQuant`, the CPU-side cache that serve's `--kv` also maps onto.
  - As registered, `KVPrecision` f16 and i8 are refused at load (`o2-o4-tiny.log`), on main as here. The reasons:
    - f16 KV on CUDA does not cover `olmo_hybrid`;
    - CUDA has no int8 KV cache.
  - The corrected test (`TestOlmoHybridResident_kvOptions`, `o4-kv-options.log`) finds Olmo resident under the default,
    `KVPrecision` f32, and `KVQuant` i8 and f16. The two `KVPrecision` refusals are logged.
  - `TestCUDADeltaNetPrefill_matchesPerToken` now runs `olmo_hybrid-tiny` instead of skipping it, and passes.
  - Both Olmo smoke tests pass. `TestOlmoHybridResidentSmokeCUDA` requires a resident build, so it has been failing since c1
    wherever the fixtures are placed.
- **O-5: PASS** (`o5-identity-cuda.log`, `o5-outcomes.txt`).
  - `gate identity -backend cuda d9ff47903 374f1654e`: 28 IDENTICAL, 1 DIFFERENT and 14 NOT RUN.
  - The DIFFERENT family is `olmo_hybrid`, whose two cells move from the CPU to CUDA resident, as named. 55 cells are resident,
    against 53.
  - `qwen3_5`, `qwen3_5_moe` and `qwen3_next` are IDENTICAL.
- **O-6: PASS** (`o6-censuses.log`, `o6-vs-before-c1.txt`).
  - 21 golden lines change, all `olmo_hybrid`'s.
  - Its 9 admission lines and 12 memory lines are byte-identical to the goldens before c1's census commit (`f41a28c42^`).
- **O-7:** the push record follows.
- **A slip of mine:** one `go vet` ran outside the lock during my own shared job (no timed run was on the box).
- **Card time:** about 10 minutes, under the estimated 21: O-1's setup about 1, the first job about 6, the second about 2.
- **Card time in all (#604's review, point (d)):** about 19 minutes against the 21 registered. That is about 10 for the gates above,
  about 8 for the two push records (07:11:54 to 07:15:53 and 07:17 to 07:21:43 PST, the first red), and under a minute for the
  clean-worktree `TestLoadedSublayers` checks, the review follow-up tests (`o8-review-followups.log`) and the CI-fix runs.

## #604's review (Gemini, relayed by the coordinator 2026-11-02), folded in

- `TestOlmoHybridResidentParityCUDA_tiny` now requires the ONormEps plant red as well. It read 0.620, with drift 0.362, in O-2.
- `TestOlmoHybridResident_kvOptions` requires the resident f16 and i8 KV caches to be refused at load, with the KV cache's own
  message, where it only logged them before.
- `dnSettingPlantForTest` is reset by `t.Cleanup` as well as after each build.
- The registration's two `--kv i8` lines are annotated above.
- The logs before the tests moved into `qwen35_resident_parity_test.go` name `olmo_hybrid_parity_test.go`. The tests are the same.

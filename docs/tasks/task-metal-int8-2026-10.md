# Metal runs int8 weights natively (W8A8) — 2026-10

**Status: slice 1 built, native path off by default until F3 and S pass (started 2026-10-01, owner: "lets start it").** Gates below were written and committed
before any implementation or timed run.

## Why

- **G10** (`docs/tasks/task-gpu-paths-2026-09.md`): every int8-kind weight is re-quantized to int4 (W4A8) when the Metal
  resident builds (`metal/model.go`, `int4BufA`). `--quant int8`, `int8int8` and `int4mix` on Metal therefore run int4
  numerics, and f32 weights decline to the CPU. G10 offered "a W8A8 GEMV on Metal" or an honest label, sized the
  kernel "medium", and only the label was done.
- **R17** (`docs/tasks/task-first-hour.md`): because of that, `-backend auto` keeps every non-int4 model on the CPU
  (`autoMetalPrecision`, `decoder/residency.go`). The model-included `goinfer-chat` files are int8int8 bundles, so on a
  Mac they never use the GPU.
- **What already exists.** Metal has W8A8 kernels: `gemv_w8a8_coal` and `gemv_w8a8_amax` (`metal/kernels.go`), and
  `mc3_lm` for the batched step (`metal/batch.go`). They serve only the LM head, which runs at 161–163 GB/s against a
  178–182 GB/s read ceiling (`docs/measurements/metal-decode-gemv-s0-2026-09-26.md`). The runner began as W8A8 and moved
  to W4A8 as the target quant (`docs/completed/task-metal-cgofree-spike.md`). Native int8 was never tried and rejected.
- **Why it can be exact.** W8A8 accumulates in int32, so the sum is exact and any kernel shape gives the same bits. The
  activations are already int8 on Metal (one f32 scale per vector, the `quant_vec` / `rmsnorm_quant` family) and the CPU
  quantizes the same way (max/127, multiply by the reciprocal, round half away from zero). Both rescale as
  `float(acc)·aScale·wScale`.

## Slices

1. **Dense int8 decode** — this doc's first deliverable.
   - **Kernels**: W8A8 GEMVs in the int4 kernels' argument order, with the four epilogues decode dispatches: plain,
     +bias (fused QKV), +residual (o-proj, down-proj) and bias+residual (FeatOutBias). One simdgroup per row first.
     Faster shapes (staged activations, several rows per simdgroup) are bit-identical by construction and can follow.
   - **Build**: int8 upload, an exact int8 concat for fused QKV and gate|up (per-row scales concatenate exactly), and
     the `.giw` alias where a bundle allows it (`int8BufA`).
   - **Dispatch**: a model-level flag, set when every dense body projection is int8-kind, either int8int8 or
     weight-only int8 (CUDA and WebGPU also run that one with int8 activations). Each GEMV site branches on it.
   - **Declines**: the int4-only paths must not read int8 bytes. These are the f16 decode lane, fast prefill (a w8
     model prefills token by token, slower to first token), MC3 batched decode, and MoE / Gemma-4 MoE / DeltaNet /
     qGate layers. A model with any of those keeps today's int4 re-quantization.
   - **Reporting**: `DecodePath` names `metal-resident (int8int8)`, not the re-quant label. The resident reports the
     precision it built, since the decoder cannot see which path Metal took.
   - **Unchanged**: `autoMetalPrecision`. Auto keeps int8 models on the CPU until slice 2 lands and gate S passes.
2. **int8 prefill GEMM**, so first-token latency matches; aikit's ViT W8A8 GEMMs are a template.
3. **MC3 batched decode and the verify step** in int8, from `mc3_lm`.
4. **MoE int8 GEMVs**, then int4mix (mixed kinds in one layer), then f32 by quantizing to int8 at build, as CUDA and
   WebGPU do.
5. **Let auto pick Metal for int8 models**: `autoMetalPrecision` admits them once slices 1–2 and gate S pass.

Out of scope: `q4k`. It needs per-32 activation quantizers, a Q4_K GEMV and a prefill answer, and has no `.giw` form.

## Gates (pre-registered 2026-10-01)

**F1 — kernel exactness (by day).** Each W8A8 variant equals a CPU reference `float32(Σ int8·int8)·aScale·wScale`
(+bias, +residual, in that order) bit-for-bit on random int8 inputs, at the decode shapes of the 0.5B and 1.5B coder
(K = 896, 1536, 4864, 8960). If the build's fast math reorders the epilogue, the bar becomes at most 1 ulp per output,
with the cause recorded. It never becomes a looser tolerance.

**F2 — the same as the CPU at the same quant (by day, 0.5B; the 1.5B with gate S).** In the `TestDenseResidentParity`
harness (24 greedy steps, logit cosine and argmax per step), compare Metal int8int8 against CPU int8int8, and Metal int4
against CPU int4 in the same run. Bar: the int8 pair's minimum cosine is at least the int4 pair's, it is at least 0.99,
and its argmax agrees with the CPU at no fewer steps than the int4 pair.

**F3 — closer to f32 (by day, same scope).** Against the CPU f32 forward on the same steps, the mean per-step
KL(f32 ‖ Metal int8int8) is at most 1.10 × KL(f32 ‖ CPU int8int8). KL(f32 ‖ Metal int4) is reported beside it; that gap
is the point of the work.

**S — speed (night queue, Mac, harness defaults).** An in-process interleaved A/B of decode on the 0.5B and 1.5B at
depths 128 and 2048, at least 7 alternating repetitions. The arms are Metal int8int8 native, Metal re-quant (today's
`-backend metal` with an int8 model), and CPU int8int8 (what `-backend auto` gives an int8 model today). The 7B joins
only if the fit guard admits it; this Mac has 16 GB, and no guard is bypassed.
- **S-auto** decides slice 5 and is read after slice 2. Ship if Metal native decodes at ≥ 1.20× CPU int8int8 at both
  depths on both models, and first-token latency at a 1,000-token prompt is no worse than the CPU's. Park at 1.00–1.20×
  (auto keeps the CPU). Kill below 1.00×.
- **S-explicit** is reported, not gated: Metal native against Metal re-quant. The estimate is about 0.93× on the 1.5B
  and 0.72× on the 7B, from int8 carrying 1.78× the bytes at the LM head's measured rate; it is reasoning, not a
  measurement.

**Owner decision O1, decided 2026-10-01: native.** An explicit `-backend metal` with an int8 model has always run int4.
Once F1–F3 pass it runs int8 natively, the precision asked for, as CUDA does, with the speed disclosed in the CHANGELOG
(gate S reports it). The int4 re-quant stays only for the models slice 1 does not cover.

## Tests to update

- `TestResidentQuantLabel` (`decoder/staged_device_note_test.go`) and the R17 auto tests, for the new label.
- `TestMetalSnapshotGolden` (mixtral-tiny at int8int8) is MoE, so slice 1 leaves it on the re-quant path. Slice 4
  changes it, and its re-bake must record why.
- 37 Metal test files (57 call sites) load models with `Quant: "int8int8"`. Today they test the int4 re-quant path;
  under slice 1 the dense ones move to the native path, and each failure is read before any bar moves.

## Log

- 2026-10-01: plan and gates committed; slice 1 started.
- 2026-10-01: owner decision O1: native. F1 passed: 7 kernels × 8 shapes, 249,984 of 249,984 outputs bit-identical to
  the CPU (`TestGemvW8A8Body_matchesCPU`). Its first run failed the residual kernels by many ulps where the residual
  cancels the sum: fast math fused the product into the add. The kernels now round the product first, as the CPU does.
- 2026-10-01: slice 1 built and committed with the native path **off by default** (`nativeInt8` in `metal/model.go`, set
  only by the gate tests), so nothing changes for a user until F3 and S pass; then it becomes the default.
  - **F2 passed on the 0.5B** (`TestW8Native_F2_matchesCPUAtSameQuant`): Metal int8int8 against CPU int8int8, min cosine
    0.994500 and argmax 24/24; Metal int4 against CPU int4, 0.988340 and 23/24. Log:
    `~/goinfer-logs/metal-int8/f2-f3-0.5b-2026-10-01.log`.
  - **F3 deferred to the night queue.** The fit guard refused its f32 reference by day: the 0.5B at f32 priced 3.6 GB
    against a 2.2 GB budget, and still about 2.8 GB with the test's context pinned at 1024. A bypass is not used on
    this Mac. Gate S's harness ran once at smoke size by day to check it works; those numbers are not a result.
  - Queued for tonight on the Mac: `docs/measurements/metal-int8-2026-10/run-gates.sh` (F3 on the 0.5B and 1.5B, F2
    on the 1.5B, S), running a test binary built at `f73b980a`. Before the commit: the default Metal suite passed with
    the switch off (157.7 s).

# Metal runs int8 weights natively (W8A8) — 2026-10

**Status: slice 1 built; F3 (0.5B) and F2 (1.5B) failed on 2026-10-02, so the native path stays off by default. The per-layer comparison (2026-10-04) found no defective op: F3's gap is the f16 KV cache, fast math, and the noise any non-identical path adds at this quant. An amended bar is proposed and waits on the owner (Log, 2026-10-04) (started 2026-10-01, owner: "lets start it").** Gates below were written and committed
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

**Owner decisions, 2026-10-04** (after the per-layer comparison, Log 2026-10-04): "1, yes, 2, precise".
- **F3 amended (O2), as F3′ below.** F3 as written compared a backend with an f16 KV cache against a CPU with an f32
  one, which only bit-identity can pass at this quant.
- **Precise math for native int8 (O3).** A model headed for the native int8 path compiles its Metal library without
  fast math (`w8PreciseMath`). Fast math was about 40% of F3's gap. Its decode cost is reported by gate S's new
  fast-math arm, not gated.

**F3′ — closer to f32, against the CPU at Metal's KV precision (pre-registered 2026-10-04, before any F3′ run).**
`TestW8Native_F3amended_closerToF32`, by day on the 0.5B, on the night queue on the 1.5B if the fit guard admits its
f32 reference (it refused it on 2026-10-01: 7.7 GB against 7.6).
- **Prompts:** the first 8 files of the prefill gate's prose set A, the first 16 tokens of each, then 16 tokens of the
  CPU f32 model's greedy continuation. Positions 2-31 are scored, 240 in all.
- **Arms:**
  - the reference: CPU int8int8 with every K and V rounded to f16 as it is stored;
  - Metal native int8int8, precise math, the shipped configuration if it passes.
- **Bar:** the pooled mean KL(f32 ‖ Metal int8int8) is at most 1.10 × the pooled mean KL(f32 ‖ reference). The test
  fails otherwise.
- **Reported, not gated:**
  - positions where Metal is further from f32 (expected to be most of them: any non-identical path adds noise);
  - per-prompt ratios;
  - CPU int8int8 at f32 KV, Metal int8 with fast math, and Metal int4.

**F2** keeps its bar and re-runs on the 1.5B with precise math. **S** gains the fast-math arm. **S-auto** still waits
for slice 2. The native path turns on by default only when F3′ (0.5B and, if it runs, 1.5B), F2 on the 1.5B and S all
pass.


**Slice 2 — int8 prefill (built 2026-10-04; its gate P2 pre-registered the same day, before any graded run).**

*What is built.*
- `gemm_w4f16_tile` takes a third template parameter, `W8`. With it set, the weight staging reads the native int8
  path's row-major int8 codes and per-row f32 scales, and dequantizes each weight to f16 as `half(q·scale)`. The int4
  instances are unchanged: `TestGemmTile_bitIdentical` passes over 98,102,016 outputs.
- `gemm_w8f16_m{64,32,16}n{64,32}` are the W8 tiles; `gemmTile` takes them for a native int8 resident, with the same
  tile rule as int4. `prefillOK` no longer excludes `r.w8`.
- The prefill library compiles with precise math when the resident does (O3).
- The W8F16 form follows the hardware: M1-class GPUs have no int8 simdgroup MMA. The int4 pass's W4F16 shape is used
  for the same reason.

*Kernel check* (`TestGemmW8Tile_matchesReference`, default-run): every tile against a float64 reference over the same
f16 inputs and f16-dequantized weights.
- The bound is one f16 rounding plus f32 accumulation over K products.
- 7,315,200 outputs pass across six shapes of the 0.5B and 1.5B, 8-72 rows and three epilogues. Every tile equals the
  64×64 one bit for bit.
- Swapping the two staged words of an 8-k block fails it (an output off by 5.7 against a 0.003 bound).

*Exploratory, by day* (`TestW8Prefill_passAgainstSequential`: 6 set-A prompts of 256 tokens, the pass's last logits
against the sequential loop's, after a warm pass):

| model, arm | pass | sequential | KL(seq ‖ pass) | argmax |
|---|---|---|---|---|
| 0.5B native int8, precise | 103.2 ms | 1626 ms (15.8×) | 0.0369 | 6/6 |
| 0.5B native int8, fast math | 101.0 ms | — | 0.0349 | 5/6 |
| 0.5B int4 (shipped pass) | 102.8 ms | 1415 ms | 0.0433 | 5/6 |
| 1.5B native int8, precise | 264.9 ms | 3692 ms (13.9×) | 0.0224 | 6/6 |
| 1.5B int4 (shipped pass) | 263.0 ms | 2992 ms | 0.0165 | 5/6 |

- The pass is compute-bound, so int8's extra bytes and precise math cost nothing there.
- One exploratory cell of the gate below (1.5B, K = 256, not a decision) read "SHIPS": fast mean KL 0.0043 against
  exact's 0.0063, with 1 and 0 hard flips.

*Gate P2 — int8 prefill fidelity (night queue).* The §3.2 pooled gate (`TestPrefillGateVsReference`) on the 1.5B,
prompt set A, decision set K ∈ {256, 512, 1024}, confirmation cells off (`GOINFER_METAL_GATE_QUANT=int8int8`).
- Both arms run on the native int8 path with precise math: fast is the batched pass, exact is the sequential loop.
  Both are scored against the CPU f32 references already on disk (set A, `S-K*`).
- The rule is the one int4's pass shipped under, unchanged: critA, critB and critC pooled over the decision set.
  Native int8's pass ships iff all three hold.
- The 7B (`D7`) is left out: the fit guard does not admit it at int8int8 on this Mac.
- **Not** decided here: S-auto, which also needs first-token latency at a 1,000-token prompt, read from gate S's
  harness after this.


**S-auto's first-token half, made measurable (written 2026-10-04, before any graded run).** S-auto's second condition
is "first-token latency at a 1,000-token prompt is no worse than the CPU's". Before slice 2 nothing measured it, and
native int8 prefilled one token at a time. `TestW8Native_S_ttft` (night, `GOINFER_W8_GATE_TTFT=1`):
- The 0.5B and the 1.5B; a 1,000-token prompt (`w8SpeedPrompt`); arms Metal native int8 (precise math), CPU
  int8int8, and Metal int4 re-quant (reported). Arms rotate every repetition, 7 repetitions, and each sample loads its
  arm untimed.
- **Cold** is the first request after the load. **Warm** is a second request whose prompt shares no prefix with the
  first (the same tokens reversed).
- **Amendment, written before the run:** S-auto's latency condition is read on **warm**, the serving steady state.
  Cold, which includes Metal's one-time prefill-library compile, is reported beside it.
- The condition holds when the median native ÷ CPU ratio is at most 1.0 on both models.
- A one-sample smoke on the 0.5B checked the harness by day (labelled exploratory, not a result): warm 470 ms against
  1483 ms, cold 543 ms against 1501 ms.


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
- 2026-10-02: **the night gates (2026-10-01, 01:14–01:25 PDT, binary `metal-f73b980a.test`): F3 and F2 fail, so the
  default does not flip.** Logs: `docs/measurements/metal-int8-2026-10/night-2026-10-01/`.
  - **F3, 0.5B: FAIL.** Mean KL(f32 ‖ ·) over 22 positions: CPU int8int8 0.021989, Metal int8int8 0.027917 (1.270× the
    CPU's; the bar is 1.10×), Metal int4 0.183825. Native int8 is 6.6× closer to f32 than today's re-quant, but not as
    close as the CPU at the same quant.
  - **F3, 1.5B: did not run.** The fit guard refused the f32 reference: 7.7 GB priced against 7.6 GB (70% of 10.9 GB
    available). Not bypassed.
  - **F2, 1.5B: FAIL on argmax.** Metal int8int8 against CPU int8int8: min cosine 0.997700, argmax 21/24, worst near-tie
    0.325%, no gap over 3%. Metal int4 against CPU int4: 0.997514, 24/24. The cosine bars pass; the argmax bar (at
    least the int4 pair's count) does not.
  - **S** (decode, in-process interleaved, 7 reps, medians; reported, and S-auto is read only after slice 2):

    | model | depth | native | re-quant | CPU int8int8 | native ÷ re-quant | native ÷ CPU |
    |---|---|---|---|---|---|---|
    | 0.5B | 128 | 165.7 | 184.9 | 100.0 | 0.896 | 1.657 |
    | 0.5B | 2048 | 102.1 | 106.4 | 65.4 | 0.960 | 1.561 |
    | 1.5B | 128 | 71.0 | 89.5 | 45.4 | 0.794 | 1.565 |
    | 1.5B | 2048 | 65.8 | 82.1 | 32.6 | 0.801 | 2.020 |

    The 1.5B's native ÷ re-quant is 0.79–0.80, under the 0.93 the doc estimated from the LM head's rate.
  - What it means: F1 held every kernel bit-identical to the CPU, so the gap from CPU int8int8 comes from outside the
    W8A8 GEMVs. The next step is a by-day per-layer comparison of Metal int8int8 against CPU int8int8 on the 0.5B
    (no f32 reference needed), to name the first layer and op that diverges.
- 2026-10-04: **the per-layer comparison, and where F3's gap comes from** (by day, the 0.5B; the fit guard admitted the
  f32 reference this time, so F3 ran by day and reproduced the night's numbers exactly: CPU int8int8 0.021989, Metal
  int8int8 0.027917).
  - **The per-layer comparison** (`TestW8Native_perLayerBisect`; residual stream relative L2 after each layer at the
    last position of a 32-token prose prompt):
    - Metal int8 against CPU int8 is 3.3% after layer 0 and 8.8% after the last layer.
    - Rounding only the CPU's own K and V to f16 moves the CPU's stream by 3.2% after layer 0 and 14.1% after the last.
      The int8 activation quantization turns any one-ulp difference into rounding crossings, layer after layer.
    - Metal int8 sits inside that perturbation at every layer from 9 on, and the int4 control pair (which passed F2)
      shows the same profile, 11.3% at the end.
    - **No layer and no op stands out, so per-layer distance cannot name a defect here.** F1 already holds every W8A8
      GEMV bit-identical given the same inputs.
  - **The KL split** (F3's sequence, reported arms added to `TestW8Native_F3_closerToF32`):

    | arm | KL(f32 ‖ ·), 22 positions | further from f32 than the f16-KV CPU |
    |---|---|---|
    | CPU int8int8 (F3's reference, f32 KV) | 0.021989 | — |
    | CPU int8int8, K/V rounded to f16 (Metal's KV precision) | 0.024045 (1.093× the above) | — |
    | Metal int8int8, precise math (`GOINFER_W8_F3_PRECISE=1`) | 0.025440 (1.058×) | 20 of 22 |
    | Metal int8int8, shipped fast math | 0.027917 (1.161×) | 19 of 22 |

    - Of the 1.270× gap, the f16 KV cache is about a third, fast math about 40%, and the remaining quarter is what
      precise math still leaves.
    - Any path that is not bit-identical to the CPU lands consistently further from f32 at a quant this sensitive,
      because independent noise adds. The CPU against itself with f16 KV moves 1.093×, already near F3's 1.10 bar.
  - **What that means for the gate.** F3 as written is in effect a bit-identity bar against the CPU, which no
    f16-KV backend can meet. It is not a quality bar.
  - **Proposed for the owner, not applied** (pre-registered bars change only by owner decision): measure Metal int8
    against the CPU at Metal's own KV precision, over more than one prompt. Two choices:
    - (a) keep fast math and take the extra 1.16×;
    - (b) compile the library with precise math for int8 models, at the decode cost precise math was measured at on
      2026-08-04 (about 4% at depth 2048, 7% shallow), and re-measure for int8 before deciding.

    Nothing is regraded and the native path stays off.
- 2026-10-04: **F3′ on the 0.5B: PASSES** (by day, `278bd1f2`, the test committed with its bar before the run).
  - Pooled mean KL(f32 ‖ ·) over 8 prompts × 30 positions: CPU int8int8 with f16 KV (the reference) 0.098462, Metal
    native int8int8 with precise math 0.066667, **0.677×** against the 1.10 bar.
  - Per prompt the ratio is 0.500 0.559 0.643 0.788 0.638 0.963 0.674 0.616; Metal is further from f32 at 51 of 240
    positions.
  - Reported: CPU int8int8 at f32 KV 0.086232, Metal int8 with fast math 0.062854 (0.638×), Metal int4 0.660900.
  - **Is Metal closer than the CPU because of a mismatch?** A by-day check says no.
    - `TestW8Native_hiddenVsLogits`: the same 8 prompts teacher-forced over 32 prose tokens, 240 positions.
    - The final residual stream is as far from f32 on both sides: relative L2 0.1337 Metal, 0.1379 CPU.
    - Logit KL is 0.12205 against 0.13171, 0.93×.
    - Nothing points to a different quantization in the trunk or the LM head: the two are equally good, with Metal
      slightly closer.
    - The one-prompt F3 (1.27×) and the 8-prompt F3′ disagree in direction, which is why F3′ pools prompts. The 0.677×
      magnitude rests on 8 prompts and a heavy-tailed KL; the direction rests on all 8 prompts and the teacher-forced
      check.
  - **Fast against precise** is within the noise here (0.0629 against 0.0667), while on F3's one prompt precise was
    closer (0.0254 against 0.0279). The precise-math decision stands as the owner made it; gate S prices it tonight.
  - **Queued for tonight** (Mac, `run-gates2.sh`, binary pinned at the commit that records this): F3′ on the 1.5B if
    the fit guard admits its f32 reference, F2 on the 1.5B with precise math (bar unchanged), and S with its fast-math
    arm.

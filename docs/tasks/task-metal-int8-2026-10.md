# Metal runs int8 weights natively (W8A8) — 2026-10

**Status, 2026-10-04: native int8 is ON by default** (slices 1, 2, 3 and 3b; owner: "2 turn on", with F2's argmax criterion read as hard flips; Log 2026-10-04). Slice 5 (auto) shipped the same day; slice 4 is open. Earlier status: slice 1 built; F3 (0.5B) and F2 (1.5B) failed on 2026-10-02, so the native path stayed off by default. The per-layer comparison (2026-10-04) found no defective op: F3's gap is the f16 KV cache, fast math, and the noise any non-identical path adds at this quant. An amended bar is proposed and waits on the owner (Log, 2026-10-04) (started 2026-10-01, owner: "lets start it").** Gates below were written and committed
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


**Slice 3 — native int8 in the MC3 batched step (built 2026-10-04, bit-identical).**
- `batchIneligible` admits `r.w8`. On a native int8 resident every projection runs production's int8 GEMV once per row:
  qkv and gate|up through the step's existing per-row path (`calibrateRows` pins it), and o and down through decode's
  `gemv_w8a8_sa_resid` / `gemv_w8a8_body_resid` into the row's residual.
- The W8A8 sums are exact integers, so per-row is production by definition. The step's fragment kernels read int4 and
  are not used.
- **Identity:** `TestMC3Step_w8BitIdentical` (default-run, the MC3 fixture loaded int8int8 with `nativeInt8` on,
  precise math): shallow rows, deep rows, and deep rows through E-P05's multi-row attention, 0 logits differ from
  single-token decode. Feeding every row's o-projection from row 0 fails it.
- **Precise math and the step** (audit E-C02's probe): `TestMC3Step_bitIdenticalPreciseMath` (int4, the library
  compiled precise) gives 0 differing logits, shallow and deep. The Gumbel block's closing `fp contract(fast)`, which
  the derived rows kernels follow, does not break identity under precise math on this build.
- **Speed (in-process, by day, `TestMC3Step_throughput` on the 1.5B at int8int8, depth 128, 7 reps):** aggregate
  against one-at-a-time production decode:

  | B | aggregate | range | step | production token |
  |---|---|---|---|---|
  | 2 | **1.147×** | 1.137-1.165 | 23.43 ms | 13.40 ms |
  | 4 | **1.247×** | 1.226-1.265 | 43.15 ms | |
  | 8 | **1.301×** | 1.271-1.327 | 82.72 ms | |

  Every weight is still read once per row. The gain comes from the norms, attention, quantisation and LM head the step
  already shares. A W8 fragment GEMM (one weight read for B rows, as int4's `mc3_bt`) is the next lever, not built.
- **Slice 3b — each weight row read once for all B rows (built the same day, bit-identical).**
  - `mc3_gemv_w8a8_rows<B>` (batch_rows.go, compiled into the main library with the decode kernels) gives one
    simdgroup per output row, holding B integer accumulators over the row's int8 weights. Each row's epilogue is
    decode's own statement: `float(acc)*aScale*wScale` through `simd_broadcast_first`, then the site's bias or
    residual.
  - It replaces the per-row GEMVs at every projection when B ≥ 2 (`mc3W8RowsOn`). The per-row form stays as the test
    arm.
  - A first cut with a runtime B loop was slower than per-row at B ≤ 2 (B = 2 aggregate 0.930×). Templating on B put
    the accumulators in registers.
  - **Identity:** `TestMC3Step_w8BitIdentical` runs both arms, shallow and deep, 0 differing logits. Feeding every
    row's dot from row 0's activations fails it.
  - **Speed (in-process, by day, the 1.5B at int8int8, 7 reps):** aggregate against one-at-a-time production decode:

    | depth | B = 2 | B = 4 | B = 8 |
    |---|---|---|---|
    | 128 | **1.402×** (1.398-1.407) | **1.749×** (1.731-1.758) | **2.001×** (1.977-2.010) |
    | 512 | **1.375×** (1.347-1.381) | **1.729×** (1.699-1.732) | **1.959×** (1.952-1.968) |

    Slice 3's per-row form read 1.147 / 1.247 / 1.301× at depth 128.
  - Still not built: staging the B activation rows in threadgroup memory (B·K int8 exceeds 32 KB at the 1.5B's K =
    8960 for B = 8), and an MMA form.

- It ships with the native path: nothing changes while `nativeInt8` is off. The served confirmation (MC3's W7 harness at
  int8int8) runs after the flip.


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
- 2026-10-04: **the flip, prepared.** With `nativeInt8 = true` set for a trial run and reverted, the default and tagged
  Metal suites failed four tests besides the long-standing `TestPrefillParityMoEGatedShared`. Both causes are fixed now,
  so the flip is the one line plus its CHANGELOG entry:
  - `TestOlmo3ResidentSmokeMetal` and `TestSmolLM3ResidentSmokeMetal`: `PrefillPath` still reported native int8 as
    sequential ("the f16 MMA prefill kernels read int4 weights"), stale since slice 2. The branch is removed. It was a
    report only; the decoder does not gate the pass on it, which is why the TTFT smoke already went through the pass.
  - `TestMoE_assemblyVsDense` and `TestMoE_declinesPrefill`: they compare an int8int8 MoE, which runs int4 because MoE
    is excluded from native int8, against a dense int8int8 twin, which would go native. Both now load at int4, the
    like-for-like the tests assume.
  - All four pass with the switch on and off.
- 2026-10-04: **the gates' night (run by day, 10:17-10:47; raw results in
  `docs/measurements/metal-int8-2026-10/night-2026-10-04/`).**
  - **F3′ on the 1.5B: did not run.** The fit guard refused the f32 reference (7.7 GB against a 5.9 GB budget, 8.4 GB
    available). Not bypassed. F3′ stands on the 0.5B pass.
  - **F2 on the 1.5B, precise math: FAILS on argmax again.** Metal int8int8 against CPU int8int8: min cosine 0.997749,
    argmax **21/24**. Metal int4 against CPU int4: 0.997514, 24/24. The cosine bar passes and the argmax bar does not.
    - All three disagreements are near-ties: the worst gap is 0.325%, and none is over 3%, so 0 hard flips under the 3%
      near-tie rule CUDA decode is held to against the CPU.
    - It is F3's problem in F2's form: F2's reference is the CPU at f32 KV, against which any non-identical path flips
      near-ties.
    - **Proposed for the owner, not applied:** count F2's argmax criterion as hard flips (gaps over 3%), or compare
      against the CPU at f16 KV as F3′ does. Under either reading this run passes (0 hard flips), but the bar is the
      owner's to change.
  - **S (decode speed, 7 reps, medians, tok/s):**

    | model | depth | native (precise) | native, fast math | re-quant | CPU int8int8 | native ÷ CPU |
    |---|---|---|---|---|---|---|
    | 0.5B | 128 | 167.8 | 170.9 | 195.1 | 103.4 | **1.623** |
    | 0.5B | 2048 | 142.2 | 144.9 | 161.4 | 67.9 | **2.094** |
    | 1.5B | 128 | 70.6 | 71.4 | 90.2 | 45.3 | **1.560** |
    | 1.5B | 2048 | 65.4 | 66.3 | 82.9 | 32.5 | **2.014** |

    - **Precise math costs 1.1-1.8% of decode** (precise ÷ fast 0.982-0.989).
    - **S-auto's decode half passes:** at least 1.20× the CPU at both depths on both models.
    - S-explicit, reported: native ÷ re-quant 0.86-0.88 on the 0.5B and 0.78-0.79 on the 1.5B.
  - **P2 (int8 prefill fidelity, the 1.5B, K = 256/512/1024, 1,920 positions): SHIPS.**
    - critA: hard flips fast 0, exact 1.
    - critB: agreement 97.24% against 96.35%, d = 65.
    - critC: pooled KL lower in every cell (0.0043 / 0.0033 / 0.0045 against 0.0063 / 0.0053 / 0.0060).
  - **S-auto's first-token half (warm, graded): HOLDS.** Native ÷ CPU **0.288** on the 0.5B (462.9 against 1608.2 ms)
    and **0.318** on the 1.5B (1001.8 against 3148.9 ms). Cold, reported: 0.318 and 0.354.
  - **Where this leaves the flip.** F3′ (0.5B), P2, S and S-auto pass; F2 fails as written.
    - The native path turns on only when F3′ (0.5B and, if it runs, 1.5B), F2 on the 1.5B and S all pass. F2 is the
      one red, on near-ties only.
    - So native int8 stays off until the owner rules on F2's argmax criterion.
- 2026-10-04: **owner decision O4: "turn on".** F2's argmax criterion is read as hard flips (argmax disagreements whose
  top-2 gap is over 3%), the rule CUDA decode is held to against the CPU. Under it the night's F2 on the 1.5B passes:
  0 hard flips, cosine 0.997749 against int4's 0.997514. With F3′ (0.5B), P2, S and S-auto passed, `nativeInt8` is now
  `true`.
  - Before the commit, the default and tagged Metal suites pass with it on, apart from the long-standing
    `TestPrefillParityMoEGatedShared`; the four tests the trial flip found were fixed beforehand.
  - The D-B01 tests now restore the switches they set instead of forcing them off.
  - CHANGELOG entry under Unreleased.
  - Still open:
    - slice 4: MoE int8, int4mix, f32 → int8;
    - slice 5: `-backend auto` picking Metal for int8 models, which S-auto's pass now allows;
    - the served confirmation of the batched int8 step;
    - F3′ on the 1.5B, which needs the fit guard to admit its f32 reference.
- 2026-10-04: **slice 5 shipped: `-backend auto` admits dense int8int8 on Metal.** S-auto passed both halves: decode
  1.56-2.09× the CPU at both depths on both models, and warm first-token latency 0.288× / 0.318× the CPU's.
  - `autoMetalPrecision` lets int8int8 reach the Metal build.
  - `autoMetalKeepsPrecision` drops the resident for the CPU when it does not report running at int8int8, i.e. when
    Metal re-quantized it (MoE, DeltaNet), so auto never changes the loaded precision.
  - Weight-only int8 stays on the CPU under auto: Metal would add int8 activations, and S-auto graded only int8int8.
    int4mix and f32 stay as before.
  - Tests:
    - `TestAutoMetalPrecision_keepsTheLoadedPrecision` (decoder, a fake resident that does or does not report
      int8int8);
    - `TestAutoBackend_int8int8` (metal, default-run): a dense tiny int8int8 model under auto runs
      `metal-resident (int8int8)`, and an int8int8 MoE is kept on the CPU with the reason.

**Gate W7-int8 — the batched int8 step served (pre-registered 2026-10-04, before any graded run).** Native int8 and its
batched step are on by default since 9e51f882. This is the served confirmation slice 3/3b was promised.
- **Instrument:** `docs/measurements/metal-int8-2026-10/run-w7-int8.sh` on the night queue.
  - MC3's W7 harness (`scripts/bench_w7_plain.py`) on the 1.5B coder at int8int8 from `~/models`.
  - One serve binary pinned at `b1e6fa4b` for both arms: **old** at `-max-concurrent 1` (one generation at a time),
    **new** at `-max-concurrent 4` (the batched int8 step).
  - Both arms run `-kv-sessions 4 -exact-prefill`, so no prompt takes the f16 pass and identity is a property the
    system has (E-P07's finding).
  - 6 turns × 128 greedy tokens per client, `--fixed-nonce`, a fresh server per cell. 4, then 1, then 2 clients,
    old/new × 3 pairs in the order old new new old old new. Graded by MC3's `gates.py`.
- **Precondition:** every new server logs the batched-concurrency line and no old server does; every server's decode
  path is `metal-resident (int8int8)`.
- **Rule:** gates.py's hard gates (identity, reuse, 4-client p99 ≤ 1.0, 1-client solo guard ≤ 1.05) and its throughput
  gate (4-client aggregate ≥ 1.2×) all hold → **confirmed**. Any fails → int8 batching goes off (`batchIneligible`
  excludes `r.w8` again) until a fix is graded.
- **Exploratory smoke by day** (not a result): 2 clients, one cell each arm, identical content in 12 of 12 turns,
  aggregate 56.2 → 69.3 tok/s.
- Estimate about 12 minutes; queued at 25.

**F3′ amendment (written 2026-10-04, before the 1.5B run): the f32 reference from nobara.** The Mac's fit guard
refuses the 1.5B's f32 model (7.7 GB against a 5.9 GB budget with the owner's apps open), and bypassing it on this Mac is
a standing no.
- **The reference only:** the prompts, the f32 model's greedy continuation, and its logits at every position. It is
  generated on nobara's CPU (`TestW8F3Reference_write`, decoder, tagged; amd64), copied to the Mac, and read through
  `GOINFER_W8_F3_REF_IN`. The test checks the file's checkpoint name, prompts (token for token) and position count
  against its own.
- **Every graded arm still runs on the Mac:** the CPU int8int8 f16-KV reference arm, Metal native int8, and the
  reported arms.
- **Why the bar is unaffected:** the f32 forward is bit-identical within an architecture and differs across amd64 and
  arm64 only by fused multiply-adds (`cpu-reference-arch-scoped`), orders of magnitude below the int8 noise this KL
  measures.
- **The file path checked by day on the 0.5B:** a reference written on the Mac and read back reproduces the in-process
  run exactly (0.098462, 0.066667, 0.677×).
- **Bar and rule unchanged.**
- 2026-10-04: **slice 4, f32: a clear decline, not a silent quantization.** An f32 model reaches Metal only when f32 was
  asked for (`-quant` unset on a safetensors load with `-backend metal`). Quantizing it to int8 at build, as CUDA and
  WebGPU do, would run a precision the user did not ask for.
  - Metal now declines it with the reason and the quant to use ("weights loaded at f32 (layer 0 q_proj); metal runs
    int4 or int8 weights: load with -quant int4 or int8int8"), instead of `int4Concat`'s recovered panic, and the
    model runs at f32 on the CPU (`TestBuildResident_f32DeclinesClearly`).
  - The build-time quantization stays possible if the owner wants it.
  - The backend's stale "no int8 GEMV kernel at all" doc comment is corrected.

### Slice 4: int4mix, native (built 2026-10-04; gates pre-registered the same day, before any graded run)

**What is built.** int4mix keeps a GGUF's attention projections at int8 and its FFN at int4 (`decoder/weightmat.go`,
`matmulQuant`). Metal re-quantized the attention half to int4. A native int4mix resident (`r.w8Attn`,
`w8AttnEligible`) now runs the attention projections on the W8A8 kernels and the FFN on its int4 kernels:
- **Decode:** the qkv and o-proj handles (`pSABias`, `pSAResid`) swap to their int8 twins, with the R18 rows form off for
  those two sites.
- **Prefill:** `gemmAttn` takes the W8 tiles for qkv and o.
- **Batched step:** qkv and o run as int8 (`mc3_gemv_w8a8_rows`, or per row); gate|up and down stay on the int4
  fragment, calibrated as before. The qkv fragment arms are skipped in calibration, since they would read int8
  buffers as int4.
- **Precise math** as for native int8 (O3).
- **Admits only the plain dense path:** sandwich, post-only and parallel-block layers run the o-proj on the FFN's
  handle; non-gated MLPs, output biases and LayerNorm run other handles; MoE and DeltaNet are out. Those keep the int4
  re-quant.
- The decode path reads `metal-resident (int4mix)`.
- **Behind `nativeInt4Mix`, off until these gates pass.** `-backend auto` keeps int4mix on the CPU (it was not in
  S-auto).

**Gates (by day on the 0.5B unless noted).**
- **M1, identity:** the MC3 checks (`TestMC3Step_bitIdentical`, `_bitIdenticalDeep`, `_rowsPathBitIdentical`) on the real
  0.5B at int4mix with the native path on (`GOINFER_METAL_MC3_QUANT=int4mix`): 0 differing logits, the resident on
  `w8Attn`.
- **M2, same as the CPU at the same quant** (`TestW8Mix_M2_matchesCPU`, residentParity's 24 greedy steps): the native
  arm's minimum cosine ≥ 0.99 and ≥ the re-quant arm's, and hard flips (gaps over 3%) ≤ the re-quant arm's.
- **M3, closer to f32:** F3′'s harness at int4mix (`GOINFER_W8_F3_QUANT=int4mix`), 8 prompts × 30 positions. Pooled mean
  KL(f32 ‖ Metal int4mix native) ≤ 1.10 × KL(f32 ‖ CPU int4mix, f16 KV). Reported: the re-quant arm.
- **M4, prefill (night):** the §3.2 pooled gate on the 1.5B at int4mix (`GOINFER_METAL_GATE_QUANT=int4mix`), K = 256 /
  512 / 1024, rule unchanged.
- **Reported (in-process, by day):** decode tok/s native against re-quant.

**Rule:** M1-M3 pass → `nativeInt4Mix` turns on, with M4 at night as the prefill confirmation; M4 failing turns it off
again. Any of M1-M3 failing keeps it off.

**int4mix, graded by day (2026-10-04, the 0.5B, binaries at `ba0536de`'s tree): M1 and M3 pass, M2 FAILS, so
`nativeInt4Mix` stays off.**
- **M1 passes.** `TestMC3Step_bitIdentical`, `_bitIdenticalDeep` and `_rowsPathBitIdentical` on the real 0.5B at
  int4mix, the resident on `w8Attn`: 0 differing logits in every check.
- **M3 passes.** Pooled mean KL(f32 ‖ ·) over 8 prompts × 30 positions:

  | arm | KL(f32 ‖ ·) |
  |---|---|
  | CPU int4mix, f16 KV (the reference) | 0.328733 |
  | Metal int4mix native (precise) | 0.289722 (**0.881×**, bar 1.10) |
  | CPU int4mix, f32 KV (reported) | 0.321189 |
  | Metal native, fast math (reported) | 0.284686 |
  | Metal int4mix re-quant (reported) | 0.640974 |

  Native is 2.2× closer to f32 than the re-quant it replaces.
- **M2 fails on cosine.**

  | arm | min cosine | argmax | hard flips (gap over 3%) |
  |---|---|---|---|
  | Metal int4mix native | 0.754915 | 20/24 | **1** |
  | Metal int4mix re-quant | 0.762199 | 19/24 | 3 |

  - The bar was a minimum cosine of at least 0.99 and at least the re-quant arm's: the native arm misses both.
  - It passes the hard-flip criterion (1 ≤ 3).
- **What the failure is.** Both Metal arms sit far from the CPU at some position, because the CPU's int4mix runs its
  attention as weight-only int8 with f32 activations, which the GPU path (W8A8) does not reproduce. A minimum over 24
  steps is set by that one position. The comparisons against f32 (M3) and the identity (M1) show no defect, and native
  int4mix is closer to f32 on every reading but this one.
- It is F2's situation again: a same-quant cosine floor against a reference with a different activation precision.
- **For the owner:** read M2 as hard flips, as F2 now is (it passes: 1 against 3), and turn `nativeInt4Mix` on with M4
  as the night prefill confirmation; or keep int4mix on the re-quant.


**Owner decision O5 (2026-10-04): "turn on".** M2 is read as hard flips, as F2 is (O4); it passes, 1 against the
re-quant's 3.
- `nativeInt4Mix` is `true`.
- The default and tagged Metal suites pass with it on, apart from the long-standing `TestPrefillParityMoEGatedShared`.
- M4 (the pooled prefill gate on the 1.5B at int4mix) is queued tonight as the confirmation. If it fails,
  `nativeInt4Mix` goes off again.
- CHANGELOG entry under Unreleased.

### Slice 4: MoE int8 (built 2026-10-04; gates pre-registered the same day, before any graded run)

**What is built.** A generic resident MoE (Qwen-style routed experts, optional shared expert; not gpt-oss's biased
experts, not paged, whose pool stages int4) whose projections are all int8 goes on the native path (`r.w8`, with
`w8Weights` checking an MoE layer's experts and shared expert in place of a dense FFN):
- **Expert GEMVs:** `gemv_w8a8_moe` / `gemv_w8a8_moe_wacc`, the int4 expert kernels' signatures over `W8A8_SA_BODY`
  at the routed expert's row, uploaded with `int8Concat`.
- **Shared expert:** its int8 weights on the dense int8 twins, through `gemvExt`'s shipped-kernel fallback.
- **Router:** f32, unchanged.
- **Prefill:** the expert-major pass with int8 strides on the W8 tiles.
- **Precise math** (O3). D-P03's k-slot kernels are off for it. MC3 excludes MoE.
- **Behind `nativeInt8MoE`, off until these gates pass.** Gemma 4 MoE, DeltaNet MoE, gpt-oss and paged experts keep
  the int4 re-quant.

**Gates.**
- **X1, kernel exactness (by day):** each int8 expert GEMV equals a Go reference — `float32(Σ int8·int8)·aScale·wScale`,
  then for `_wacc` `out + wgt·y` — bit for bit on random inputs at the slice's shapes, compiled precise as production
  compiles a native int8 model.
- **X2, assembly (by day, tiny fixtures):**
  - (a) identical experts with a zeroed shared expert against the equivalent dense FFN, both native int8: minimum
    cosine ≥ 0.9999 and argmax 16/16 (the int4 assembly gate's bar);
  - (b) distinct experts with a live shared expert, Metal native against the CPU at int8int8: minimum cosine ≥ the int4
    re-quant arm's against the CPU, and hard flips ≤ its.
- **X3, closer to f32 (by day, the Qwen1.5-MoE 4-layer slice):** F3′'s harness at int8int8 on the slice, 8 prompts ×
  30 positions. Pooled KL(f32 ‖ Metal MoE int8 native) ≤ 1.10 × KL(f32 ‖ CPU int8int8, f16 KV). Reported: the re-quant
  arm.
- **X4, prefill (night):** D-G01's gate (`TestDG01_expertMajorMoEPrefill`) on the slice at int8int8, its bars unchanged.

**Rule:** X1-X3 pass → `nativeInt8MoE` on, X4 at night as the prefill confirmation; X4 failing turns it off. Any of
X1-X3 failing keeps it off.

**MoE int8, by day (2026-10-04, `e9ce33b4`'s tree): X1 and X2 pass; X3 moves to the night.**
- **X1 passes.** `TestMoEW8_X1_kernelsMatchReference`: 29,952 outputs bit-identical to the Go reference (three shapes,
  6 stacked experts, three routed slots each, overwrite and weighted accumulate), compiled precise.
- **X2 passes.** `TestMoEW8_X2_assembly`:
  - (a) identical experts against the dense FFN, both native int8: min cosine 1.000000, argmax 16/16;
  - (b) distinct experts against the CPU int8int8: native min cosine **0.999250**, 0 hard flips; the int4 re-quant arm
    0.404219, 4 hard flips.
  - Reading every slot's weights from the first routed expert fails both X1 and X2b.
- **X3 did not run by day.** The Metal memory guard declined the native int8 slice: the build needs 5.46 GB (int8
  weights, host copy, KV) against a 3.56 GB budget, 70% of the memory live-available with the owner's apps open. Not
  bypassed. It goes on the night queue unchanged.
  - Its f32 reference comes from nobara as F3′'s 1.5B did (`TestW8F3Reference_write` on the slice, same sha256
    `7e48d607`), file sha256 `edccc9374170749a`. The prompts come from the slice's own `tokenizer.json`.
  - The amendment is the same as F3′'s: the reference file only; every graded arm on the Mac; bar unchanged.
- **X4** (`TestDG01_expertMajorMoEPrefill` at `GOINFER_DG01_QUANT=int8int8`, the gate checking the resident is on the
  native MoE path) runs in the same night job, after X3.

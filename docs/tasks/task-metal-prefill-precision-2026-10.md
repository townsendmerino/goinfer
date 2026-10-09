# Task — Metal's batched prefill loses precision on Qwen2.5 (scoped 2026-10-09)

**Status:** **STEP 0 DONE 2026-10-09 — THE PREMISE IS REFUTED. Recommended close; no fix written.** The batched f16
pass is the MOST precise of the four prefill arms, not the lossy one: against an f64 truth on the same int4 weights it
is 27-54x closer than the decode path at layer 0, and 4-8x closer per layer at depth. The collapses S16's control saw
are W4A8's per-row int8 activation scale on Qwen2.5-VL-3B, and they hit the CPU-throughout arm too. Fixes (a)-(c) and
gates G-P1 to G-P3 were not run: there is nothing in the batched pass for them to fix. See "Step 0 — result" below.
(Scoped 2026-10-09, gates registered before any code; owner: "yes" to scoping it and to starting Step 0.)
**Prompted by:** S16's real gate (`docs/tasks/task-multimodal-support-2026-10.md`, "By day 2026-10-09"). S16 stays off
for Qwen2.5-VL until this is fixed, because turning it on would move image turns onto the lossy pass.

## What is known

- **The symptom.** On Qwen2.5-VL-3B (int4, its directory sidecar, the M1 Pro), S16's text control decodes 8
  teacher-forced steps from the K/V that Metal's batched f16 prefill leaves:
  - **Some steps fall to cosine 0.70-0.75 against CPU-throughout**, each with a different argmax (n = 512 steps 7-8,
    n = 1024 step 1, n = 1500 steps 5 and 8; -0.36 at worst over S16's four seeds and up to 1,562 rows).
  - The same steps from the CPU prefill uploaded to Metal read 0.948-1.0. From Metal's sequential path (decode's
    kernels) they read 0.82-0.998.
  - The last prompt row is fine (0.95-0.99 random tokens, 0.82-0.99 real text). No non-finite values; max |logit| 12-28.
  - Logs: `docs/measurements/multimodal-support-2026-10/s16/real-2026-10-09/q25vl-diag.log` and `q25vl-dec.log`.
- **Live in production.** Every Qwen2.5-VL text prompt over the batched-prefill floor takes this pass today. Whether
  Qwen2.5-1.5B and 7B (the bench set's text models) show it too is part of Step 0. Their §3.2 prefill gate graded
  K = 256-3,900 by KL, which can hide sporadic steps.
- **Where the batched pass keeps values in f16** (`metal/prefill.go`):
  - the residual stream (`xF` is `half`; `residual_f16` adds in float and rounds back to half);
  - every GEMM's activations and outputs (`gemm_w4f16_*`: A in half, C stored as half);
  - the norms' inputs and outputs (`rmsnorm_f16`, `rmsnorm_quant_f16`).

  Decode (the sequential path) keeps its residual in f32.
- **The hypothesis, to be measured, not assumed:**
  - Qwen2.5 models carry a few very large activations in the residual stream (the attention-sink dimensions), and f16
    near 8,192 has a step of 8. So the residual stream, rounded to half after every sublayer, loses the small
    components riding on those dimensions.
  - Post-norm activations are O(1) and survive f16, so the GEMM inputs may not be the problem at all.
  - S17 lever B's f16-activation failure on the vision towers is the same family of failure, at the GEMM input.

## Step 0 — localize, by day (minutes on the real checkpoint)

1. **Residual magnitudes:** max |residual| per layer on Qwen2.5-VL-3B for the control prompt and for real text, from
   the CPU forward (the capture seam, `KVCache.captureLayers`).
   - If no layer exceeds about 2,048 (where the f16 step reaches 2), the residual hypothesis is weak; go to step 3.
2. **Per-layer differencing** (CLAUDE.md: name the first divergent layer and op in one run):
   - a test seam in `prefillLast` copies `xF` after each layer;
   - compared row by row against the sequential path's residual on the same rows (the `ResidualAll` path, per layer
     through a matching seam);
   - the first layer whose divergence jumps, and whether it jumps at the residual add or inside a sublayer.
3. **The cheapest falsifying prototype, as a test seam:** the residual stream in f32 (norms read f32 and emit f16 for
   the GEMMs; the o-proj and down-proj outputs added in f32). Re-run the decode-step control.
   - If the 0.70-0.75 steps rise to the uploaded-CPU arm's level, the cause is found.
   - If not, the next prototype is f32 GEMM outputs, then f32 GEMM inputs (lever B's shape).
4. The same control on Qwen2.5-1.5B and 7B: whether the defect is the family's or this checkpoint's.

## Step 0 — result (2026-10-09, by day, the M1 Pro; exploratory diagnostics, not a graded gate)

Records: `docs/measurements/metal-prefill-precision-2026-10/step0/` (four logs; the two throwaway probes kept as
`.go.txt`, deleted from `metal/`; tree at 33ee6a01; checkpoints from `~/models`, Qwen2.5-VL-3B through its
`.int4.metal.giw`, Qwen2.5-Coder-1.5B through its `q4_k_m.int4.metal.giw`).

**The decode path quantizes activations; the batched pass does not.** Metal's decode GEMVs and every CPU int4 projection
are W4A8: each activation row is scaled by one max/127 into int8 (`decoder/actquant_hazard.go` describes the same
mechanism for Phi-3). The batched pass's GEMMs read f16 activations, about 11 significant bits against int8's 7 for the
row's largest value and far fewer for the small ones. S16's control graded the batched pass against the CPU prefill,
which is W4A8, so it measured agreement with the less precise arm.

1. **Residual magnitudes (step 0.1).** On the CPU forward, dimension 318 sits at about 1,500-2,500 from layer 2 to 31,
   with at most one value per layer above 2,048. The f16-residual hypothesis was already weak.
2. **Layer 0's V against an f64 truth.** The truth uses the same int4 weights dequantized, with f64 activations and
   accumulation and no f16 or int8 step (the K/V probe, n = 1,024, seed 1). It needs no residual, so any difference
   is the projection's own arithmetic.

   | layer-0 V, relative L2 vs f64 truth | Qwen2.5-VL-3B | Qwen2.5-Coder-1.5B |
   |---|---|---|
   | Metal batched (f16 activations) | **0.00045** | **0.00037** |
   | CPU int4, per-32 activation scales (`ActQuantGroup` 32) | 0.00814 | 0.00499 |
   | Metal sequential (W4A8) | 0.02453 | 0.01190 |
   | CPU prefill uploaded (W4A8) | 0.02453 | 0.01190 |

3. **Per layer, against the per-32 CPU arm ("F": the same int4 weights, near-f32 activations).** On Qwen2.5-VL-3B the
   batched K/V sits at relative L2 0.04-0.08 from layer 2 on. Both W4A8 arms sit at 0.14-0.42, with worst rows above
   1.0. On the 1.5B, batched is at 0.005-0.024 and W4A8 at 0.03-0.12. The large layer-2 V gap that started this
   (batched against sequential 0.457) is the sequential arm's error, not the batched pass's.
4. **Decode steps against two references,** 8 teacher-forced steps from five arms with E's greedy chain as the
   teacher:
   - **E** is CPU weight-only int8 from the safetensors directory: f32 activations and better weights, the near-truth.
   - **F** is the per-32 CPU arm above.

   Two runs: seed 1 at n = 512/1,024/1,500 (27 steps), and seeds 1-4 at n = 20 and 1,562, the S16 control's worst
   lengths (72 steps).

   | arm | mean cosine vs E | mean vs F |
   |---|---|---|
   | D, CPU throughout (W4A8) | 0.841 | 0.880 |
   | A, CPU prefill uploaded, Metal decode | 0.837 | 0.881 |
   | **B, Metal batched, Metal decode** | **0.878** | **0.943** |
   | C, Metal sequential, Metal decode | 0.838 | 0.881 |
   | F, CPU per-32 | 0.917 | n/a |

   The table is the 72 steps of the n = 20/1,562 four-seed cells. The 27 seed-1 steps at 512/1,024/1,500 had no
   collapse in any arm, and there B is the closest arm to F (mean 0.990, better than A in 20/27 steps) but not to E
   (0.963 against A's 0.969, better in 7/27). That is a small, unresolved difference on correlated steps, and it is
   not the effect S16 saw. The collapses below 0.3 hit **every W4A8 arm, CPU included**:
   - n = 1,562 seed 1 step 4: D -0.41, C -0.39, while A and B hold 0.94.
   - n = 20 seed 4 step 6: D 0.01, A -0.02, C 0.22, while B holds 0.986.
   - n = 1,562 seed 2 step 4: A 0.06, C 0.24, while B holds 0.95 and D 0.91.

   B collapses too on some steps, because its decode steps are W4A8 (n = 1,562 seed 4 step 1: -0.40, while F holds
   0.71). A few steps are low in every int4 arm and in F, against E only: int4 weight error, not activations.
5. **Hypotheses refuted on the way, recorded so they are not re-chased:**
   - an f16 residual stream (step 0.1);
   - f16 GEMM rounding (step 2: the batched pass is the closest arm);
   - f16 K/V storage or Metal decode attention mishandling W4A8 K/V: the CPU-throughout arm collapses on steps where
     the f16-cache arms do not.

**What follows:**
- **This task:** no change to the batched pass. Fixes (a)-(c) would make the most precise arm more precise. G-P1 to
  G-P3 are moot. Recommended close.
- **S16 for Qwen2.5-VL:** the reason it was held ("the lossy batched pass") is void. Turning it on moves image turns
  from the W4A8 CPU prefill onto the more precise arm. Its control needs a reference that is not W4A8 before it means
  anything: the per-32 CPU arm, or E. This is a re-registration of S16's bar for this family, so it is the owner's
  call, not made here.
- **Found, not owned by this task:** Qwen2.5-VL-3B is unstable under per-row W4A8 on these random-token prompts, on
  every backend. Per-32 activation scales recover most of it: mean 0.917 against 0.84, worst 0.25 against -0.47. That
  is `ActivationQuantHazard`'s territory, the Phi-3 guard. Whether real text shows it is unmeasured (S16's real-text
  control ran against the same W4A8 reference). Metal has no per-group activation kernel and declines `ActQuantGroup`
  to the CPU.

## The fix, chosen by Step 0's result (SUPERSEDED 2026-10-09: Step 0 found nothing to fix; kept as registered)

In order of cost, the first one that clears G-P1:

- **(a) an f32 residual stream** in the batched pass: `xF` in f32, `residual_f32`, norms reading f32. It changes no
  GEMM and doubles one buffer's bandwidth.
- **(b) f32 outputs** for the projections that feed the residual (o-proj, down-proj).
- **(c) f32 GEMM inputs** (lever B's `tower_gemm_w16` shape adapted to int4 weights). The costliest.

Whichever lands, decide by the measurement whether it applies to every family (more precise everywhere, if the speed
gate allows it) or only where Step 0 shows the loss.

## Gates, registered before any code (not run: Step 0 refuted the premise)

- **G-P1, decode-step agreement** (by day, the real checkpoints):
  - **The prompts:** S16's control (random tokens, seeds 1-4, n = 512, 1024, 1,500) and three real-text prompts of
    the same lengths, each decoded 8 teacher-forced steps.
  - **The bar:** the batched pass's worst step cosine against CPU-throughout is at least the uploaded-CPU-prefill arm's
    worst minus 0.005, and every argmax difference is an R10 near-tie.
  - **On Qwen2.5-VL-3B (must pass), and Qwen2.5-1.5B and 7B** (must pass if Step 0 shows the defect there, and must
    not regress otherwise).
  - **Planted defect:** the fix reverted to f16 for one layer must turn it red on Qwen2.5-VL-3B.
- **G-P2, no regression elsewhere:**
  - the Metal prefill and parity suites (`gate quick`);
  - the tiny prefill gates;
  - for a family the fix reaches, its existing prefill bar unchanged (the batched pass may only get closer to the
    sequential path).
- **G-P3, speed, at night:**
  - **What:** batched-prefill TTFT on a ~512- and a ~2,000-token text prompt, Qwen2.5-1.5B and Qwen2.5-VL-3B, the fix
    against today's pass, 5 passes, order alternating, under the timing lock.
  - **The rule:** median per-pass ratio new/old TTFT at most 1.03 (a precision fix may cost a little); 1.03-1.10
    parked for the owner; above 1.10 the fix narrows to the families Step 0 named.
- **Then S16 for Qwen2.5-VL:**
  - re-run G-S16c real (the control should now be healthy) and the served comparison;
  - ship by the S16 rule as registered.

## Cost

- Step 0: about an hour by day.
- Fix (a): a few hours with its tests.
- G-P1: about 10 minutes of real-checkpoint decoding.
- G-P3: one night job of about 20 minutes.
- (c), if needed: about a day.

# Task — Metal's batched prefill loses precision on Qwen2.5 (scoped 2026-10-09)

**Status:** SCOPED 2026-10-09, gates registered below before any code (owner: "yes" to scoping it). Not started.
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

## The fix, chosen by Step 0's result

In order of cost, the first one that clears G-P1:

- **(a) an f32 residual stream** in the batched pass: `xF` in f32, `residual_f32`, norms reading f32. It changes no
  GEMM and doubles one buffer's bandwidth.
- **(b) f32 outputs** for the projections that feed the residual (o-proj, down-proj).
- **(c) f32 GEMM inputs** (lever B's `tower_gemm_w16` shape adapted to int4 weights). The costliest.

Whichever lands, decide by the measurement whether it applies to every family (more precise everywhere, if the speed
gate allows it) or only where Step 0 shows the loss.

## Gates, registered before any code

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

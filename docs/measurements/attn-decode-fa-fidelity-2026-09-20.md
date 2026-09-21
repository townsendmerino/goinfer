# attn_decode_fa fidelity gate on held-out set B: PASSES on both cells (D7 decision, S confirmation)

Pre-registration: `attn-decode-fa-fidelity-PREREGISTERED.md` (committed before Phase A launched, not edited). S=16 was fixed in
`attn-decode-fa-ladder-2026-09-20.md` before any scoring. Gate: `TestFlashDecodeGateVsReference` (`cuda/flash_decode_gate_test.go`).
Reference: CPU f32 weights and activations, exact f64 attention, one worker, held-out prompt set B (`testdata/prefill-gate-prose-b/`),
built by the decoder test binary from `ec3cd97a` (D7@8000 5h45m, S@3900 ~25 min). Gate binary built from `897f6d29`. Both arms in one
binary and session, teacher-forced on the reference tokens, `skMinKeys = faMinKeys = 0`, 10 prompts x 64 positions.
Logs: `attn-decode-fa-fidelity-D7-2026-09-20.log`, `attn-decode-fa-fidelity-S-2026-09-20.log`. **Set B has now been scored once; a
re-run on it is a re-roll, not a check** (the parked-gate rule).

## Preconditions (all held on both cells)

1. Vacuity: lane launches >= layers x 63 decode rows on every prompt (checked by launch count) and the exact arm launched it 0 times;
   629-630 of 630 decode rows differ from exact. 2. A/A: two lane runs bit-identical over all 64 rows (prompt 1). 3. Seed row identical
   between arms on all 10 prompts. 4. The real-K/V f64 oracle (`attn-decode-fa-oracle-realkv-2026-09-20.log`): at all 28 D7 layers at K=8000 the lane's
   attention error is below the exact path's (median ~2e-7 vs ~1.5e-6 of max|ref|).

## Result

| cell | arm | hard flips | mean agree | mean KL | KL ratio | paired wins |
|---|---|---:|---:|---:|---:|---:|
| **D7 K=8000 (decision)** | exact | 7/640 | 92.50% | 0.053793 | — | — |
| | lane S=16 | 7/640 | 92.81% | 0.053050 | **0.9862** | 7/10 |
| S K=3900 (confirmation) | exact | 11/640 | 91.56% | 0.050040 | — | — |
| | lane S=16 | 10/640 | 91.72% | 0.051064 | **1.0205** | 7/10 |

(a) amended `lane <= exact + 2*sqrt(exact)`: met on both (strict `lane <= exact` also met on both). (b) agreement >= exact - 1.0 pt and
>= exact on >= half the prompts: met (+0.31 pt / +0.16 pt; 7/10 each). (c) KL <= 1.10x: met, and **outside the 1.05-1.10x parked band on
both**. Verdict per the registered rule: **PASSES** on the decision cell, and the confirmation cell agrees.

Paired KL deltas (lane - exact) across prompts: D7 mean -0.00074 +- 0.0018 (s.e.); S mean +0.00102 +- 0.00063. Neither is more than
2 standard errors above zero (S is ~1.6), so the registered "inside the pass region but > 2 s.e. above zero" flag is not triggered.

## Things to read alongside the PASS, not to explain away

- **D7 prompt 9:** exact KL 0.00207, lane 0.01431 (+0.0122), one lane hard flip against none, and D7's worst near-tie gap for the lane
  is **29.4% against 9.2% for exact**. It is one prompt of ten and the aggregate criteria pass with it in, but it is the largest single
  deviation in either cell and is not explained here. Prompts 4-6 have exact KL of 2e-5-4e-4 (near-deterministic continuations),
  where any change shows as a large relative jump on a tiny base.
- S has 7 of 10 prompts with lane KL above exact (a 2% mean excess, ~1.6 s.e.); D7 has the opposite sign. Two cells cannot say whether
  the lane is neutral or slightly worse on the smaller model. The lane's attention is *more accurate* than the exact path's against an f64
  recompute (precondition 4), so a systematic KL excess would not come from attention accuracy; it is unexplained if real.
- The step-1 result stands unchanged: the V-sum spike's earlier KL excess did not reproduce. This gate is about a different kernel.

## What a pass does and does not do

Per the pre-registration: it makes the lane eligible to be **offered opt-in** and defines what a default-on proposal must beat. It does
not flip a default, does not touch speculative decoding (the lane refuses to coexist with it), and moves no golden. **Still not done:**
the same-session peer sweep, a parity-goldens run with the lane on for a family beyond Qwen2.5 dense (gemma3-1b is covered only by the
synthetic wiring check and the real-K/V oracle at K=3000), and any served-quality check on chat-templated prompts.

# OP1-o1: R17's speed re-grade of the routed attention_fa_blk (2026-11-09)

## Registration (written and committed before the run)

- **Question.** Does routing `attention_fa_blk`'s score through `attn_score` (PR #810) change the kernel's speed? Its output does
  not change: without a cap it is bit-identical to the graded kernel (`backend/metal/attn_fa_blk_test.go:TestAttnFABlk_routedScoreIsTheGradedKernel`,
  the resident's own pipeline on the 1.5B and 7B, G1-K's 1,097 digests). So R17's kernel-accuracy and fidelity readings
  (`docs/measurements/metal-decode-attn-r17-2026-09-25.md`) are unchanged by construction, and are not re-run.
- **Instrument.** `backend/metal/decode_attn_r17_test.go:TestR17AttentionProto`, one process, on the MacBook (M1 Pro, 16 GB),
  under the exclusive timing lock. Its arms:
  - A, "current (production)": the resident's own `r.pAttnFA`, the routed kernel;
  - B, "prototype S=16": `r17Kernels` compiled on its own, the graded text, at production's split of 16;
  - the no-op attention arm, the baseline both subtract.

  Checkpoint: `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw`, depth 3900, 10 reps × 20 tokens.
  `GOINFER_METAL_R17_ABBA=1` alternates A and B within each rep (A B, B A, …), so the reps run A B B A.
- **Metric.** The median, over the 10 reps, of the rep's ratio A ÷ B of in-sequence attention ms (the arm's median token less the
  no-op arm's). The harness prints it as "speedup"; its KILL/PARK/SHIP labels are R17's and do not apply here.
- **Bands.**
  - median ≤ 1.02: PASS. `r17ShippedKernelSourceSHA256` moves to the routed text's hash (`293f868f…`), and #810 merges.
  - 1.02 < median ≤ 1.05: ambiguous. Re-run once in the same session. Ambiguous twice means parked, for Francis.
  - median > 1.05: FAIL. The pin does not move, and #810 is closed unmerged.
- **Stopping rule.** One run; one more only in the ambiguous band.
- **Void rule** (the timed-run skill): the cell is void if, at its start, the 1-minute load average is above 4.0, or reclaimable
  free memory is below the fit need plus 2 GB. A void cell is re-run and listed here, never averaged.

## Reading: PASS (2026-11-09, 03:18:50–03:20:08 PST)

- **The cell:** one, not void. At its start load1 was 3.08 (bar 4.0) and reclaimable memory 6.06 GB (bar 3.6 GB: the fit
  check's 1.12 GB dense plus 0.44 GB KV, plus 2 GB). Commit 944e5113, the exclusive timing lock, the test binary built
  beforehand. Raw: `run.log`.
- **Metric: median A ÷ B = 1.0161, inside the PASS band (≤ 1.02).** The harness prints it rounded as "speedup 1.02x".

| rep | order | A routed (ms) | B graded (ms) | A ÷ B |
|---:|---|---:|---:|---:|
| 1 | A B | 2.457 | 2.455 | 1.0008 |
| 2 | B A | 2.337 | 2.431 | 0.9613 |
| 3 | A B | 2.453 | 2.395 | 1.0242 |
| 4 | B A | 2.451 | 2.395 | 1.0234 |
| 5 | A B | 2.437 | 2.423 | 1.0058 |
| 6 | B A | 2.469 | 2.157 | 1.1446 |
| 7 | A B | 2.539 | 2.429 | 1.0453 |
| 8 | B A | 2.232 | 2.235 | 0.9987 |
| 9 | A B | 2.520 | 2.498 | 1.0088 |
| 10 | B A | 2.482 | 2.361 | 1.0512 |

In-sequence attention is the arm's median token less the no-op arm's (median 8.954 ms full token). The arms' own medians are
2.455 ms routed and 2.409 ms graded, and full token 11.425 and 11.278 ms; the full-token median ratio is 1.0034.

- **What it does and does not say.**
  - The routed arm was slower on 8 of the 10 reps. That is a +1.6% point estimate, against per-rep spread of roughly ±5%
    (0.961 to 1.145).
  - A sign test on 8 of 10 is not significant at 0.05, and the order halves do not split (A first 1.009, B first 1.023).
  - So the run does not resolve a direction. It reads inside the registered band, and the band decides.
- **Consequence, as registered:** `r17ShippedKernelSourceSHA256` moves to the routed text's hash, `293f868f…`
  (`backend/metal/attn_fa_blk_test.go:r17ShippedKernelSourceSHA256`), and #810 merges.

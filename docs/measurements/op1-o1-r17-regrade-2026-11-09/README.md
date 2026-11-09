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

## Reading

(filled in after the run)

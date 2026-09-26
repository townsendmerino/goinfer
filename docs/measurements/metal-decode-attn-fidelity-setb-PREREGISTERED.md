# Metal decode-attention fidelity decision on set B — PRE-REGISTERED 2026-09-25

**Not edited after any result.** Results are recorded in
[`metal-decode-attn-r17-2026-09-25.md`](metal-decode-attn-r17-2026-09-25.md). This file is committed, and the run is
built from the commit that adds it, before any set-B decode-attention cell is computed.

## What is decided

This run grades two candidate decode-attention kernels against the shipped exact per-query-head `attention` kernel
(W4A8 int4 weights, int8 activations), on the Metal backend:

| candidate | status today | how it runs |
|---|---|---|
| `attention_fa`, production split rule | default-on since 2026-09-21; the gate PASS that admitted it is void | `GOINFER_METAL_R17_CAND=attention_fa GOINFER_METAL_R17_SPLIT=0` |
| R17 prototype `attention_fa_blk<G>`, **S = 16** fixed | test-only | `GOINFER_METAL_R17_CAND=proto GOINFER_METAL_R17_SPLIT=16` |

A reported-only control also runs: `exact-null` (the exact kernel with each 4-term q·k partial sum reversed).

## The owner amendment (2026-09-25)

**Scope.** Decode-attention kernels that change only the reduction order: `attention_fa` and the R17 prototype. Prefill
lanes, activation or weight quantization lanes, and every other §3.2 gate keep the strict §3.2 form. For those,
the candidate is expected to be better, and the strict form was designed for that case.

**Change.** For these kernels, the 2026-09-18 policy's gate (`docs/tasks/red-october.md`: "the same pooled §3.2
gate ... not a relaxed one") is replaced by the two parts below.

- **P1, kernel accuracy vs float64**, is a new precondition.
- **P2, the end-to-end gate.** critA, critB and the per-cell 1.1× ceiling are unchanged. critC is replaced by CUDA
  R6's KL-ratio form. The strict critC is printed beside it on every run and does not decide.

**Mechanism.** Recorded in `metal-decode-attn-r17-2026-09-25.md`. It is not "a number moved":

1. *Structural.* Strict critC (pooled mean KL ≤ exact **and** lower on at least half the prompts) has no noise
   allowance. Its sign clause fails a truly equal arm about 40–50% of the time. It was written on 2026-09-05 for a
   candidate predicted to be *better* (f16-activation prefill, 4–15% lower KL).
2. *Measured.* Accuracy-neutral variants of the shipped kernel itself (a reversed q·k order, 1–7 ulp softmax-scale
   nudges, chunked or reversed V sums) fail strict critC 8 of 18 times over all ten set-A prompts, and 12 of 18 times
   on the six whose references are valid.
3. *Resolution.* Kernels measured about 3× closer to float64 than the shipped kernel (median and p99, 1.5B and 7B)
   fail strict critC. At 10 prompts the end-to-end KL resolves about ±2–3% of the exact arm's KL, and it cannot
   resolve kernel arithmetic errors of 1e-7.
4. *Parity.* CUDA R6's decode-attention lane was registered under, and shipped on, the R6 form.

## Disclosure: what has been seen before this registration

- **Set A.** Every set-A gate result is void as a verdict. Four of ten set-A prompts are scored against references
  generated from different text. That covers R2's 2026-09-21 PASS, the clean fails, and all the null runs. On the
  valid six prompts the candidates sat at 1.018× (`attention_fa`) and 1.030× (prototype S=16). Those numbers are
  post hoc, from development data, and informed this amendment.
- **Kernel accuracy on set-A prompts** (3 prompts at 3900 keys, 1 prompt at 2048 and 3900): both candidates had a
  lower median and p99 than the exact kernel on both models. On the **1.5B**, both candidates' **single worst head was
  worse** than the exact kernel's (a layer-0 head, 2.6× worse). So the max is reported and **not** gated, and that is
  stated here in advance.
- **Set B.** No Metal decode-attention comparison has been computed on set B's S-K3900 references. Its references
  (`~/goinfer-logs/prefill-ref-b/`, written 2026-09-09 12:29) were generated from the frozen snapshot
  `testdata/prefill-gate-prose-b/`, which has not changed since. The gate's per-prompt identity check verifies this at
  run time.

## P1 — kernel accuracy vs float64 (both models)

- **Test.** `TestR17KernelAccuracy` with `GOINFER_PREFILL_GATE_PROMPTS=b`, `GOINFER_METAL_R17_ACC_ARMS=decision`,
  `GOINFER_METAL_R17_PROMPTS=10` and `GOINFER_METAL_R17_DEPTHS=2048,3900`.
- **Models.** The 1.5B and the 7B, each from its `.int4.metal.giw` sidecar in `~/models`.
- **What is measured.** For each prompt and depth there is one decode step, encoded layer by layer. On every layer
  each kernel runs standalone on the same captured q/K/V, and is compared per head with a host float64 attention
  computed from the same f32 q and f16 K/V.
- **Pass, per model and per candidate:** the candidate's pooled **median** per-head relative L2 error is ≤ the exact
  kernel's, **and** its **p99** is ≤ the exact kernel's. The capture sanity check must be 100% bit-identical. The max
  is reported, not gated.

## P2 — the end-to-end gate (1.5B)

- **Test.** `TestR17_decodeFidelityGate` with `GOINFER_PREFILL_GATE_PROMPTS=b` (the gate derives the reference
  directory from the set).
- **Setup.** `~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, K=3900, 10 prompts × 64 teacher-forced positions, the
  executor flushed at every arm toggle.
- **Identity precondition.** Every prompt's seed-row KL(reference ‖ prefill) must be ≤ 1.0. Otherwise the verdict is
  **VOID**, and this registration is re-run only after the references are fixed.
- **Verdict.** The gate prints the "decode-attention verdict (2026-09-25 amendment)" line:
  - **PASSES** if critA ∧ critB ∧ the 1.1× ceiling ∧ KL ratio (pooled candidate/exact mean KL) ≤ 1.05;
  - **PARKED** if the same holds except that 1.05 < ratio ≤ 1.10;
  - **DOES NOT PASS** otherwise.
- **The 7B is not graded end to end.** This Mac has no 7B decode reference at K=3900, so P1 alone covers the 7B.

## Decision rule and consequences

A candidate's fidelity verdict is **P1 passes on both models ∧ P2 PASSES**.

**`attention_fa`:**
- **Pass:** the default stays on, recorded as re-gated.
- **Parked or fail:** the 2026-09-18 policy applies, and the default reverts to opt-in. The owner confirms the revert
  commit.

**Prototype S=16:**
- **Pass:** R17 precondition 1 is met. Next come preconditions 2, 3 and 4, with parameters fixed here:
  - *Precondition 2* (≤ 3% at 128 keys): structural, because the kernel engages only at ≥ `attnFADepthFloor` (1536)
    keys, like `attention_fa`. It is checked in the wiring.
  - *Precondition 4, the confirmation run:*
    - setup: a fresh session, the 1.5B `.gguf`, idle-gated at load1 < 2.0;
    - harness: `TestR17AttentionProto` with `GOINFER_METAL_R17_DEPTHS=3900 GOINFER_METAL_R17_SPLITS=16
      GOINFER_METAL_R17_REPS=7 GOINFER_METAL_R17_TOKENS=20`, the current kernel as the do-nothing arm;
    - graded on the median in-sequence attention speedup against R17's registered bands: ship ≥ 2.5×, park 1.5–2.5×,
      kill < 1.5×.
  - *Precondition 3, instrument amended* (differencing single post-idle tokens is not resolvable): the candidate's
    median **full-token** time right after 2 s idle must be ≤ the current kernel's, over the same 7 reps.
- **Parked or fail:** R17 is parked or killed on fidelity under its own rules.

**`exact-null`:** its verdict is reported. If it fails P2 under the amended form, that is recorded as a false fail of
the amended form and flagged for the owner. It does not re-grade anything.

**One run.** Every measurement above is deterministic and runs once. A re-run needs a stated mechanism, never a
re-roll. Logs are archived under `docs/measurements/metal-decode-attn-r17-2026-09-25/`.

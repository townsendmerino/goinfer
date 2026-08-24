# Task (goinfer): the LM head — largest per-token cost, cheapest known fix

> **For:** Claude Code, in `~/tmcode/goinfer`, on the M1 Pro. Written 2026-08-24, from the
> W4A8 plumbing phase's closing diagnostic (recorded in `docs/task-w4a8-neon-bandwidth.md`):
> the int8-pinned LM head runs weight-only Q8 with **no `Workspace` — a fresh 151936-row
> scratch allocation every token — achieving 11-13 GB/s against 40+ expected**, making it the
> single largest per-token cost in 1.5B decode, bigger than the entire W4A8 matmul. Scope per
> that phase's own close-out: goinfer-side, no aikit work expected. **This task gates Zeno
> Phase 1** — no competitive benchmarking with this on the hot path.

## Why this is two steps, not one

The finding bundles a mechanical defect (allocation per token) with a policy question (is
weight-only Q8 the right head path at all). They have different risk profiles, so they ship
separately, mechanical first.

## Step 1 — the allocation fix (bit-identical, no gate needed)

Find and eliminate every per-token allocation on the head path. Known suspects from the code:
`lmHeadN` does `logits := make([]float32, M*arch.VocabSize)` per call (~600 KB/token at this
vocab), and the weight-only Q8 path runs without a `Workspace` — persist both across tokens
(the decoder scratch machinery that already holds attention buffers is the natural home).
Same kernel, same math, same buffers' contents — the logits must be exactly `==` before and
after, which makes this the zero-risk portion; prove it with an exact-equality check, not
rel-err. Then re-measure the head's streaming rate with the stub method. If allocation was
the whole story, the head approaches the box's measured streaming rates and Step 2 may not be
needed at all — that's the cheapest good outcome, take it.

Watch one thing: bit-identity hides dispatch inertness (the named lesson from the plumbing
phase). The observable that the fix is live is the head's measured GB/s, not any test.

## Step 2 — quant policy, only if Step 1 leaves real money on the table

If the allocation-fixed head still streams well below the W4A8 kernels, the remaining gap is
the weight-only path itself (f32 activations, no SDOT), and the candidate is moving the head
to W8A8 — int8 activations through the same class of kernel the rest of decode uses. This is
the **precision-gated** portion, because the head is int8-*pinned* for quality reasons and its
output feeds sampling directly:

- Gate: the quality/cosine sweep shape plus an argmax-agreement check on real prompts — logits
  numerics shift, so greedy outputs can flip; measure how often before deciding, and record
  the number either way.
- Spec-decode: the head change applies identically to decode and batched verify (same kernel,
  M-independent per-output order), so decode==verify survives — assert it with the spec
  parity gates, same as the plumbing phase did.
- If the gate says the flips matter, Step 2's answer is "no, pinned stays" — recorded, done.
  Step 1's win ships regardless.

## Measurement and acceptance

- Stub-method head component, before / after Step 1 / (if taken) after Step 2, both model
  sizes, quiet box.
- End-to-end `bench_peer` cells against the plumbing phase's corrected baselines (1.5B
  22.9-23.5, 0.5B 45.31 tok/s int4; the int8int8 cells share the head and should move too —
  re-run them, the zero-cost-guidance numbers get better for free).
- **Derive the projection band before measuring** (house discipline): head bytes ≈ 0.23 GB
  (1.5B) / ≈ 0.14 GB (0.5B) per token; at the current 11-13 GB/s vs a fixed rate near the
  W4A8 kernels', back out expected ms saved and predicted tok/s, and hold the measurement to
  it. Note the 0.5B case is proportionally the bigger win — the head is a far larger share of
  its token — so if the 1.5B number moves and the 0.5B number doesn't, something's wrong.
- Results, including a Step-2 "no" if that's the answer, appended to the campaign doc's
  follow-up section.

## Not in scope

- aikit changes — `Workspace` machinery exists; if a real API gap surfaces, record it for the
  next aikit release rather than forcing one for this.
- LM head at int4 (quality question beyond this task), any W4A8/attention re-tuning, the
  `.giw` kind, Zeno work (this task unblocks it; it doesn't include it).

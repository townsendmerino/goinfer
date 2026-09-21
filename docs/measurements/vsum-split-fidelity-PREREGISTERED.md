# PRE-REGISTERED — V-sum split spike, §3.2 fidelity gate

**Written 2026-09-13 BEFORE Phase A was launched. Not edited after any result was seen.**
The record goes in `vsum-split-fidelity-2026-09-13.md`; this file stays as written.

## The question

`GOINFER_SPLITKV_VSUM_SPLIT` splits the decode V-sum across the key axis (`splitkv_vsum_partial` +
`splitkv_vsum_combine`). It measures +40.4–42.7% on the attention block and is **not bit-identical**
to the shipped path — it changes the reduction tree. `vsum-split-spike-2026-09-13.md` §"What this
does NOT establish" owes this gate: an agreement / hard-flip / KL comparison against a reference
that is neither arm.

The shipped test `cuda/vsum_split_fidelity_test.go` does **not** answer it. It scores the spike
against the bit-identical path — the exact "scored the exact path and called it truth" mistake that
`docs/task-prefill-gap.md` §3.1 corrected — on synthetic embeddings, in a near-tie regime where
argmax flips on nothing. Its 85.4% top-1 number is not evidence and is not carried forward.

## Form: §3.2, reused not reinvented

`decoder/prefill_ref_gen_test.go` (Phase A, CPU, own process) and `cuda/prefill_gate_ref_test.go`
(Phase B) already implement this gate for prefill. **Both arms are scored against a third thing**:
the CPU backend's own forward with f32 activations and `GOINFER_CPU_FAST_ATTENTION=0` (the exact
f64-accumulating attention). Reference rows are the prompt-final logits plus 64 teacher-forced
continuation steps — and that continuation already runs the resident **decode** path one token at a
time (`rf.Forward(...)`, `cuda/prefill_gate_ref_test.go:284`), which is exactly where this change
lives. Phase B is therefore an arm swap, not a new harness.

## Arms

| arm | what it is |
|---|---|
| **exact** | today's default: bit-identical `splitkv_vsum`. `GOINFER_SPLITKV_VSUM_SPLIT` unset |
| **spike** | `GOINFER_SPLITKV_VSUM_SPLIT=4` (S=4 — the sweep over S=4/8/16 was flat, so 4 is the operating point) |

`GOINFER_SPLITKV_MIN_KEYS=0` in **both** arms, so both take split-KV and the only difference between
them is the V-sum tree. Teacher-forced on the same reference tokens, so a per-position difference is
attributable to the arm.

## Cells

| cell | model | geometry | K |
|---|---|---|---|
| **decision** | D7 — qwen2.5-7b-instruct-q4_k_m | nH=28 nKV=4 hd=128 (512 KV floats/key) | 8000 |
| confirmation | S — qwen2.5-coder-1.5b-instruct-q4_k_m | nH=12 nKV=2 hd=128 (256/key) | 8000 |

K=8000 is the depth the +40% was measured at, and no reference exists there — the box's existing
cells stop at K=1024 (D7) / K=3900 (S), so Phase A must generate new ones. 10 prompts per cell, set
**A**, 64 continuation positions. Reference weights **f32** (`GOINFER_CPU_REF_QUANT_D7=""`), matching
this box's existing D7 cells (`prefill-l2l3-phase3-2026-09-05.md`) rather than the Mac's int8
fallback; 62 GB fits the 28 GB f32 7B.

## Decision rule — per cell, all three, copied from §3's form

- **(a) hard flips**: spike's hard-flip count vs the reference ≤ exact's, over the same continuation
  positions.
- **(b) agreement, PAIRED**: spike's mean teacher-forced agreement ≥ exact's mean − 1.0 pt **AND**
  spike ≥ exact on ≥ half the prompts. Paired, not pooled — CLAUDE.md rule 7.
- **(c) KL**: spike's mean continuation KL(reference ‖ arm) ≤ 1.1 × exact's mean.

**AMBIGUOUS → PARKED.** If (a) passes and (b) lands in the last 0.2 pt of its allowance, or (c) in
1.05–1.10×, the cell is **inconclusive, not a pass**: recorded, no promotion, and the reason to
re-run is a mechanism, never a re-roll. The decision cell is D7; S confirms and cannot rescue a D7
failure.

## Three checks that must pass before any of the above is read

1. **VACUITY — the spike must actually be active.** The two arms' continuation logits must differ in
   ≥ 1 position, and `splitkv_vsum_partial`/`splitkv_vsum_combine` must appear in the spike arm's
   loaded pipelines while `splitkv_vsum` does not. *This is the check the shipped test lacks, and the
   failure it would have hidden: an inactive spike scores a perfect pass.*
2. **SPLIT-KV must be taken in both arms** — not the single-block path, not the CPU fallback. Every
   run greps its log for `DECLINED`; an earlier session silently profiled the CPU for ten minutes
   after a 0-byte allocation took the resident path down.
3. **A/A determinism.** The spike run twice on the same input must be bit-identical to itself. It
   gives up bit-identity to *history*, never determinism — the combine is a fixed ascending order and
   never atomics. If A/A is not bit-identical, the gate stops and the kernel is wrong.

## Prediction on record, written before the run

**The spike passes, and is more likely to beat exact than to lose to it.**
`reduction-tree-accuracy-2026-09-12.md` measured a blocked fold 1.76–4.93× closer to f64 than the
sequential left fold it replaces; a shorter per-accumulator chain is the whole mechanism. The
sequential fold at K=8000 is 8000 adds deep, which is where it should be worst.

Registered so it can be wrong in public: I predicted the spike's *speed* at 5–15% twice and measured
+40%. That was a reasoning failure about stall reasons, not about arithmetic — but it is the reason
this prediction is written down rather than recalled afterwards.

## What a PASS does and does not buy

A pass does **not** flip the default. `cuda/prefill.go:701` records the invariant: the exact path
"remains bit-identical to the M=1 decode kernels, remains what spec-decode verify and the parity
gates run". Promoting the spike breaks (1) `TestSplitKV_bitIdentical` by construction, (2)
spec-decode losslessness — `--drafter` and `--spec ngram` ship *gated lossless*, which holds only
because a verified token equals a decoded one — and (3) the decode goldens. A pass buys the right to
keep it opt-in **with evidence**, and it is the prerequisite for the re-canonicalisation scoped in
`docs/scoping-decode-tree-recanon.md` (79 test files, 116 goldens, marked DO NOT START). Nothing
here authorises that.

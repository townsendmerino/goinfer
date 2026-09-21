# W4F16 decode lane — fidelity gate, pre-registered 2026-09-21

Pre-registered per R1's own Build gate (3) (`docs/tasks/red-october.md`), before any gate cell
runs, per this repo's own rule that the pre-registration must exist first.

## What is being decided

Whether `GOINFER_METAL_DECODE_LANE=w4f16` (the f16-activation decode GEMV lane; shipped path
`W4A8`, int8-activation) may become a candidate for default-on, subject to the separate served-
tok/s band in the same brief. **This gate alone does not flip any default** — same invariant as
every other fidelity gate in this repo (`cuda/vsum_split_gate_test.go`'s own header).

## Reference

The CPU f32-weight, f32-activation reference already built for the prefill gate
(`decoder/prefill_ref_gen_test.go`, `TestPrefillGateReference`) at K=64: prompt-final logits plus
64 teacher-forced continuation rows through the resident **decode** path (`ForwardForTest`, one
token at a time), exact f64-accumulating attention (`GOINFER_CPU_FAST_ATTENTION=0`). This is
already the right shape for a decode gate (K=64 is a prefill depth of 64 tokens feeding into a
64-position teacher-forced decode continuation — the "10 prompts × 64 positions" gate (3) asks
for) — no new reference generation needed. Files: `~/goinfer-logs/prefill-ref/S-K64-p{0..9}.bin`,
10 prompts, prompt set "a" (`GOINFER_PREFILL_GATE_PROMPTS` unset), freshly regenerated 2026-09-20
for R6's own gate and confirmed still valid.

- **S (qwen2.5-coder-1.5b, int4)** is the decision cell, matching the brief.
- **D7 (qwen2.5-7b)** is the confirmation cell **only if its K=64 reference already exists or fits
  the fit-guard on the day without a bypass**. As of this pre-registration no D7-K64 reference
  file exists on this machine, and generating one loads a second CPU model at ~7.6 GB alongside
  this session's own ~11 GB of ambient overhead (VSCode + this Claude Code session) — the same
  class of near-incident this session already hit once today building a *smaller* checkpoint
  (`docs/measurements/r12-vision-peer-2026-09-20.md`, `mac-16gb-model-size-limits` memory). D7 is
  **not attempted** unless the box shows real headroom at the time; its absence does not affect
  the decision (S is what decides, per the brief).

## Arms

Both built from **one Metal resident** (`qwen2.5-coder-1.5b`, `Quant:"int4"`), the lane toggled at
runtime via `r.decodeLaneW4F16` (the same technique `docs/measurements/r1-layer26-rootcause-2026-09-20.md`
validated: `canUseF16Lane` reads the field on every encode call, so a runtime flip is honored
mid-process, confirmed by the arms NOT being bit-identical). Both arms build the K=64 prefill via
**sequential single-token `Forward` calls** (not the batched `PrefillLast` fast-prefill path,
which is a separate, already-shipped mechanism this gate is not about) so every position — prefill
and continuation alike — goes through the same per-token dispatch sites R1 touches. Both arms are
then teacher-forced on the reference's own greedy continuation tokens (never either arm's own
output), exactly as `metal/prefill_gate_ref_test.go`'s established `teacherForceOnRef` does.

## Criteria — reusing the implemented §3.2 pooled form, not re-deriving

`docs/completed/task-prefill-gap.md` §3.2 defines the pooled criteria; `metal/prefill_gate_ref_test.go`'s
`poolCells`/`cellSummary` is its one implementation in this repo, already used for two decode/prefill
gates (the Metal fast-prefill lane, and R6's flash-decode attention). `cuda/vsum_split_gate_test.go`
amended its own criterion (a) by owner decision on 2026-09-13 to match this exact implementation,
noting the brief's originally-registered strict form (`fast HF <= exact HF`) has no resolving power
at these sample sizes. **This gate reuses `poolCells` verbatim** (same package, same function,
called directly — not a new formula written for this gate) rather than re-deriving R1's own
brief-prose criteria (`hard flips ≤ exact + 2√exact, agreement ≥ exact − 1.0 pt and ≥ half the
prompts, KL ≤ 1.10× exact with 1.05–1.10 parked`), which are close to but not identical to the
implemented form (criterion (b) here is noise-aware — `exact − 2√d/N` — not a flat 1.0-point
margin; criterion (c) here is a boolean 1.1× ceiling, not a three-way ship/park/kill split). This
choice is made explicitly, before any cell runs, for the same reason vsum's amendment gives: reuse
of a vetted implementation over a fresh, unvalidated one.

- **(a)** pooled hard-flip count, f16 lane ≤ W4A8 + 2·√(W4A8) (`NearTieArgmaxForTest`'s 3%-near-tie rule)
- **(b)** pooled top-1 agreement, f16 lane ≥ W4A8 − 2·√d / N (d = positions where exactly one arm
  matches the reference's own argmax)
- **(c)** pooled mean KL(reference ‖ arm), f16 lane ≤ W4A8's, AND f16 lower on ≥ half the 10
  prompts (paired), AND no single decision cell's f16 mean KL > 1.1× that cell's W4A8 mean KL

A pass requires all three. **AMBIGUOUS → PARKED, not shipped**: this gate has one decision cell
(K=64) rather than the pooled-over-K form other gates use (K=64 is R1's own gate-3 depth, not a
multi-K sweep), so there is no separate "1.05–1.10 KL band" to check independently of criterion
(c)'s own ceiling — criterion (c) failing outright at the ceiling is the parked/killed signal here.

## Decision rule

Ships only if **both** this fidelity gate passes (all three criteria) **and** the separately
registered served-tok/s band passes (R1's own brief: 1.5B ships at ≥92 tok/s, parked 81–92, killed
below 81; 7B ships at ≥27.5, parked 24–27.5, killed below 24). A fast arm that fails this fidelity
gate is a negative regardless of speed — the brief's own precondition, unchanged here.

## What this does not establish

One prompt set (10 prompts), one K (64), one model size (S). Not a multi-K sweep. D7 confirmation
likely skipped for memory reasons (see above). Position-0 behavior specifically (the BOS
attention-sink token) was already characterized separately in
`docs/measurements/r1-layer26-rootcause-2026-09-20.md` and is included here only as prompt-position
0 of each of the 10 prompts, not isolated.

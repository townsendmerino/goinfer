# Metal short-prompt prefill floor — R3, 2026-09-20

**Result: floor lowered from 256 to 64 tokens.** `metal/backend.go`'s `metalFastPrefillFloor`
constant, gating `PrefillLast`'s batched f16-MMA prefill path vs the sequential per-token loop.

Brief: `docs/tasks/red-october.md` R3 ("Metal short-prompt floor, and what stays sequential").
Box: `apple-m1pro` (M1 Pro, 6P+2E, 16 GB). Model: S (qwen2.5-coder-1.5b-instruct-q4_k_m, int4).
goinfer commit `aaf73268` (dirty: this session's own in-progress edits). Ollama v0.32.5.

## Step 1 — fidelity gate (§3.2 pooled)

New CPU f32-activation reference generated for K∈{64,128} (`decoder.TestPrefillGateReference/S`,
`GOINFER_CPU_REF_KS=64,128`; 10 prompts each, prompt set "a"). The 1.5B model needed the fit
guard bypassed for this one load (`GOINFER_NO_FIT_GUARD=1`, user-approved) — 7.7 GB needed
(f32 weights) against ~6.8 GB free at the time, held down by ordinary desktop/editor overhead,
not the oversized-checkpoint class of risk this guard otherwise exists for.

`metal.TestPrefillGateVsReference/S`, `GOINFER_METAL_GATE_DECISION_KS` overridden per run:

| decision set | critA (hard flips) | critB (agreement) | critC (KL) | verdict |
|---|---|---|---|---|
| K=64 alone | exact=5 fast=3 (≤5+2√5) | exact=91.41% fast=92.03% | exact=0.0465 fast=0.0456, fast lower 7/10 | **SHIPS** |
| K=64+128 pooled | exact=8 fast=6 | exact=91.17% fast=92.03% | exact=0.0421 fast=0.0408, fast lower 15/20 | **SHIPS** |

Both pass all three §3.2 pooled criteria with no marginal calls — fast is at least as good as
exact on every axis at both cells, not just within noise. K=64 is the smaller of the two
registered candidates {64, 128}, and it ships on its own single-cell pool as well as pooled with
128, so — per the brief's own resolving-power caution about single-cell pools — the *combined*
result is the one being relied on for the decision, with the K=64-alone result as corroboration,
not the sole basis.

`startPos > 0` (the prefix-reuse dimension R3's Build section also names) was **not** re-tested
here. G-08 (`docs/audit-metal-2026-09-12.md`) already closed the coverage question for that code
path with a dedicated correctness test (`metal/prefill_startpos_test.go`, K=48/from=24, cosine
0.999941, no defect found) rather than extending this pooled statistical gate to a second
dimension — a judgment call from that session this one is deferring to rather than re-litigating,
since the decision rule below is stated purely in terms of K.

## Step 2 — speed (three-arm TTFT, `scripts/bench_peer_prefill.py`)

`goinfer_exact` (`--exact-prefill`, sequential), `goinfer` (batched, `GOINFER_METAL_FAST_PREFILL_FLOOR=0`
forced so the batched arm actually engages below the *current* 256 floor — see the false-negative
note below), `ollama`. Interleaved per depth, n=6 fresh prompts per cell, 32/64/128 newly
calibrated (`scripts/bench_prompts_calibrate.py`), 256 pre-existing. Metal serve binary built
fresh at this commit (`CGO_ENABLED=0`, `-tags metal,goinfer_testhooks`).

| K | exact tok/s | batched tok/s | batched/exact | ollama tok/s |
|---|---:|---:|---:|---:|
| 32  | 104.9 | 263.4 | 2.51x | 328.7 |
| 64  | 95.4  | 365.5 | **3.83x** | 485.6 |
| 128 | 90.6  | 422.2 | **4.66x** | 639.5 |
| 256 | 86.1  | 337.6 | 3.92x | 782.6 |

Both candidate floors clear the ≥2x ships band by a wide margin — this is not a borderline call.
(K=32 is informative only: no fidelity cell was gated there, so it is not eligible to set the
floor regardless of its speed number.)

**A methodology trap, caught and corrected before trusting the first two runs.** The first attempt
set only `GOINFER_METAL_FAST_PREFILL=1` (the enable toggle) and measured batched/exact ratios of
~1.0-1.1x at K=32/64/128 — apparently a clear miss on the speed band. That was wrong: the floor
(`GOINFER_METAL_FAST_PREFILL_FLOOR`, checked inside `PrefillLast` itself) is a **separate** gate
from the enable toggle, and it was still at its shipped default (256) in both the first and
second attempts — so the "goinfer" arm was silently falling back to the sequential path at every
depth below 256 in both of those runs, making it near-identical to `goinfer_exact` by
construction, not by a real absence of speedup. Only the third run, with
`GOINFER_METAL_FAST_PREFILL_FLOOR=0` set explicitly, actually exercised the batched path below
256 — producing the real numbers above. Recorded because it is the kind of false negative that
looks like a clean, decisive result if not checked.

## Decision

Per the brief's registered rule ("the floor moves to the smallest K in {64, 128} at which the
§3.2 pooled gate ships; at that K the batched arm must beat sequential by ≥2x on TTFT"): K=64
ships on both fidelity and speed. **New floor: 64.** Shipped as the new
`metalFastPrefillFloor` default in `metal/backend.go` (was 256, set 2026-09-12 per M-02).

## Out of scope, deferred

- **The `noHead` executor-job completion of M-01** (R3's Build section's conditional follow-on):
  gated on "the sequential residue measures above 10% of its TTFT in the LM head" — not measured
  in this pass. The `goinfer_exact` numbers above are the real current sequential-arm TTFT, but no
  LM-head-specific breakdown was taken. Left for whoever picks this up next; the trigger condition
  is unevaluated, not evaluated-and-declined.
- `startPos > 0` on the *speed* axis (as opposed to the correctness axis G-08 already covers) —
  not measured. A prefix-reuse chat turn's batched-vs-sequential ratio at K=64 with a nonzero
  `startPos` is a plausible follow-up if it turns out to matter in practice.
- CUDA's own floor (512, R3's own out-of-scope note) — untouched, separate question.

# PRE-REGISTERED — CUDA flash-decode lane (`attn_decode_fa`), fidelity gate on held-out prompt set B (R6)

**Written 2026-09-20 BEFORE Phase A (the set B reference) was launched and before the kernel exists. Not edited
after any result was seen.** The record goes in `attn-decode-fa-fidelity-2026-*.md`; this file stays as written.

## What is being gated

`docs/tasks/red-october.md` R6 step 2: a decode-attention lane that splits the key axis (S chunks, one CTA per
(kvHead, keySplit), all query heads of the group per CTA, online softmax, a deterministic fixed-order combine)
and is **not bit-identical** to the shipped exact split-KV path. It is opt-in until this gate passes; the exact
path stays what ships when the lane is off and what the parity gates and speculative-decode verify run.
`vsum-mechanism-2026-09-20.md` established that the previous spike's KL excess (1.0585x) does not reproduce on
the current build, that S=1 of that spike is bit-identical to exact, and that paired KL deltas there are within
about one standard error (~0.002) of zero. That result does **not** clear this lane: this is a different kernel
(it also changes the softmax reduction), on held-out prompts.

## Prompts, references, arms

- **Held-out prompt set B** (`testdata/prefill-gate-prose-b/`, ten snapshots, never used to score the spike). Set A is
  not used for any decision here.
- **Reference**: CPU backend, **f32 weights and activations**, `GOINFER_CPU_FAST_ATTENTION=0` (exact f64-accumulating
  attention), **one worker** (`GOINFER_CPU_REF_QUANT_D7=""`, `GOINFER_CPU_REF_WORKERS=1`), 64 teacher-forced greedy
  continuation positions after the prompt, exactly as `vsum-split-fidelity-PREREGISTERED.md`. Built once per cell, by
  the test binary built from the commit that carries this file; both arms are scored against the same rows.
- **Arms, per prompt, one binary, one session, back to back:** `exact` (lane off, the shipped bit-identical path) and
  `lane` (on, at its registered S). `skMinKeys=0` in both arms so both take the split-KV family and the lane is the
  only difference. Teacher-forced on the reference tokens.
- **Cells:** **D7** (qwen2.5-7b-instruct-q4_k_m, nH=28 nKV=4 hd=128) at K=8000 — the **decision** cell; **S**
  (qwen2.5-coder-1.5b-instruct-q4_k_m, nH=12 nKV=2 hd=128) at K=3900 — the **confirmation** cell, which cannot rescue a
  D7 failure. 10 prompts x 64 positions each.
- **The lane's S** is fixed before the gate is scored, from the kernel's own ncu/ladder record, and written into the
  record; it is not chosen by looking at gate numbers. Default 4 (the spike's flat trend: S=4 captured 98% of its win).

## Decision rule — per cell, all three, as in the parked gate (a) amended by the owner on 2026-09-13

- **(a) hard flips**: lane's count vs the reference <= exact + 2*sqrt(exact) (the strict `lane <= exact` is printed too).
- **(b) agreement, PAIRED**: lane's mean teacher-forced agreement >= exact's mean - 1.0 pt AND lane >= exact on >= half
  the prompts.
- **(c) KL**: lane's mean continuation KL(reference‖arm) <= 1.10 x exact's mean.
- **AMBIGUOUS -> PARKED**: (b) within its last 0.2 pt, or (c) in 1.05-1.10x, is inconclusive, not a pass; a re-run
  needs a named mechanism, never a re-roll.
- **Reading the noise floor into (c), registered now:** on D7 the exact arm's mean KL is ~0.10 and the paired
  per-prompt delta's standard error for a *benign* alternative tree was ~0.002 (S=1..16 of the previous spike, n=10),
  so the 1.05x line (+0.005) sits at about 2.5 standard errors. A lane result inside (c)'s pass region but with a
  paired delta more than 2 standard errors above zero is reported as such, with the delta and its s.e.

## Preconditions — all must hold before any criterion is read (a failed one voids the cell, it is not a fail)

1. **Vacuity**: the lane's decode rows differ from exact's on >= 1 position of every prompt, and the lane's pipelines
   (not `splitkv_vsum`) are the ones launched — checked by launch count, not by trusting the flag.
2. **A/A determinism**: two runs of the lane on the same input agree bit for bit over all 64 rows (first prompt).
3. **Seed row identical** between arms on every prompt (the lane touches M=1 decode attention and nothing else).
4. **Oracle check, before the gate (the new kernel's defect detector; S=1 identity is unavailable because online
   softmax is a different reduction):** on the real K/V of one set-B prompt at K=8000, the lane's attention output at
   every layer must be **no further from an f64 recompute than the exact path's own output is** (median and max
   over heads within 1.0x of exact's error, plus a small absolute allowance recorded with the result). A lane that is
   noisier than exact fails here and is fixed before the fidelity gate is spent.
5. Both arms use the same reference rows; no `DECLINED` in either arm's log.

## Speed, registered separately (does not gate fidelity, and fidelity does not gate the measurement)

Served, greedy, `scripts/bench_peer.py` protocol: **1.5B at 3900 >= 170 tok/s ships, 150-170 parked, < 150 killed;
D7 at 8000 >= 50 ships, 44-50 parked, < 44 killed** (R6 band; the previous spike alone reads 44.8). No regression
beyond 2% at 128. Paired lane-on/off arms, alternating order, fresh serve per arm, an idle box, at
{128, 512, 2048, 3900} on 0.5B/1.5B/phi3-mini/gemma3-1b and 8000 on D7. A geometry where the single-block path wins
(phi3 class) is expected to decline via the gate; a decline is not a regression, a forced regression there is.

## What a pass does and does not do

A pass makes the lane eligible to be *offered* opt-in and defines what a later default-on proposal must beat. It does
not flip a default, does not touch speculative-decode verify (which must keep the exact tree, so the lane declines
under spec-decode exactly as the previous spike's guard did), and moves no golden.

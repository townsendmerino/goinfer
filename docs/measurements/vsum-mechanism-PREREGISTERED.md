# PRE-REGISTERED — V-sum split spike: what explains KL 1.0585× on D7@8000 (R6 step 1)

**Written 2026-09-20 BEFORE the run. Not edited after any result was seen.** The record goes in
`vsum-mechanism-2026-09-20.md`; this file stays as written.

## The question

`vsum-split-fidelity-2026-09-13.md` parked the spike: on D7 at K=8000 its KL against the f32/f64
reference was 1.0585× exact's (8 of 10 prompts higher) while its argmax metrics were *better*
(27 vs 30 hard flips, +1.25 pt agreement); on S at 3900/8000 the KL ratio was 1.0026×. That record
says a re-run "needs a named mechanism, never a re-roll". `red-october.md` R6 step 1 names three
candidate shapes: a **defect in the spike's combine**, a **prompt-set effect**, a **reference effect**.
This run separates them. It re-uses Phase A's cached reference rows (`~/goinfer-logs/prefill-ref/
D7-K8000-p{0..9}.bin`, the 3h58m part) and re-scores arms only; it is **analysis of the existing cell,
not a promotion re-score** — any promotion decision needs a fresh pre-registration on held-out prompt
set B, as the parked record says.

## Arms (D7 = qwen2.5-7b-instruct-q4_k_m, K=8000, prompt set A, all 10 prompts × 64 positions)

`exact` (`skVsumSplit=0`, the shipped bit-identical path) and the spike at **S ∈ {1, 2, 4, 8, 16}**,
`skMinKeys=0` in every arm (as the original gate), teacher-forced on the reference tokens, one binary,
one session, arms run back to back per prompt. S=1 is made reachable by the one-line guard change
`skVsumSplit > 1` → `>= 1` in `splitKVAttnDecode` (0 stays "off"; the env-var loader still requires >1
and the spec-decode guard still keys on >1, since S=1 is bit-identical by construction).

## What is measured, per arm and prompt

Mean KL(reference‖arm) over the 64 positions, split **exactly** (no renormalisation, same 1e-12 floor as
`KLDivergenceForTest`, and asserted to sum to it) into the reference's **top-100 tokens** ("head") and
the remaining tokens ("tail"); hard flips and mean agreement as in the gate; **KL(exact‖arm)** (how far
the arm moved from exact, the perturbation size); and the paired delta arm − exact per prompt, its mean
and standard error across the 10 prompts. Per-position delta in three buckets (positions 1–16, 17–48,
49–63) to see whether the excess compounds through the continuation.

## Hypotheses and what each predicts (registered before any data)

- **H-defect** (the combine or partial kernel is wrong somewhere): S=1 is **not** bit-identical to
  exact on some row. This is a code fact, not a statistic — 0 + x == x, the partial fold is the same
  fused-multiply-add chain in the same order, and the combine multiplies by the same `invStore`; so any
  differing bit at S=1 is a defect and the mechanism. Also predicted by a defect: KL excess that does not
  behave smoothly across S.
- **H-noise** (any change of tree is a perturbation of the same size regardless of accuracy; downstream
  int8-activation rounding amplifies it and adds independent noise, which raises KL against a fixed
  reference quadratically): S=1 bit-identical; KL excess positive at every S≥2, **not ordered by tree
  accuracy** (roughly flat across S=2..16, within ±0.02× of one another), and it tracks KL(exact‖arm).
- **H-tail** (the excess lives in the 152k-token tail serving does not sample from): head-part delta
  ≈ 0 (|delta| ≤ 2% of exact's head KL) and the tail part carries ≥ 70% of the S=4 excess.
- **H-compounding**: per-position excess grows with position (bucket 49–63 ≥ 2× bucket 1–16).
- **H-prompt**: excess concentrated in ≤ 3 prompts (the record showed 8 of 10 higher, so this is
  predicted false; recorded so a change of story is visible).

**My prediction, so it can be wrong:** S=1 bit-identical (near-certain, by construction); excess at S=4
reproduces near 1.05–1.07×; and the pattern is **H-noise with a tail-dominated split** — I expect the
excess to be flat in S and mostly tail. I hold H-defect at under 10%.

## Decision rule for the written finding (fixed now)

1. **S=1 differs anywhere → DEFECT FOUND**; stop, fix, and the fidelity question restarts from the fix.
2. **S=1 identical and S=4 reproduces** (KL ratio within 1.0585 ± 0.02): the record's number is real
   and reproducible on this build; report which of H-noise / H-tail / H-compounding hold, each by its
   numeric test above. A mechanism is **"found"** only if at least one hypothesis's test passes **and**
   the others' predictions are consistent with it; otherwise the finding is **"not explained"**, stated
   as such, and the spike stays parked.
3. **S=1 identical but S=4 does not reproduce** (outside 1.0585 ± 0.02): the original ratio was
   build- or session-dependent; report it and do not unpark on this run either.
4. **Ambiguous band:** a hypothesis test landing within 10% of its threshold is reported as "ambiguous",
   not as a pass. **Nothing in this run unparks the spike by itself**: an unparked spike still needs the
   fidelity gate on held-out set B, pre-registered separately.

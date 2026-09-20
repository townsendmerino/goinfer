# V-sum spike mechanism run (R6 step 1): the 1.0585× KL excess does NOT reproduce, and S=1 is bit-identical

Pre-registration: `vsum-mechanism-PREREGISTERED.md` (committed `dcc7978f` before the run, not edited). Harness:
`cuda/vsum_mechanism_test.go` (`TestVsumMechanism`), log `vsum-mechanism-2026-09-20.log`. nobara-pc, RTX 2070 SUPER,
driver `595.91.07`, D7 = qwen2.5-7b-instruct-q4_k_m, K=8000, prompt set A, 10 prompts x 64 positions, the SAME cached
Phase A f32/f64 reference rows the parked record used (`~/goinfer-logs/prefill-ref/D7-K8000-p*.bin`). One binary, one
session, arms run back to back per prompt, 53.7 min. Build: goinfer `dcc7978f` + the one-line `skVsumSplit >= 1` guard
change this harness needs (uncommitted at run time, committed with this record).

## Result

| arm | mean KL | KL / exact | hard flips | agree % | KL(exact‖arm) |
|---|---:|---:|---:|---:|---:|
| exact | 0.100480 | 1.0000 | 15 | 86.09 | — |
| S=1 | 0.100480 | 1.0000 | 15 | 86.09 | 0 (bit-identical, 0 rows differ) |
| S=2 | 0.098524 | 0.9805 | 16 | 85.94 | 0.0444 |
| S=4 | 0.099048 | **0.9857** | 17 | 86.41 | 0.0430 |
| S=8 | 0.100011 | 0.9953 | 14 | 85.94 | 0.0443 |
| S=16 | 0.100407 | 0.9993 | 17 | 85.31 | 0.0458 |

Paired per-prompt KL delta (arm − exact), mean ± s.e. over 10 prompts: S=2 −0.0020 ± 0.0020, **S=4 −0.0014 ± 0.0019**,
S=8 −0.0005 ± 0.0037, S=16 −0.0001 ± 0.0021; higher on 4–5 of 10 prompts at every S. Every spike arm differed from
exact on all 630 decode rows, so each ran (no vacuous arm).

## Against the pre-registered decision rule

1. **S=1 differs anywhere → DEFECT: not the case.** S=1 is bit-identical to exact on all 640 rows of all 10 prompts.
   H-defect (a defect in the partial/combine kernels' arithmetic or order) is **ruled out** for this geometry.
2. **S=4 reproduces within 1.0585 ± 0.02: not the case.** S=4 reads **0.9857x**, outside the band by 0.07. Rule 3 applies:
   *the original ratio was build- or session-dependent; report it and do not unpark on this run either.*
3. Consequently H-noise, H-tail and H-compounding are not tested against an effect, because **there is no KL excess on this
   build to explain**: the paired deltas are within about one standard error of zero at every S, are not positive, and
   are not ordered by S. Nothing here supports a tail-, compounding- or prompt-localised excess (the head/tail and
   position-bucket columns are in the log; the tail part is ~1e-4 of the total and changes sign, which is noise).

So the finding is **"the parked KL excess is not reproducible on the current build"**, not "the mechanism is X". The
spike remains **opt-in, unparked and uncleared**: by the pre-registration nothing in this run unparks it, and a promotion
still needs its own fidelity gate on held-out prompt set B.

## What this does NOT establish, and one thing it makes worse

- **Why the parked record read 1.0585x.** Not explained. The parked record's *exact* arm differs materially from this run's
  exact arm on the same references: hard flips **30 → 15**, mean agreement **83.91% → 86.09%** (KL 0.100720 → 0.100480, nearly
  the same). Its own text says its scoring binary was built before a four-times rebase that brought 169 Go-file changes
  (`cuda/resident.go`, `backend.go`, `prefill.go`, ...), and it warned that absolute figures are for that pre-rebase
  build. This run confirms the exact path itself moved by that much. I have NOT bisected which change did it, so "the
  rebase" is a candidate, not a finding; nor can I say whether the old spike arm carried a defect that has since been fixed
  or the old figure was simply a different draw of the same noise. The paired-delta s.e. here (~0.002 per arm) says a
  +0.0059 mean delta would be about 3 s.e. away from this run's result, so it is not naively "one prompt draw".
- **Fidelity clearance.** Same cell (prompt set A) as the record, reused deliberately for mechanism analysis; a clean
  result on the data the gate already saw is not a pass of the gate.
- **S at 3900 / the 1.5B, other geometries, K ≠ 8000.** Not run.
- Hard flips are near noise (14–17 of 640 per arm, ±√15 ≈ 4), so they do not separate any arm here either.

## Consequence for R6

The registered blocker on the spike ("KL 1.0585x, ambiguous") was the reason step 2 was framed as "the kernel must beat its
own first step *and* clear a fidelity gate the first step could not". On the current build that gate's motivating number is
gone, which makes the S=4 tree a credible starting point for `attn_decode_fa` — provided step 2's fidelity gate is
pre-registered on **held-out set B** with a fresh f32/f64 reference (Phase A, D7 at 8000: about 4 h of CPU, which can run
while the kernel is written). The old number must not be quoted as either a pass or a fail from now on.

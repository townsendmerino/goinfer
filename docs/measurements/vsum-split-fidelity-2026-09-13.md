# V-sum split spike — §3.2 fidelity gate (IN PROGRESS)

Pre-registration: `vsum-split-fidelity-PREREGISTERED.md`, committed `a56a7306` before Phase A was
launched. Gate: `cuda/vsum_split_gate_test.go` (`ed2624bd`).

**Status: S (confirmation) scored — DOES NOT PASS, on criterion (a) by one hard flip. D7 (decision)
reference still generating; no D7 result exists. The gate's verdict is D7's, and is not in yet.**

## S — confirmation cell — DOES NOT PASS

qwen2.5-coder-1.5b-instruct-q4_k_m, nH=12 nKV=2 hd=128, K=8000, prompt set A, S=4. Reference: CPU,
f32 weights and activations, exact f64 attention, 1h39m. Phase B log
`goinfer-logs/vsum-fidelity-phaseB-S-20260913-121657.log`, 5m39s.

| S, 10 prompts x 64 positions | exact | spike | pre-registered criterion |
|---|---:|---:|---|
| hard flips vs reference | 7/640 | 8/640 | **(a) spike <= exact — FAIL** |
| mean teacher-forced agreement | 94.84% | 94.06% | (b) >= exact − 1.0 pt AND >= half: −0.78 pt, 6/10 — pass (outside the 0.2 pt parked band by 0.02) |
| mean KL(reference ‖ arm) | 0.042577 | 0.042689 | (c) <= 1.10x: **1.0026x** — pass |
| worst near-tie gap | 5.458% | 5.725% | — |

**Preconditions, all held:** A/A bit-identical over 64 rows; the prefill seed row identical between
arms on every prompt; **630/630 decode rows differing** — the spike was active on every position of
every prompt, so this is a measurement and not the exact path scored against itself.

Per prompt (agreement exact / spike, hard flips exact / spike, KL exact / spike):

| p | agree | HF | KL |
|---:|---|---|---|
| 1 | 96.9 / 96.9 | 1 / 1 | 0.06323 / 0.06229 |
| 2 | 93.8 / 95.3 | 2 / 1 | 0.03020 / 0.02961 |
| 3 | 95.3 / 93.8 | 0 / 1 | 0.02241 / 0.02375 |
| 4 | 100.0 / 100.0 | 0 / 0 | 0.00776 / 0.00778 |
| 5 | 96.9 / 96.9 | 0 / 1 | 0.06743 / 0.06947 |
| 6 | 85.9 / 84.4 | 1 / 1 | 0.08363 / 0.08829 |
| 7 | 89.1 / 84.4 | 1 / 1 | 0.05702 / 0.05598 |
| 8 | 95.3 / 93.8 | 1 / 1 | 0.04535 / 0.04423 |
| 9 | 100.0 / 100.0 | 0 / 0 | 0.02387 / 0.02357 |
| 10 | 95.3 / 95.3 | 1 / 1 | 0.02487 / 0.02191 |

The verdict stands as registered. S is the confirmation cell: it cannot decide the gate, and it
cannot be rescued.

### Two statements recorded BEFORE the D7 result, so neither can read as a rescue of it

**1. A labelling error in the pre-registration — and the rule is NOT being switched.** The
pre-registration is titled a "§3.2" gate, but criterion (a) was written from **§3**'s strict form,
`spike HF <= exact HF`. The Metal §3.2 pooled gate (`prefill-gate-l1-ref-b-2026-09-09.md`) uses a
noise-aware ceiling, `exact + 2√exact`; for S that is 7 + 5.3 = 12.3, which 8 would clear. **It is
not adopted, for S or for D7.** The committed rule text is unambiguous, and moving to the friendlier
bar after watching a cell fail the strict one is precisely what pre-registration exists to prevent.
The mislabel is mine and is recorded as an error, not as a loophole.

**2. The registered criterion (a) has almost no power at these counts.** 7 against 8 hard flips is
well inside Poisson noise (σ ≈ √7 ≈ 2.6). D7 will likely sit in the teens per arm — its K=1024
prefill cell was 16 v 16 — so D7's (a) outcome will be largely set by noise whether the spike is
harmless or mildly harmful. The metric that *can* resolve a difference here is KL, and on S the arms
are indistinguishable at 1.0026x. **This is a statement about the gate's resolving power, written
while D7's outcome is unknown.** It does not change D7's rule, and if D7 fails (a) this paragraph is
not a reason to overlook that; it is the reason a follow-up with a noise-aware rule would have to be
pre-registered on NEW data (held-out prompt set B), not applied to this run.

**An observation, not a finding:** on the prompts where spike trails exact on agreement without
losing a hard flip — 7 and 8 — its KL is *lower*. Those argmax disagreements sit at positions the
reference itself scores as near-ties; the arms land on opposite sides while their distributions stay
equally close. Two prompts is not evidence of a mechanism.

## Deviations from the pre-registration

Every one of these was decided **before any reference file for K=8000 existed**, so none can have
been steered by a result. Each is stated with what it was, why, and what it can and cannot affect.

### D1 — D7's reference weights: f32 → int8 (decided 2026-09-13 11:24, Phase A at 48 min, S cell unfinished)

The pre-registration named f32 reference weights for D7 (`GOINFER_CPU_REF_QUANT_D7=""`), to match
this box's existing D7 cells. It was changed to int8 weight-only (`GOINFER_CPU_REF_QUANT_D7=int8`)
after the run itself showed the pricing was wrong:

- **The run was slower than projected**, which is what exposed it. S at K=8000 had not finished one
  prompt after 45 minutes. Scaled from this box's own Sep-5 cells — S K=1024 3.5 min to first batch,
  K=3900 18.5 min, i.e. 5.3x the cost for 3.8x the tokens, attention's K² showing — D7 at K=8000
  projects to 4-5.5 h, not the ~3 h quoted when the run was launched.
- **The memory was never priced before launch, and the fit guard was bypassed.** S at K=8000 sat at
  16.5 GB RSS, ~1.3 GB per worker over its weights. Scaled to D7's wider layers plus 28 GB of f32
  weights, that lands near ~50 GB against 58 GB available — close enough that an OOM hours into the
  D7 cell was a live risk, and an OOM kill of a detached process writes nothing to its log.

**Why int8 is a valid reference for this gate**, by the harness's own reasoning
(`decoder/prefill_ref_gen_test.go`, `d7RefQuant`): the reference must have f32 **activations**,
because activations are the axis the arms could differ from it on. Reference weight precision is
**common-mode**: both GPU arms load identical int4 weights and differ only in the V-sum reduction
tree. int8 moves where the reference sits, equally for both arms; it cannot favour either.

**What it can affect:** absolute agreement / KL figures for D7 are not comparable to the Sep-5 f32
D7 cells. **What it cannot affect:** the pre-registered criteria, which are all exact-vs-spike
comparisons against the same reference.

**Mechanics.** The S cell was left to finish at f32 (as registered — S's f32 weights are 6 GB and
were never at risk). A detached watcher (`goinfer-logs/vsum-fidelity-handoff.sh`, log alongside)
waited for the D7 subtest to start, verified all ten `S-K8000-p*.bin` were on disk, stopped the f32
run, and relaunched `-run TestPrefillGateReference/D7` at int8.

## Separately recorded, not a deviation

The served A/B that the spike record had called unestablished was re-run and archived before Phase
A launched — `vsum-split-spike-2026-09-13.md` §"What this does NOT establish" item 2. It is a speed
measurement and bears on nothing in this gate.

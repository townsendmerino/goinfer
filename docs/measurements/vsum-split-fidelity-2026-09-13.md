# V-sum split spike — §3.2 fidelity gate (IN PROGRESS)

Pre-registration: `vsum-split-fidelity-PREREGISTERED.md`, committed `a56a7306` before Phase A was
launched. Gate: `cuda/vsum_split_gate_test.go` (`ed2624bd`).

**Status: Phase A running. No result has been seen by anyone. Nothing below is a verdict.**

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

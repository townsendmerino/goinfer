# WG3-coh: where Command-R7B's position-13 gap opens (registered 2026-11-24)

**Status: REGISTERED 2026-11-24 10:30 PST, before the run.** Francis's call (2026-11-24 ~10:05, via the coordinator):
option (a) of the Command-R7B decision; #1255 stays unarmed until this reports.

**Question.** On the real Command-R7B at int4, `TestPairwiseRoPERealWebGPU` reads a worst per-position logit cosine of
0.860 against the CPU, at prompt position 13 of README.md's tokens. The CPU against itself with ±1-ULP norm noise
(`TestCPUQuantSensitivityREADME`) reads 0.948670 there, its own worst. Which sublayer opens WebGPU's gap at position 13?

**Instrument.** `TestResidentCaptureParityWebGPU` with `GOINFER_GPU_CAPTURE=1`, `GOINFER_PARITY_README=20` and
`GOINFER_PARITY_QUANT=int4`, on the real checkpoint in nobara's local model store (the same one the gate read), at the WG3-coh
head. For every position 0-19 and layer l it prints the relL2 between the CPU's and the resident's sublayers. Each
layer starts from the GPU's own input residual, so three columns are that layer's own error: `ctx` (attention context,
which also reads each side's own earlier K/V), `attn-contrib`, and `post-attn` (the MLP's contribution). One run; tier 1,
exclusive lock; stated start and end first. No bar is moved by it.

**Outcomes**, read in this order (for a column s and layer l, "baseline" is the max of rel(l, s) over positions 0-12):
1. **Not reproduced:** position 13's logit cosine is ≥ 0.95 in this run. Report that, and treat it as outcome 4.
2. **A WebGPU bug in a sublayer:** some (l, s) at position 13 has rel ≥ 5× its baseline and ≥ 5× the median of rel(·, s)
   over the layers at position 13. Fix that sublayer before #1255 is armed.
3. **Activation-quant error amplified at a sensitive position:** no (l, s) meets 2, and every own column at position 13
   is ≤ 2× its baseline, while the logit cosine still falls there. Position 13's own errors are ordinary, and the
   checkpoint amplifies them (the CPU shows the same position as its most sensitive). This goes to Francis with the
   numbers, recommending a recorded acceptance and arming.
4. **No clear cause:** anything else. Parked, to Francis with the numbers.

The log is kept beside this file as `capture-pos13.log.gz`, and the outcome goes in `README.md` here.

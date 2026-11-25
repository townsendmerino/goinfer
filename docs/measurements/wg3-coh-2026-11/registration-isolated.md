# WG3-coh: the layer-isolated capture at position 13 (registered 2026-11-24)

**Status: REGISTERED 2026-11-24 15:05 PST, before the run.** Francis's call (2026-11-24 ~14:30, via the coordinator), after
[the first capture](README.md) read outcome 4 (no clear cause): run each CPU layer on the resident's own input, so each
layer's difference is its own. #1255 stays unarmed until this reports.

**Instrument.** `TestResidentLayerIsolatedWebGPU` (`GOINFER_PARITY_README=20`, int4, the resident at the gate's context of
512) on the same real Command-R7B in nobara's local model store, at the WG3-coh head. At every position the resident runs
first; the CPU then runs layer l on the resident's input to it (`KVCache.SetLayerInputsForTest`), so its K/V rows come from
the same inputs. Per (position, layer): relL2 of the attention context, the attention contribution, the rest of the
layer, and the whole layer's contribution (`layer`); and the logit cosine of the CPU head on the resident's last-layer
input. One run; tier 1, exclusive lock; start and end stated first. Validated on cohere2-tiny (mbp2 Iris): every column
about 1e-7 clean, 0.05-0.18 in every layer with WebGPU's LayerNorm planted as RMSNorm.

**Outcomes**, on the `layer` column (baseline: the max over positions 0-12 of the same layer):
1. **A WebGPU defect in a layer:** some layer at position 13 has `layer` rel ≥ 5× its baseline and ≥ 5× the median over
   layers at position 13. The defect is that layer's (its sub-columns say which part): fix before #1255 is armed.
2. **Rounding amplified downstream:** every layer at position 13 is ≤ 2× its baseline. Each layer's own error is
   ordinary there, so the position-13 gap is ordinary per-layer int4/int8 rounding compounded through the stack at a
   token the checkpoint is sensitive to (the CPU's own floor is lowest there). This goes to Francis with the numbers,
   recommending a recorded acceptance and arming.
3. **No clear cause:** anything else. Parked, to Francis with the numbers.

The log is kept beside this file as `isolated-pos13.log.gz`, and the outcome goes in `README.md`.

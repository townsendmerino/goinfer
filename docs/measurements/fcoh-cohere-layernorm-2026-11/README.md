# F-coh: CUDA's Cohere LayerNorm, one kernel for decode and the batched paths (nobara-sb01, registered 2026-11-08 20:43 PST, before any code)

**The finding** (`docs/completed/task-forward-architecture-2026-10.md` §4.5, backend findings; the queue row F-coh in
`docs/tasks/task-forward-and-memory-2026-11.md`).
- On CUDA a Cohere step row is not bit-identical to `Forward`: `cohere-tiny` and `cohere2-tiny` differ by one ulp in a logit.
- **Why:** decode's `layernorm_quant` (`backend/cuda/glue.cu`) rounds `((x-mean)*w)*inv`, while the batched `layernorm_quant_batched`
  (`backend/cuda/layernorm_quant.cu`) rounds `((x-mean)*inv)*w`. A normed value can then differ in its last bit and move an int8 code.
- **Inherited by:** the step, `PrefillLastN` and speculative verify on Cohere.

**The call, as CUDA's owner: match.** The kernel to keep is the batched one, for three reasons.
- Its order is the CPU reference's: `decoder.layerNormInto` computes `float32((x-mean)*inv)*w + b`.
- It is the SigLIP tower's kernel too, so keeping it moves nothing in the towers.
- Apart from the multiply order, both kernels are the same: 256 threads, the same mean and variance reduction trees, and the same
  `rsqrtf`.

**The change:**
- Decode's `cudaResident.layerNormQuant` runs `layernorm_quant_batched` at M = 1, through `bLayerNormQuantB`.
- Where the batched kernel did not load, `bLayerNormQuantB` already falls back to `layernorm_quant` row by row, so decode and the
  batched paths still share one kernel.
- No PTX changes. `glue.ptx`, an audited artifact at NVRTC 12.6.85, is untouched.
- **What moves:** Cohere's CUDA decode outputs, where a normed value's last bit moved an int8 code. Nothing else, since no other text
  family uses LayerNorm on this backend.

## Gates (tier 0; card tests under the shared timing lock)

- **G-coh-1, the identity this buys.** `TestCUDAStepBatch_matchesForward` gains `cohere-tiny` and `cohere2-tiny`: a step row's logits
  equal `Forward`'s bit for bit.
  - **Red first:** both fixtures fail on `main` (the one-ulp logit). Green with the change.
- **G-coh-2, no farther from the CPU.** On `cohere-tiny` and `cohere2-tiny`, decode's teacher-forced logits are compared with the CPU
  f32 path over 16 positions, before and after.
  - **Bar:** after the change, the worst cosine is ≥ the old one minus 1e-6, and the existing Cohere CUDA tests pass.
  - **Expectation:** equal or closer, since the order now matches the CPU's.
- **Plant:** `layerNormQuant` back on `layernorm_quant` makes G-coh-1 red.
- **Ambiguous band:** G-coh-2 between −1e-6 and −1e-4 → parked, to Francis with the numbers. Under −1e-4 → FAIL, and the change is
  dropped in favour of documenting the ulp.

**Not gated here:** a real Command-R checkpoint. The change is a reduction-order alignment toward the reference, not a new
approximation, so tier 0 answers it.

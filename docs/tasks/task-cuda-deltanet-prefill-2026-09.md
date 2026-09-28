# Task: batched prompt prefill for Gated-DeltaNet models on CUDA — 2026-09

> **Status: OPENED 2026-09-28; pre-registered here before any code.** Owner: "really i never want to see a 6 hour
> run" — decisions D6a on Qwen3.5-9B estimated ~6 h on CUDA, because every prompt token ran the full per-token
> decode path.

## Why

CUDA's batched prefill declines every Gated-DeltaNet model (`cuda/prefill.go`: "recurrent state advances one token
at a time and cannot be batched"), so qwen3_5 / qwen3_5_moe / qwen3_next prompts run through decode, one token at a
time.
- **Measured 2026-09-28 on the RTX 2070 SUPER, Qwen3.5-9B Q4_K_M:** ~14 ms per prompt token (561 tokens in 8.0 s).
- **The cost is the weights, not the recurrence.** Each token re-reads every projection's weights.
  - The recurrence (conv ring, gates, q/k norm, delta rule, gated norm) is five small kernels per layer per token.
  - Only the recurrence has to be sequential.

## Design

Inside `prefillCore`'s layer loop, a layer with `Ly.isDeltaNet` runs:
1. **Batched over the M prompt rows** (the existing batched kernels): the pre-norm + quant, and the four input
   projections `dnQKV`, `dnB`, `dnA`, `dnZ`, each through `bGemvB`.
2. **Row by row, in order, through decode's own launches** (`deltaNetMixer`'s five kernels), on views of the batched
   buffers (`Buffer.At`): `delta_conv` (ring), `delta_gates`, `delta_norm`, `delta_rule` (state), `delta_gnorm`.
   Each row's gated output lands in its row of a batched buffer. The layer state advances exactly as M decode steps
   would advance it.
3. **Batched again:** quantize the gated rows, then `dnOut` accumulating into the batched residual.
4. **The FFN** is the existing batched path. A MoE FFN (qwen3_5_moe / qwen3_next) takes the existing row-by-row MoE
   path.

The family's softmax layers (`qGate`: a double-width q_proj, with the context scaled by sigmoid(gate)) gain their
batched form:
- `delta_qsplit` as one launch over M·qDim elements;
- `delta_attn_gate` as one launch over M·qDim.

Both are elementwise per head, and M rows are M·nH contiguous heads, so each element computes exactly what decode
computes.

`nonBatchableKind` and the `r.dnet != nil` refusals (`prefill.go`; `prefillReady && r.dnet == nil` in
`resident.go`) are narrowed to what is still true.

## Gates (hard unless marked)

1. **Correctness against the per-token path**, on the same resident: last-prompt-token logits.
   - **Below 16 rows:** the batched GEMVs are `gemv_w4a8_rn`, decode's own kernel, and every recurrence launch is
     decode's. **Bit-identical, 0 differing logits**, on the tiny qwen3_5, qwen3_5_moe and qwen3_next fixtures,
     prompts of 3 / 8 / 15 tokens.
   - **At 16 rows and above:** the projections take the batched GEMM, which dense-model prefill already runs in
     production and which is not bit-identical to decode. The bar is **argmax equal and cosine ≥ 0.9999** on
     Qwen3.5-9B Q4_K_M at 145 / 561 / 621 tokens (the D6a resident-check rows and a long one), and on the tiny
     fixtures.
   - **State continuity:** after prefill, 16 greedy decode tokens equal the per-token path's on the 9B at 561
     tokens, and a second prefill chunk continues the state (a prompt split 300 + 261 reads the same last-token
     argmax, cosine ≥ 0.9999).
2. **Suites:** the tagged CUDA suite (`cuda goinfer_testhooks`) and the heavy decode parity pass, plus gofmt, vet and
   the pinned staticcheck.
3. **Speed:** the 561-token prompt on the 9B, prefill wall time against the per-token path's 8.0 s. **Ships at ≥ 5×**
   (≤ 1.6 s). **Correct but < 5×:** reported, and the owner decides. Tokens per second are reported either way.

On passing, it ships as the default: it is what CUDA prefill already does for every other family.

# Spike (goinfer/aikit): TurboQuant-class 3-bit quantization — KV first, weights maybe

> **Status: SPIKE, not a commitment.** The roadmap already watches TurboQuant
> (llama.cpp #20969) for weights. This doc scopes the *cheaper, higher-fit*
> application first — the KV cache — and defines the evidence bar before any
> kernel work. Timebox: 2–3 days reading + a CPU prototype gate.

## Why KV before weights

- **Weights:** goinfer already ships int4/W4A8 + broad K-quant dequant. A
  3-bit weight format competes against a mature, parity-gated int4 path for
  ~25% size — meaningful for the embed-demo size/quality curve, but it buys
  no decode speed (W4A8 is ALU-bound) and adds a permanent format branch.
  Weak ROI until the demo-size complaint is real.
- **KV:** the published claim is the interesting one — **3-bit KV with ~zero
  measured accuracy loss** (and 8x-faster attention logits on H100, which is
  hardware-irrelevant here, but the *accuracy* result transfers). If it
  holds on CPU-implementable math, it slots into the exact seam
  `task-cpu-kv-quant.md` Increment 4 reserves (int4 territory), beating
  int4-with-KIVI on both size and quality — 10.7× vs f32 before scale
  overhead.

## What the spike answers

1. **Is the transform CPU-cheap?** TurboQuant-family methods leans on a
   randomized rotation/Hadamard-style decorrelation before low-bit rounding.
   Per-append cost must be O(kvDim·log headDim) or better with SIMD-friendly
   structure (fixed Hadamard on 64/128-dim heads is — fused into the
   RoPE/QK-norm pass it may be near-free). If the published method needs
   anything data-dependent per token beyond that, the answer is no.
2. **Does the dot stay integer?** The win requires scoring without
   dequantizing the history (the int8 story's structure): rotated-then-
   quantized K must admit `DotI8`-shaped scoring against a rotated query.
   Values side likewise for the weighted sum.
3. **Does accuracy hold at 8k+ keys on a real checkpoint?** Reuse the
   kv-quant gate verbatim (argmax + 3% near-tie + cosine, 1.5B + Gemma E2B).
   Bar to *replace* the planned int4 increment: ≥ int8's measured cosine at
   3 bits, or ≥0.995 — otherwise it's a paper result that didn't transfer
   and the spike closes with a written negative.

## Deliverable

A short findings note appended here + go/no-go: **go** ⇒ it becomes
`task-cpu-kv-quant.md` Increment 4 (replacing plain int4), with the aikit
kernel additions (rotation + 3-bit pack/dot) scoped then; **no-go** ⇒ the
watch item closes for KV and stays weights-only-if-triggered.

## Priority

Below kv-quant Inc 1–3, ring eviction, and GPU int8 — those are shipped
kernels + known math. This is the only research-risk item in the memory
program; spike it when the proven rungs are in.

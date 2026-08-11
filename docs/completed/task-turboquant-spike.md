# Spike (goinfer/aikit): TurboQuant-class 3-bit quantization — KV first, weights maybe

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


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
  `completed/task-cpu-kv-quant.md` Increment 4 reserves (int4 territory), beating
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
`completed/task-cpu-kv-quant.md` Increment 4 (replacing plain int4), with the aikit
kernel additions (rotation + 3-bit pack/dot) scoped then; **no-go** ⇒ the
watch item closes for KV and stays weights-only-if-triggered.

## Findings + verdict (2026-06-14): **NO-GO**

Throwaway recon spike (decoder test, since removed): real qwen2.5-coder-1.5B,
512-token prefill, post-RoPE K/V, per-head relative-Frobenius reconstruction error
over 3584 head-vectors (headDim 128). K-recon error ∝ score error averaged over
query directions; V-recon bounds the weighted-sum output — the principled cheap
screen (the metric the #6 weights spike used). The rotation tested is the
**CPU-cheap foldable** one the pitch promises: a fixed random-sign Walsh–Hadamard
per head (QuaRot incoherence transform; orthogonal ⇒ cancels in the q·k dot, so the
integer-dot story survives — spike Q2 OK).

| | int8 | int3 plain | int3 + rand-sign WHT |
|---|---|---|---|
| **K** | 0.0116 | 0.4398 | **0.2593** (22.4× int8) |
| **V** | 0.0087 | 0.3570 | **0.2666** (30.6× int8) |

Rotation helps 3-bit by **41% (K) / 25% (V)** — real, but nowhere near enough. The
bar to replace the int4 increment was *≥ int8's cosine, or ≥0.995*; int3+rot recon
~0.26 vs int8 ~0.01 misses by 22–30×. Cross-check that validates the measurement:
int3-plain (0.44) ≈ **42× int8 (0.0116) — exactly the level-count ratio 127/3**, so
the gap is fundamental quantizer coarseness, not an implementation artifact; the
rotation can't overcome a 42× level deficit.

The published "near-zero-loss 3-bit KV" does **not transfer** to a CPU-cheap
foldable rotation. Closing the gap would need a data-dependent / per-channel scheme
(full online rotation, per-channel-key KIVI split, asymmetric quant) — exactly the
online cost the foldable-Hadamard premise (spike Q1) was meant to avoid. Mirrors
**#6** (foldable rotation too weak for 3-bit *weights*; here a full per-head H is
stronger — 41% vs #6's 7% — yet still far short for *KV*).

**Verdict: NO-GO.** The KV watch item closes. int4 KV (task-cpu-kv-quant.md Inc 4)
stays DEFERRED behind the already-shipped 20×-stacked int8 (sufficient, no demand).
TurboQuant stays a weights-only-if-triggered watch item. This was the last
research-risk item in the v0.6 KV-memory program — the program is now fully resolved.

## Priority

Below kv-quant Inc 1–3, ring eviction, and GPU int8 — those are shipped
kernels + known math. This is the only research-risk item in the memory
program; spike it when the proven rungs are in.

# Metal W4A8 GEMV — optimization ask (for a fresh pass, e.g. Fable)

> **⚠ Peer numbers below predate the Ollama v0.32.5 re-anchor (2026-08-04).** Competitive figures
> in this doc (e.g. Ollama-CUDA ~149, Ollama-Metal 83.3, llama.cpp-CUDA 72.8, and any "×Ollama"
> multiple) were measured against **Ollama 0.5.7 (2025-01) / Ollama-Metal 0.32.0 / llama.cpp as of
> v0.5.0** — historical working records, not current claims. Current same-box numbers vs Ollama
> **v0.32.5** are in `docs/benchmarks.md` §B2 (CUDA) / §B3 (Metal).


You are an expert Apple-Silicon Metal / MSL kernel engineer. I have a working, correct
decode kernel set for pure-Go LLM inference and I've hit an efficiency wall on the two
dominant GEMV kernels. I want your best structural rewrite of the W4A8 GEMV, plus any
attention/pipeline ideas. **Correctness is non-negotiable** (constraints below). Give me
concrete MSL, not hand-waving, and explain *why* each change removes a specific bottleneck.

## The system in one paragraph

`goinfer` is a cgo-free (`CGO_ENABLED=0`) LLM inference engine written in pure Go. The Metal
backend talks to the GPU with **zero C** — via `purego`/`objc` (dlopen + `objc_msgSend`), not
Metal-cpp and not cgo. Kernels are MSL source compiled at runtime through
`newLibraryWithSource:options:error:` at **MSL 3.1**. It decodes a real model
(qwen2.5-coder-1.5b, int8 weights re-quantized to int4/W4A8 on load) one token at a time,
one Metal command buffer per token (~337 dispatches/token: 12 per layer × 28 layers + final).
Apple UMA means buffers are `newBufferWithBytes` shared — no host/device copy.

## Where we are

- **58.8 tok/s** decode (best-of-40 warm), qwen2.5-coder-1.5b, int8→W4A8, on Apple Silicon.
- Correct: 21/24 argmax matches vs a CPU reference decode; last-logit cosine 0.989.
- Target ("GO bar"): **~71 tok/s** = 85% of the measured Ollama-Metal peer (83.3 tok/s, int4).
- We got here by profiling: fixed attention (one-thread-per-head → threadgroup-per-head,
  1139µs→23µs) and coalesced the GEMV (stride-4 → word-per-lane). Both were real wins.

## The wall (per-kernel profile, µs/dispatch at real dims)

```
qkv gemv     (2048 × K1536)      28.8    ← fused QKV, uses _bias
gate/up gemv (17920 × K1536)    220.5    ← 2*I=17920 rows; 28× per token = 42% of the token
down gemv    (1536 × K8960)     117.7    ← uses _resid
o gemv       (1536 × K1536)      24.0    ← uses _resid
lm head gemv (151936 × K1536)  1813.5    ← V rows; once per token = ~12% of the token
rmsnorm_quant (H1536)            10.7
swiglu_quant  (I8960)            22.0
attention (nH12, nKeys32)        22.8
---- per-token: 28×457µs + lm 1813µs ≈ 14.6 ms → ~68 tok/s estimate; 58.8 measured ----
```

The gap between 68 (kernel-sum estimate) and 58.8 (measured) is per-token command-buffer
encode + commit/wait overhead across ~337 dispatches issued from Go via `objc_msgSend`.

**The core problem:** gate/up and lm head both sit at **~3× the memory-bandwidth floor**.
gate/up moves 17920×1536 int4 = 13.7 MB; at ~200 GB/s that's ~68µs, we spend 220µs.
lm head moves ~116 MB → ~583µs floor, we spend 1813µs. Same ~3× ratio on both → this looks
**ALU-bound on int4 nibble unpacking**, not bandwidth-bound. That's the hypothesis to confirm
or break.

## The current kernel (this is the hot path — one simdgroup per output row)

Weights are pre-packed: for each row of K int4 weights, K/8 `uint` words (8 nibbles/word,
little-endian nibble order: element k is at bit `4*(k%8)` of word `k/8`), plus K/32 f32
**group scales** (group = 32 consecutive elements = 4 consecutive words). Nibble value is
stored as `q+8 ∈ [0,15]`, so the real signed weight is `(nibble-8)`. Activation is a single
per-vector int8 quantization: `aq[k] ∈ [-127,127]` with one f32 scale `asc[0]`.

```metal
// Launch: total threads = N*32, threadgroup = 32 (one simdgroup = one output row).
// gid = threadgroup_position_in_grid (row), lid = thread_index_in_threadgroup (lane 0..31).
#define W4A8_BODY \
    uint wpr = K/8u;                                      /* uint words per row */ \
    device const uint*  brow = bq  + (uint)gid*wpr; \
    device const float* srow = bsc + (uint)gid*(K/32u);  /* f32 group scales   */ \
    float acc = 0.0f; \
    for (uint wi = lid; wi < wpr; wi += 32u) {           /* lane l → word l, l+32,… (coalesced) */ \
        uint x = brow[wi]; device const char* a = aq + wi*8u; \
        int gi = (int((x)&0xF)-8)*int(a[0]) + (int((x>>4)&0xF)-8)*int(a[1]) \
               + (int((x>>8)&0xF)-8)*int(a[2]) + (int((x>>12)&0xF)-8)*int(a[3]) \
               + (int((x>>16)&0xF)-8)*int(a[4]) + (int((x>>20)&0xF)-8)*int(a[5]) \
               + (int((x>>24)&0xF)-8)*int(a[6]) + (int((x>>28)&0xF)-8)*int(a[7]); \
        acc += float(gi) * srow[wi>>2];                  /* 4 words share one group scale */ \
    } \
    acc = simd_sum(acc);

kernel void gemv_w4a8_coal(device const uint* bq[[buffer(0)]], device const float* bsc[[buffer(1)]],
    device const char* aq[[buffer(2)]], device const float* asc[[buffer(3)]], device float* out[[buffer(4)]],
    constant uint& K[[buffer(5)]], uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    W4A8_BODY
    if (lid == 0) out[gid] = acc * asc[0];
}
// _bias variant fuses a per-row bias add; _resid fuses += into out (residual). Same body.
```

Note the math trick that keeps this exact: within a 32-element group all 4 words share one
f32 scale, so summing `(nibble-8)*int8` per word as an **integer** `gi`, then doing one
`float(gi) * groupScale`, is bit-identical to per-element float MAC — but with int accumulate.
Activation scale `asc[0]` factors out to the very end. This parity is validated bit-for-bit
against a CPU reference (cosine > 0.99999, maxrel < 1e-4).

## Constraints (a rewrite that violates any of these is useless to me)

1. **Bit-exact-ish**: must preserve the group-scale/int-accumulate structure so argmax parity
   holds (currently 21/24). You may reassociate within a group (all share a scale) but not
   across groups with different scales in a way that changes rounding materially.
2. **MSL 3.1**, compiled from source at runtime. No Metal Performance Shaders (MPS) — I can't
   reach the ObjC MPS API cleanly from purego-objc, and I want to keep the kernel self-contained.
   (If you think an MPSMatrix/MPSGraph path is *the* answer, say so and explain the purego-objc
   surface it needs — but a pure-MSL win is strongly preferred.)
3. `CGO_ENABLED=0`. Kernel is pure MSL; only constraint that imposes is "no host-side C helper."
4. Weights are already packed as described (K/8 uints + K/32 f32 scales, nibble=q+8). I *can*
   change the packing layout on the Go side if a different layout unlocks the kernel — tell me
   exactly what layout you want (e.g. interleaved, transposed, f16 scales, 2×uint per iter).
5. GEMV only (batch = 1, single decode token). This is memory-latency territory, not big-GEMM.
   The weight matrix does NOT fit in cache; it's streamed once per token.
6. Dispatch is one command buffer per token. Fewer/bigger dispatches are fine; I already fused
   epilogues (bias, residual). Fusing gate+up+swiglu into one dispatch is on the table.

## Specific questions

1. **Is this actually ALU-bound on unpacking, or occupancy/latency-bound?** One simdgroup per
   row = 17920 simdgroups for gate/up, each doing only `192/32 = 6` word-iterations then a
   `simd_sum`. Is the reduction + tiny per-lane work starving the ALU? Would **N rows per
   simdgroup** (more sequential work per lane, fewer reductions) or **multiple simdgroups per
   row** (split-K) help? Which, and why?
2. **Vectorized loads**: should I load `uint4` (128-bit) per lane instead of `uint`, unpack 32
   nibbles at once? Does that change coalescing math? Show the indexing.
3. **char4/packed int8 activation loads** and `dot`-style int fused ops — is there an int8 SIMD
   dot on Apple GPUs I should use (e.g. `dot` on packed types, or reduce via `simd_shuffle`)?
4. **f16 group scales** instead of f32 — halves the scale traffic (K/32 f32 → f16). Worth it
   for parity risk? gate/up scales are 17920×48×4B = 3.4MB, ~20% of the 13.7MB — non-trivial.
5. **lm head specifically**: I only need the **argmax** logit for greedy decode, not all 151936
   values. Can the kernel do a fused block-argmax reduction (each threadgroup emits (maxVal,
   maxIdx), a tiny second pass finds the global max) so I skip both computing full precision I
   don't use AND reading back 608KB of logits? What's the cleanest MSL for that?
6. **Threadgroup-staged activation reuse**: activation `aq` (K int8 = 1.5KB) is re-read from
   device by all 17920 rows. Stage it into threadgroup memory once per threadgroup (shared by
   the rows that threadgroup handles)? Payoff estimate?

## What I want back

- A rewritten `W4A8_BODY` (or a replacement kernel) with the dispatch geometry that goes with
  it, in compilable MSL 3.1.
- If it needs a repacking change, the exact new layout (I'll implement the Go packer).
- A one-line-per-change rationale tying each to a specific bottleneck above.
- Your estimate of the resulting µs/dispatch for gate/up (220µs today) and lm head (1813µs).
- The fused-argmax lm-head kernel if you think it's worth it.

I'll measure whatever you give me against the real model (parity + best-of-40 tok/s) and report
back numbers. Optimize for *measured decode tok/s*, and be honest about which ideas are
speculative vs. certain.

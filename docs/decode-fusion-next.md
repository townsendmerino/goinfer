# Decode-perf via kernel fusion — status & next steps

The decode path's throughput ceiling on WebGPU is the long, serially-dependent
dispatch chain (~535 dispatches/token: ~197 GEMV + ~338 glue), where per-dispatch
barrier cost dominates over kernel work. Cutting dispatch/barrier count is the lever;
it helps cogentcore today and, because per-dispatch overhead is exactly wgpu-native
v29's tax, it also shrinks the v29 decode penalty (see `perf-dot4-report.md`).

## Done — Increment 1 (shipped, `e8add57`)

Fused `rope(q) + rope-store(k) + store(v)` → one `qkvFinalize` dispatch (f32 KV;
f16/int8 KV keep their packed-store kernels). Bit-exact (full parity suite green).
**−56 dispatches/token; decode 89.8 → 91.1 tok/s (+1.5%)** on the synthetic 1.5B-int8
bench, identical across runs.

### What it taught us (recalibration)
The gain was smaller than the naive ceiling because **q/k/v-finalize are INDEPENDENT**
(different buffers) — the GPU already overlapped them, so fusing saved per-dispatch
*launch/record* cost, not *barrier* cost. The expensive barriers are between
**dependent** dispatches. So the rule for future increments: **fuse dependent links,
not independent ones.**

## The real-model bench (shipped, `901b8aa`)

`TestDecodeRealModel_throughput` loads a real GGUF model GPU-resident (int8) and
measures steady-state inter-token rate (best-of-rounds, prefill excluded). Defaults to
Qwen2.5-coder-1.5B; `GOINFER_DECODE_GGUF` overrides. This is the gate the family-specific
folds (2 & 3) needed: the synthetic `TestDecodeToken_throughput` dispatches no bias /
qk-norm, so it can't see them. **Baseline (Increment 1 in): 102.2 tok/s.**

## Done — Increment 2 (shipped, `38babc4`)

q/k/v bias → GEMV epilogue: `gemvW8A8Bias` adds a 7th binding and computes
`dst[n] = scale·acc + bias[n]`, deleting the three standalone `biasAdd` dispatches/layer
(84/token on the 28-layer 1.5B) for **Qwen2/Qwen2.5** bias models. A *dependent*-chain
fusion (biasAdd reads the GEMV output) → removes barrier cost. W8A8 only; W4A8+bias falls
back to gemv+biasAdd. Bit-exact: `TestResidentForwardN_parity` (Qwen2.5-coder-0.5B)
cosine=1.0, maxAbsDiff=0. **Measured: 102.2 → 104.5 tok/s (+2.3%)** — beats Increment 1's
+1.5%, confirming dependent folds pay more than independent ones.

## Deferred — Increment 3 (no fitting model on this box)

- **Increment 3 — QK-norm → fold into rope/GEMV.** `qkNorm` depends on the GEMV output
  and precedes rope; benefits **Qwen3 / GLM / Mellum**. Same dependent-fold shape as
  Increment 2, so it should pay similarly. **Blocked on measurement, not design:** the
  only qk-norm models here are Qwen3-35B-A3B MoE (won't fit 8 GB) and GLM-4.5-Air (106B);
  no small dense qk-norm model is available to run the real-model bench against. Implement
  + measure when a Qwen3-dense (e.g. 1.7B/4B) or Mellum GGUF lands on a box that fits it.

## Hard ceiling

WGSL cannot fuse across the **matmuls** (workgroup-per-output geometry ≠ the
elementwise glue geometry) — the "single megakernel per layer" that CUDA/Metal can
express. So decode-fusion headroom is bounded to the glue/elementwise chain; the GEMV
stream (~4.3 ms/token, ~98% of the bandwidth roofline) is already optimal and unfusable
further here.

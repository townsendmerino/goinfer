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

## Deferred — Increments 2 & 3 (gated on a real-model decode bench)

Both are *dependent*-chain fusions (bigger per-dispatch payoff than Increment 1), but
they only fire for specific families and **the synthetic throughput bench
(`TestDecodeToken_throughput`) has neither**, so they can't be measured there:

- **Increment 2 — q/k/v bias → GEMV epilogue.** The `biasAdd` dispatch depends on the
  q/k/v GEMV output; fold it into the GEMV epilogue the way `gemvAdd` folds the
  residual. Removes a barrier-bound dispatch ×3/layer. Benefits **Qwen2/Qwen2.5**
  (which have qkv bias); no effect on bias-free models.
- **Increment 3 — QK-norm → fold into rope/GEMV.** `qkNorm` depends on the GEMV output
  and precedes rope. Benefits **Qwen3 / GLM / Mellum**.

**Decision (do not start until there is a real-model decode bench):** these are gated
on a tok/s benchmark that loads an actual bias / qk-norm model (e.g. Qwen2.5-1.5B GGUF,
Qwen3-1.7B), so the family-specific saving is *measured*, not assumed. Each stays
bit-exact-gated by the existing parity tests. Until that bench exists, they are
optional family-specific polish, not a headline lever.

## Hard ceiling

WGSL cannot fuse across the **matmuls** (workgroup-per-output geometry ≠ the
elementwise glue geometry) — the "single megakernel per layer" that CUDA/Metal can
express. So decode-fusion headroom is bounded to the glue/elementwise chain; the GEMV
stream (~4.3 ms/token, ~98% of the bandwidth roofline) is already optimal and unfusable
further here.

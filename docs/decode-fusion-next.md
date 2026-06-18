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

## Tried + rejected — Increment 3 (QK-norm fold REGRESSES, −2.7%)

Measured on Qwen3-1.7B-Q8_0 (downloaded for exactly this; resident int8, 16 q / 8 kv
heads, hd 128). Built `qkNormFinalize`: one **workgroup-per-head** dispatch doing
per-head qk-norm + rope(q) + rope-store(k) + store(v), replacing the two `qkNorm`
dispatches *and* `qkvFinalize`. Bit-exact (`TestResidentForwardN_parity` cosine=1.0,
maxAbsDiff=0; qknorm-vs-CPU parity green). But decode went **104.6 → 101.8 tok/s
(−2.7%)**, consistent across runs — so it was reverted (commit never landed).

**Why it loses (the lesson the bench bought):** qk-norm is a per-head reduction over
headDim, so it *must* be workgroup-per-head — only nH=16 workgroups on a 40-SM card,
low occupancy. rope/store want the opposite: a flat, high-occupancy grid (that's what
`qkvFinalize` is). Fusing them drags rope into the low-occupancy per-head geometry AND
serializes two reductions (q then k) behind barriers inside each workgroup. The
dispatch-count cut (3→1) didn't pay for the occupancy/serialization loss. This is the
mirror image of Increment 2, which folded into the GEMV — already workgroup-per-output
high-occupancy, with a trivial `+ bias[n]` epilogue.

**Corollary to the Increment-1 rule:** "fuse dependent links" holds *only if the fused
kernel keeps the better launch geometry*. Folding a reduction-shaped op (qk-norm,
softmax, any cross-element reduce) into a flat elementwise kernel — or vice-versa —
trades dispatch count for occupancy and can net out negative. Dispatch-count reduction
is necessary, not sufficient. The real-model bench is what caught this; the
"dependent-fold = win" heuristic alone would have shipped a regression.

GLM-4.5 / Mellum share the qk-norm geometry, so the same fold would regress them too —
not worth retrying without a fundamentally different approach (e.g. a higher-occupancy
norm that doesn't need per-head workgroups).

## Hard ceiling

WGSL cannot fuse across the **matmuls** (workgroup-per-output geometry ≠ the
elementwise glue geometry) — the "single megakernel per layer" that CUDA/Metal can
express. So decode-fusion headroom is bounded to the glue/elementwise chain; the GEMV
stream (~4.3 ms/token, ~98% of the bandwidth roofline) is already optimal and unfusable
further here.

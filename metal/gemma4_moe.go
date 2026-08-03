//go:build darwin

package metal

// Gemma-4 enable_moe_block (26B-A4B) MSL kernels — the parallel dense‖MoE FFN that the generic
// moe.go path (Mixtral/Qwen/GLM shape) cannot express. Kept in their OWN file/const, concatenated
// after moeKernels, so moe.go's audited MoE kernels are not touched — the same discipline as CUDA's
// separate router_f32.cu (which exists so adding a kernel doesn't rewrite the audited moe.ptx).
//
// This step (9c Step 5, router-first) adds ONLY gemv_f32_f32 — the router's pure-f32 projection.
// The remaining gemma4-MoE kernels (weightless out-of-place norm, per-expert-scale fold, per-layer
// scale) land with the full dense‖MoE forward, after the router idx-parity gate proves selection.
const gemma4MoeKernels = `
// gemv_f32_f32: pure-f32 GEMV — f32 weight [N,K] × f32 activation [K] → out[N], one simdgroup (32
// lanes) per output row. This is the Gemma-4 MoE ROUTER projection, and it quantizes NOTHING on
// purpose. The router is the one DISCRETE-failure path in the whole MoE delta: a quant error near a
// top-k tie picks a DIFFERENT expert — a cliff, not a small error. gemv_wf32_a8 (moe.go) quantizes
// the activation to int8 (~1e-2), which can flip a decision near a tie; safe only when the margin is
// wide, and the tiny fixture's margin was CONSTRUCTED wide, so "no flip" there would be circular for
// a trained 128-expert/top-8 router whose 8th-vs-9th boundary is far tighter. Quantizing nothing
// removes the perturbation entirely: this is bit-exact to the CPU f32 router modulo f32 reduction
// order (~1e-6), so routing cannot flip from activation quant at ANY expert count — routing is off
// the resident suspect list permanently. Mirrors cuda/router_f32.cu gemv_f32_f32.
kernel void gemv_f32_f32(device const float* wf[[buffer(0)]], device const float* a[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    device const float* wr = wf + (uint)gid*K;
    float acc = 0.0f;
    for (uint k=lid; k<K; k+=32u) acc += wr[k]*a[k];
    acc = simd_sum(acc);
    if (lid==0) out[gid] = acc;
}
`

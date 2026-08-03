//go:build darwin

package metal

// Gemma-4 enable_moe_block (26B-A4B) MSL kernels — the parallel dense‖MoE FFN that the generic
// moe.go path (Mixtral/Qwen/GLM shape) cannot express. Kept in their OWN file/const, concatenated
// after moeKernels, so moe.go's audited MoE kernels are not touched — the same discipline as CUDA's
// separate router_f32.cu (which exists so adding a kernel doesn't rewrite the audited moe.ptx).
//
// The router-first step (9c Step 5a) added gemv_f32_f32. This block adds the remaining primitives
// the parallel dense‖MoE forward needs — each verified in isolation against a CPU oracle before the
// composition is wired (gemma4_moekernels_test.go), the same 2b→2c→2d order CUDA followed.
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

// rmsnorm_nw: weightless RMSNorm, OUT-OF-PLACE (src → dst). The Gemma-4 MoE router norms the RAW
// residual h WITHOUT mutating it — h still feeds the parallel dense branch, the expert branch, and
// the final residual add — so the in-place rmsnorm_f32 cannot be used here. dst[i] = src[i] *
// rsqrt(mean(src^2)+eps). The learned routerScale and hidden^-0.5 are folded into the router weight
// columns at build (RouterProjScaled), so nothing else is applied here. One threadgroup, tree
// reduction (matches rmsnorm_f32's reduction so the two norms of h agree to f32 order). Mirrors
// cuda/router_f32.cu rmsnorm_nw.
kernel void rmsnorm_nw(device const float* src[[buffer(0)]], device float* dst[[buffer(1)]],
    constant uint& H[[buffer(2)]], constant float& eps[[buffer(3)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=src[i]*src[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs) dst[i]=src[i]*rms;
}

// scale_wgt_by_expert: fold Gemma-4's learned per-expert scale into the routed weights, AFTER
// moe_route's top-k + renormalize: wgt[k] *= perExpertScale[idx[k]] (CPU: wts[j] =
// (topv[j]/sum)*perExpertScale[idx[j]]). idx is a moe_route device output, so this must run ON-GPU:
// a host fold would read idx/wgt back per token, reintroducing the per-token sync the on-device
// router exists to avoid. K lanes (K = top_k, tiny), one dispatch. Mirrors cuda/router_f32.cu.
kernel void scale_wgt_by_expert(device float* wgt[[buffer(0)]], device const uint* idx[[buffer(1)]],
    device const float* perExpertScale[[buffer(2)]], constant uint& K[[buffer(3)]],
    uint k[[thread_position_in_grid]]) {
    if (k>=K) return;
    wgt[k] *= perExpertScale[idx[k]];
}

// scale_vec: x[i] *= s. Gemma-4's per-layer output scalar (out = (h + combined) * layerScalar),
// applied to the residual after the joint post-norm. s is a one-float buffer (Metal uniforms are
// buffers) so it can be per-layer. Mirrors cuda/router_f32.cu scale_vec.
kernel void scale_vec(device float* x[[buffer(0)]], device const float* s[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i] *= s[0]; }

// zero_vec: x[i] = 0. Clears the Gemma-4 expert accumulator g4x2 before the fixed-k weighted-
// accumulate loop (gemv_w4a8_moe_wacc always does out[row] += ...). A multiply-by-zero would NOT
// do — g4x2 is persistent scratch that can hold a stale NaN on the first token (NaN*0 = NaN).
kernel void zero_vec(device float* x[[buffer(0)]], uint i[[thread_position_in_grid]]) { x[i] = 0.0f; }
`

// megakernel.cu — fused decode-layer kernel(s) for goinfer's cgo-free CUDA backend.
//
// SCAFFOLD — NOT YET FUNCTIONAL. The box (2070 SUPER) fills the bodies per
// docs/cuda-megakernel-spec.md §2–§4 and compiles with:  go generate ./cuda
//   (nvcc -arch=sm_75 -ptx megakernel.cu -o megakernel.ptx)
//
// Dense residency-eligible Qwen2/Llama decode only. Correctness bar: greedy
// argmax token-identical to the CPU golden (spec §0) — match the token and the
// quant packing exactly; f32 attention/RoPE need only cosine ≈ 1.0, not bit-exact.
//
// Launch structure (spec §5.2): three fused super-kernels per layer, split at the
// cross-block sync boundaries so plain cuLaunchKernel suffices (gocudrv has no
// cooperative launch). Per layer: k1_preattn -> k2_attn -> k3_ffn; the launch
// boundaries are the grid syncs. Collapses the WGSL path's ~13 dispatches -> 3.
//
// Quant packing the GEMVs MUST reproduce (spec §3):
//   W8A8 : per-row int8, scale = maxabs/127, K padded to mult-16, 16 int8 / 128b load.
//          acc = sum(int8 * int8) as i32, then * aScale * bScale[row].
//   W4A8 : group = 32, per-group scale = maxabs/7 (f16), nibble = q+8 in [1,15];
//          element k -> word k/8, nibble 4*(k%8); acc = sum((nibble-8)*int8act) i32,
//          * f16 group-scale, * aScale.  Activation = per-token int8 (one scale).
//   KV   : f32, layout [pos*kvDim + head*hd + d]; K stored post-RoPE; RoPE
//          theta = pos * invFreq[d], rotate (d, d+half) for d < half.

extern "C" {

// k1_preattn: RMSNorm(x)+quant -> Q/K/V GEMV (+optional bias) -> RoPE(q) +
// RoPE-store(k) + store(v) into the resident KV cache at base pos*kvDim.
// TODO(box): implement per spec §2 stages 1-3, §3 packing, §4 KV layout.
__global__ void k1_preattn(/* weights, x, kvCache, uniforms... */) {
    // scaffold
}

// k2_attn: attention (QK^T, online softmax, .V) over KV[0..pos] with GQA
// (kvh = qh/group) -> quant(ctx) -> O-proj -> residual add into x.
// TODO(box): implement per spec §2 stages 4-6.
__global__ void k2_attn(/* q, kvCache, Wo, x, uniforms... */) {
    // scaffold
}

// k3_ffn: RMSNorm(x)+quant -> gate/up GEMV -> SwiGLU((g/(1+e^-g))*u)+quant ->
// down GEMV -> residual add into x.
// TODO(box): implement per spec §2 stages 7-10.
__global__ void k3_ffn(/* Wgate, Wup, Wdown, x, uniforms... */) {
    // scaffold
}

} // extern "C"

//go:build cuda

package cuda

import (
	_ "embed"
	"math"
)

// The production decode kernels (NVRTC to PTX, go:embed, driver-JIT via cuModuleLoadDataEx), embedded in a non-test file
// so BuildResident/cudaResident can load them; the standalone bandwidth/parity tests reference the same vars.
//
// Every .ptx below is a build artifact of the .cu of the same name, byte-for-byte reproducible by ./build_ptx.sh (NVRTC
// rather than nvcc). Regenerate, never hand-edit: a .ptx whose .cu is missing is a kernel nobody can review or change.
// Most kernels are their own module on purpose: adding a kernel to an existing .cu regenerates its PTX and risks
// shifting codegen for the kernels the parity gates rest on. moe.ptx, glue.ptx and gemv_fwd.ptx are the audited
// artifacts, pinned at NVRTC 12.6.85 (cuda/testdata/REGEN.md); the others are built at the ambient NVRTC.
//
//go:generate ./build_ptx.sh

// gemvFwdPTX: the LLM-specific forward kernels, kv_store / rope_kv. The generic quantized GEMVs live in aikit/gpu
// (gpu.QuantGEMVPTX, Device.NewQuantGEMV; see backend.go); the name is kept for continuity with the .cu it is built
// from.
//
//go:embed testdata/gemv_fwd.ptx
var gemvFwdPTX []byte

// gemvW8BatchedPTX: gemv_w8a8_batched, the weight-stationary batched GEMV for M=len prefill on int8 bundles.
// Bit-identical to aikit's gemv_w8a8_fwd per output element by construction: int8 is per-row symmetric, so the dot is an
// exact int32 __dp4a sum (reorder-independent) and the scales apply once at the end in the same explicit
// __fmul_rn/__fmaf_rn form.
//
//go:embed testdata/gemv_w8a8_batched.ptx
var gemvW8BatchedPTX []byte

// prefillBatchedPTX: the batched (M=len) glue kernels of the weight-stationary prefill: rmsnorm_quant_batched,
// rope_kv_batched, attn_batched (causal, per-row sliding window), glu_quant_batched, residual_batched. Each is the M=1
// kernel with an M dimension added and the per-row math copied verbatim, so each row is bit-identical to its sequential
// counterpart.
//
//go:embed testdata/prefill_batched.ptx
var prefillBatchedPTX []byte

// decodeSplitKVPTX: the high-occupancy, bit-identical decode attention (splitkv_scores, splitkv_softmax, splitkv_vsum).
// attn_batched(M=1) is split along its independent axes (scores over keys, V-sum over dims), so every order-dependent
// softmax fold stays whole and in order and the result is byte-identical, but fills the SMs. See
// docs/tasks/task-decode-splitkv-attention.md.
//
//go:embed testdata/decode_splitkv.ptx
var decodeSplitKVPTX []byte

// deltaNetPTX: the Gated-DeltaNet decode mixer (delta_conv / delta_gates / delta_norm / delta_rule / delta_gnorm /
// delta_qsplit / delta_attn_gate), loaded only for that family; a load failure declines the resident build. Nothing else
// here carries recurrent state, so the conv and state plumbing are new code, not a reuse.
//
//go:embed testdata/deltanet.ptx
var deltaNetPTX []byte

// gptOssActPTX: glu_quant_gptoss, gpt-oss's clamped interleaved-SwiGLU expert epilogue (per-expert biases, an asymmetric
// clamp, an alpha-scaled sigmoid gate and a +1 on the linear branch). Everything after the activation (max-reduce,
// symmetric int8 quant, packed store) is glu_quant's unchanged, so the down-projection GEMV consumes an identical
// format.
//
//go:embed testdata/gptoss_act.ptx
var gptOssActPTX []byte

// actGroupPTX: per-32 activation quantization for the resident decode path: rmsnorm_quant_g32, quant_vec_g32,
// glu_quant_g32 and the gemv_w4a8_g32 / gemv_w8a8_g32 GEMVs that read one activation scale per 32 elements (actgroup.cu;
// docs/tasks/task-actquant-pergroup-2026-09.md). A per-row model never loads it.
//
//go:embed testdata/actgroup.ptx
var actGroupPTX []byte

// gemvRNPTX: gemv_w4a8_rn, the register-blocked batched GEMV (RN output rows per warp) that reuses each coalesced
// activation load across RN rows. Bit-identical (per-row facc, one reduce each).
//
//go:embed testdata/gemv_w4a8_rn.ptx
var gemvRNPTX []byte

// gluePTX: the per-token elementwise/attention glue: rmsnorm_quant, quant_vec, rope, attention (GQA online softmax),
// swiglu_quant, residual. The audited trio is pinned at NVRTC 12.6.85 by cuda/testdata/REGEN.md's procedure, which also
// records the per-kernel hash audit.
//
//go:embed testdata/glue.ptx
var gluePTX []byte

// argmaxPTX: argmax_reduce, the greedy-decode reduction, split out of glue.ptx so a kernel fix does not regenerate the
// audited glue kernels. See cuda/argmax.cu.
//
//go:embed testdata/argmax.ptx
var argmaxPTX []byte

// topkPTX: topk_select, the device-side bounded top-K the sampled-decode fast path reads instead of the full logits row.
// Built at the ambient NVRTC (12.9.86). See cuda/topk.cu.
//
//go:embed testdata/topk.ptx
var topkPTX []byte

// gumbelPTX: gumbel_stage1/gumbel_stage2, device-side temperature-only sampling by Gumbel-max. Built at the ambient
// NVRTC. See cuda/gumbel.cu.
//
//go:embed testdata/gumbel.ptx
var gumbelPTX []byte

// decodeFAPTX: fa_partial_{64,128,256} / fa_combine, the flash-decode lane (default on; GOINFER_CUDA_FLASH_DECODE=0
// turns it off), loaded only when the lane is requested. Built at the ambient NVRTC (12.9.86). See cuda/decode_fa.cu.
//
//go:embed testdata/decode_fa.ptx
var decodeFAPTX []byte

// loraPTX: lora_delta_down/lora_delta_up, compute-time LoRA (docs/tasks/task-gpu-paths-2026-09.md). A new kernel pair is
// its own module per cuda/testdata/REGEN.md's rule (new .cu, new .ptx, built at whatever NVRTC is present). See
// cuda/lora.cu.
//
//go:embed testdata/lora.ptx
var loraPTX []byte

// attnBlockPTX: attn_block_full, the DFlash block drafter's non-causal attention over [ctx||block]. A verbatim copy of
// prefill_batched.cu's attn_batched with one line changed (nKeys = startPos+M for every row, not startPos+m+1), because
// the drafter's block is bidirectional where the target's verify is causal. See cuda/attn_block.cu.
//
//go:embed testdata/attn_block.ptx
var attnBlockPTX []byte

// attnImgPrefillPTX: attn_img_batched, Gemma 3's image-block bidirectional prefill attention
// (decoder.ResidentImagePrefill / cuda.PrefillImageLast). A verbatim copy of attn_batched, except that a query row whose
// own position lies inside a caller-supplied [imgStart,imgEnd) range sees the whole block; text before and after stays
// exactly causal (unlike attn_block_full's uniform widening).
//
// Guardrail: the kernel's sliding-window start is derived from the row's plain causal key count, never from the
// image-widened one, matching decoder/kvcache.go's WindowStart/attendHi split, which the CPU reference this kernel must
// match bit for bit also keeps decoupled. Re-coupling it to attn_batched's formula would under-size the shared-memory
// window for a windowed layer whenever an image block starts more than one window-length into the sequence: a
// shared-memory out-of-bounds write, not a clean wrong answer, and invisible on a fixture whose image sits near the
// start of the prompt. Full reasoning: the header of cuda/attn_img_prefill.cu.
//
//go:embed testdata/attn_img_prefill.ptx
var attnImgPrefillPTX []byte

// ropeMRopePrefillPTX: rope_kv_mrope_batched, Qwen2.5-VL's m-RoPE batched-prefill rotation (decoder.ResidentMRoPEPrefill
// / cuda.PrefillMRoPELast). A copy of rope_kv_batched whose rotation angle is a per-row, per-frequency-section lookup:
// frequency index d rotates by pos[comp(d)]*invFreq[d], where comp(d) picks temporal/height/width from the model's
// MRopeSection cumulative boundaries.
//
// Every row needs the per-row lookup, not just image-block rows: decoder/rope.go's mropePositions resumes scalar
// counting after an image block from a value compressed by the merged image grid, not the sequential count
// rope_kv_batched assumes. The KV-cache store index (pos = startPos+m) stays row-sequential; only the rotation angle
// widens, as rope_kv's pos/ropePos split does for decode (cuda/gemv_fwd.cu). See cuda/rope_mrope_prefill.cu.
//
//go:embed testdata/rope_mrope_prefill.ptx
var ropeMRopePrefillPTX []byte

// ropePairwisePTX: rope_kv_pw / rope_kv_batched_pw / rope_kv_mrope_batched_pw, the GPT-J pairwise (dims 2d, 2d+1) twins
// of rope_kv / rope_kv_batched / rope_kv_mrope_batched, with the same argument lists and launch geometry, bound in their
// place when decoder.Model.PairwiseRoPEResident() says so. The NeoX kernels rotate (d, d+half), so a pairwise family run
// through them is exact at position 0 and wrong afterwards. Built at NVRTC 12.6.85, the pinned toolchain (the control is
// in docs/measurements/cuda-pairwise-rope-2026-10-01.md). See cuda/rope_pairwise.cu.
//
//go:embed testdata/rope_pairwise.ptx
var ropePairwisePTX []byte

// layernormQuantPTX: layernorm_quant_batched / layernorm_f32_batched, LayerNorm (mean and variance, weight and bias) for
// the resident SigLIP vision tower (docs/multimodal.md) and for LayerNorm text families; the primitive next to the
// RMSNorm every other family uses. See cuda/layernorm_quant.cu, cuda/vision_encoder.go.
//
//go:embed testdata/layernorm_quant.ptx
var layernormQuantPTX []byte

// geluQuantPTX: gelu_quant_batched, the resident SigLIP vision tower's plain (non-gated) MLP activation,
// FC2(GELU_tanh(FC1(x))), as opposed to glue.cu's gated glu_quant. See cuda/gelu_quant.cu, cuda/vision_encoder.go.
//
//go:embed testdata/gelu_quant.ptx
var geluQuantPTX []byte

// attnFusedPTX: attn_fused_hd64 / attn_fused_hd128, the L2 FlashAttention-style fused prefill attention
// (docs/completed/task-prefill-gap.md §4 L2). See cuda/attn_fused.cu.
//
//go:embed testdata/attn_fused.ptx
var attnFusedPTX []byte

// attnFusedBMPTX: tile-shape variants of attn_fused (32-row query tile, 64- or 32-key tile), template parameters of
// cuda/attn_fused_bm.cu. Experiment arms (docs/measurements/attn-fused-tile-PREREGISTERED.md), selected only by
// attnTile.
//
//go:embed testdata/attn_fused_bm.ptx
var attnFusedBMPTX []byte

// attnFusedVitPTX: attn_vit_hd72_bm64 / attn_vit_hd72_bm128, the non-causal fused attention for the SigLIP vision tower
// (hd 72 padded to 80); see cuda/attn_fused_vit.cu and docs/measurements/vision-tower-attn-PREREGISTERED.md.
//
//go:embed testdata/attn_fused_vit.ptx
var attnFusedVitPTX []byte

// towerBasePTX: tower_pos_add / tower_clamp / tower_clamp_copy / tower_rope_axial / tower_mul / tower_scale, the CUDA
// vision-tower base's own kernels (docs/tasks/task-multimodal-support-2026-10.md), added to aikit's gpu.ViT module.
// Built at NVRTC 12.9.86.
//
//go:embed testdata/tower_base.ptx
var towerBasePTX []byte

// gemmMMAPTX: gemm_w4a8_mma, the L3 tensor-core int4xint8 GEMM with group scales (docs/completed/task-prefill-gap.md §4
// L3).
//
//go:embed testdata/gemm_w4a8_mma.ptx
var gemmMMAPTX []byte

// moePTX: sparse mixture-of-experts — moe_route (on-GPU router), gemv_f32_a8 (the f32 router
// projection), gemv_w4a8_moe / _wacc (indexed stacked-expert GEMVs), shared_gate_combine.
//
//go:embed testdata/moe.ptx
var moePTX []byte

// routerF32PTX: gemv_f32_f32, the pure-f32 router projection (no activation quant) for Gemma 4's router. See
// cuda/router_f32.cu.
//
//go:embed testdata/router_f32.ptx
var routerF32PTX []byte

// fusedQKVPTX: the K1 super-kernel: rmsnorm+quant folded into the Q/K/V GEMV by redundant per-block recompute, removing
// a GridX:1 glue kernel and 3 launches.
//
//go:embed testdata/fused_qkv.ptx
var fusedQKVPTX []byte

// fusedQKVRowsPTX: fused_rms_qkv_rows, fused_rms_qkv with rows-per-warp as a runtime parameter (fewer redundant rmsnorm
// prologues). Built at the ambient NVRTC (12.9.86). See cuda/fused_qkv_rows.cu.
//
//go:embed testdata/fused_qkv_rows.ptx
var fusedQKVRowsPTX []byte

// fusedGURowsPTX: fused_rms_gu_rows, fused_rms_gu with rows-per-warp as a runtime parameter. Built at the ambient NVRTC
// (12.9.86). See cuda/fused_gu_rows.cu.
//
//go:embed testdata/fused_gu_rows.ptx
var fusedGURowsPTX []byte

// mlaPTX: mla_latent_store, mla_head_matvec, mla_q_rope, mla_attn — DeepSeek / Kimi
// Multi-head Latent Attention (MLA) resident decode kernels (FeatMLA).
//
//go:embed testdata/mla.ptx
var mlaPTX []byte

// nibblePosFast maps a weight's index within an 8-weight word (0..7) to its nibble slot,
// so the coalesced GEMV's even/odd byte split (word&0x0F0F0F0F / (word>>4)&0x0F0F0F0F)
// lands weights 0..3 in the low-nibble bytes and 4..7 in the high-nibble bytes.
func nibblePosFast(i int) int {
	if i < 4 {
		return 2 * i
	}
	return 2*(i-4) + 1
}

// permuteFast converts a natural-order packed word (element i at nibble i, the straight
// byte copy of the decoder's int4) into the fast nibble-permuted layout the coalesced
// forward GEMV expects (element i at nibble nibblePosFast(i)).
func permuteFast(w uint32) uint32 {
	var o uint32
	for i := range 8 {
		nv := (w >> (4 * i)) & 0xf
		o |= nv << (4 * nibblePosFast(i))
	}
	return o
}

// f32tof16 encodes an IEEE-754 float32 into a float16 bit pattern, byte-for-byte identical to decoder.f32ToF16bits, the
// canonical resident-backend f16 scale representation that metal/pack.go, aikit/linalg and the GOINFER_INT4_F16_SCALES
// CPU diagnostic replicate. The int4 group scales it encodes (ws16 in resident.go) must match every other backend's f16
// scales bit for bit. It rounds half up with gradual underflow to subnormals; it is not RNE or saturating, which would
// reintroduce a divergence.
func f32tof16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xFF) - 112 // f16-biased exponent = (exp-127) + 15
	m := b & 0x7FFFFF
	switch {
	case (b>>23)&0xFF == 0xFF: // Inf / NaN
		if m != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case e >= 0x1F: // overflow → Inf (matches the canonical; unreachable for scales)
		return sign | 0x7C00
	case e <= 0: // subnormal or underflow to zero
		if e < -10 {
			return sign
		}
		m |= 0x800000 // restore the implicit leading 1
		sh := uint32(14 - e)
		return sign | uint16((m+(1<<(sh-1)))>>sh) // round-half-up
	default: // normal f16
		half := sign | uint16(e<<10) | uint16(m>>13)
		if m&0x1000 != 0 { // guard bit set → round up
			half++
		}
		return half
	}
}

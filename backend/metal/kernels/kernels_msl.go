package kernels

import _ "embed"

// AllKernels is the full dense-decode-layer MSL kernel set in one library (W8A8 path; W4A8 is validated separately, so this
// proves assembly, not the int4 packing again).
//
// Six kernels here have no production pipeline: gemv_w4a8_bias, gemv_w4a8_sa_amax, gemv_w4a8_sa_bk, gemv_w4a8_sa_qv, gemv_w8a8
// and rope2_kv. Each but one backs a dedicated micro-benchmark or recorded-negative regression test (profile_test.go,
// batchk_test.go, sa_qv_fusion_test.go, gemv_test.go, rope2_kv_test.go). Do not delete one because nothing in production calls
// it: that breaks the test that keeps its measurement or negative result honest. gemv_w4a8_sa_amax has no reference anywhere and
// is the one genuinely dead survivor; it is kept so its history stays visible beside the others.
//
// It is the decode_*.metal files beside this file joined in the order below, between the attention helpers and the
// MoE, DeltaNet, sampling, Mamba and MLA sets: MSL needs a helper declared before its first use.
// TestMSLSource_librariesArePinned holds the joined bytes.
var AllKernels = mslPrelude + AttnScoreMSL + KVRowMSL + "\n" +
	decodeNorm + decodeGemvW8A8 + decodeGemvW4A8 + decodeGemvRows + decodeGemvW4A8Epilogue + decodeGemvW4F16 + decodeQ4K +
	decodeHead + decodeRopeKV + decodeAttention + decodeAttentionFABlk + decodeAttentionI8 + decodeMLP + decodeLoRA +
	decodeQKNormCopy +
	MoeKernels + Gemma4MoeKernels + DeltaNetKernels + GumbelMSLKernels + MoeBatchKernels + MambaKernels + MLAKernels

// mslPrelude opens AllKernels and PrefillKernels.
const mslPrelude = "\n#include <metal_stdlib>\nusing namespace metal;\n"

// AllKernels' own sources, one kernel family each.
var (
	//go:embed decode_norm.metal
	decodeNorm string
	//go:embed decode_gemv_w8a8.metal
	decodeGemvW8A8 string
	//go:embed decode_gemv_w4a8.metal
	decodeGemvW4A8 string
	//go:embed decode_gemv_rows.metal
	decodeGemvRows string
	//go:embed decode_gemv_w4a8_epilogue.metal
	decodeGemvW4A8Epilogue string
	//go:embed decode_gemv_w4f16.metal
	decodeGemvW4F16 string
	//go:embed decode_q4k.metal
	decodeQ4K string
	//go:embed decode_head.metal
	decodeHead string
	//go:embed decode_rope_kv.metal
	decodeRopeKV string
	//go:embed decode_attention.metal
	decodeAttention string
	//go:embed decode_attention_fa_blk.metal
	decodeAttentionFABlk string
	//go:embed decode_attention_i8.metal
	decodeAttentionI8 string
	//go:embed decode_mlp.metal
	decodeMLP string
	//go:embed decode_lora.metal
	decodeLoRA string
	//go:embed decode_qk_norm_copy.metal
	decodeQKNormCopy string
)

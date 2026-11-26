package kernels

import _ "embed"

// PrefillKernels is the f16 simdgroup_matrix (MMA) prefill library, for fast prompt ingestion: an MMA GEMM reuses each
// weight across all M prompt rows, which the int4 scalar-MAC decode path cannot. Activations flow in f16 (no int8
// quant); weights stay int4 and are dequantized to f16 in-kernel. A separate library from allKernels, so the decode
// path is unchanged; ensurePrefill compiles it lazily.
//
// It is the prefill_*.metal files beside this file joined in the order below, after the attention helpers;
// TestMSLSource_librariesArePinned holds the joined bytes.
var PrefillKernels = mslPrelude + AttnScoreMSL + KVRowMSL + "\n" +
	prefillGEMM + prefillGEMMKQ + prefillNormAct + prefillRopeQKNorm + prefillAttention + prefillAttentionSteel +
	prefillKVStore + prefillMoE + prefillGates

// PrefillKernels' own sources, one kernel family each.
var (
	//go:embed prefill_gemm.metal
	prefillGEMM string
	//go:embed prefill_gemm_kq.metal
	prefillGEMMKQ string
	//go:embed prefill_norm_act.metal
	prefillNormAct string
	//go:embed prefill_rope_qk_norm.metal
	prefillRopeQKNorm string
	//go:embed prefill_attention.metal
	prefillAttention string
	//go:embed prefill_attention_steel.metal
	prefillAttentionSteel string
	//go:embed prefill_kv_store.metal
	prefillKVStore string
	//go:embed prefill_moe.metal
	prefillMoE string
	//go:embed prefill_gates.metal
	prefillGates string
)

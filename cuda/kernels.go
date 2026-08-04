//go:build cuda

package cuda

import (
	_ "embed"
	"math"
)

// The production decode kernels (NVRTC→PTX→go:embed→driver-JIT via cuModuleLoadDataEx).
// These embeds live in a non-test file so the real backend (BuildResident/cudaResident)
// can load them; the standalone bandwidth/parity tests reference the same vars.
//
// EVERY .ptx below is a build artifact of the .cu of the same name in this directory, and
// is byte-for-byte reproducible by running ./build_ptx.sh (which explains why the build
// uses NVRTC rather than nvcc). Regenerate — never hand-edit — the PTX; a .ptx whose .cu
// is missing is a kernel nobody can review or change.
//
//go:generate ./build_ptx.sh

// gemvFwdPTX: the LLM-specific forward kernels — kv_store / rope_kv.
//
// The generic quantized GEMVs it used to carry (gemv_w4a8_fwd, gemv_w8a8_fwd) moved to
// aikit/gpu >= v0.4.0 in the Phase-1b blob-split and are loaded via
// gpu.QuantGEMVPTX / Device.NewQuantGEMV (backend.go). The name is kept for continuity
// with the .cu it is built from.
//
//go:embed testdata/gemv_fwd.ptx
var gemvFwdPTX []byte

// gemvBatchedPTX: gemv_w4a8_batched — the weight-stationary batched GEMV for M=len prefill.
// Bit-identical to aikit's gemv_w4a8_fwd per output element (same per-word float scale-accumulate
// order), evaluated for M activation columns per weight-row load. Its own file (gemv_w4a8_batched.cu);
// the audited PTX (moe.ptx, gemv_fwd/glue) is untouched.
//
//go:embed testdata/gemv_w4a8_batched.ptx
var gemvBatchedPTX []byte

// prefillBatchedPTX: the batched (M=len) glue kernels for the weight-stationary prefill path —
// rmsnorm_quant_batched, rope_kv_batched, attn_batched (causal + per-row sliding window),
// glu_quant_batched, residual_batched. Each is the corresponding M=1 kernel (glue/gemv_fwd) with an
// M dimension added and the per-row math copied verbatim, so each row is bit-identical to its
// sequential counterpart. Its own file (prefill_batched.cu); the audited PTX is untouched.
//
//go:embed testdata/prefill_batched.ptx
var prefillBatchedPTX []byte

// decodeSplitKVPTX: the high-occupancy, bit-identical decode attention (Campaign A) — splitkv_scores,
// splitkv_softmax, splitkv_vsum. The single-block attn_batched(M=1) is split along its INDEPENDENT
// axes (scores over keys, V-sum over dims) so every order-dependent softmax fold stays whole and
// in-order → byte-identical to attn_batched(M=1), but fills the SMs. Own file (decode_splitkv.cu);
// audited glue.ptx / moe.ptx untouched. See docs/task-decode-splitkv-attention.md.
//
//go:embed testdata/decode_splitkv.ptx
var decodeSplitKVPTX []byte

// gemvStagedPTX: gemv_w4a8_staged — the activation-staged batched GEMV. Bit-identical to
// gemv_w4a8_fwd (facc live in registers across all K-chunks, single warp-reduce), but stages the
// [MT,KC] activation tile in shared memory so it is read once per block instead of once per output
// row — the fix for the profiled activation-L2-read bound. Own file (gemv_w4a8_staged.cu).
//
//go:embed testdata/gemv_w4a8_staged.ptx
var gemvStagedPTX []byte

// gemvRNPTX: gemv_w4a8_rn — register-blocked batched GEMV (RN output rows per warp), so each coalesced
// activation load is reused across RN rows: RN× fewer L1TEX loads, the profile-justified latency fix.
// Bit-identical (per-row facc, one reduce each). Own file (gemv_w4a8_rn.cu).
//
//go:embed testdata/gemv_w4a8_rn.ptx
var gemvRNPTX []byte

// gluePTX: the per-token elementwise/attention glue — rmsnorm_quant, quant_vec, rope,
// attention (GQA online softmax), swiglu_quant, residual, argmax_reduce.
//
//go:embed testdata/glue.ptx
var gluePTX []byte

// moePTX: sparse mixture-of-experts — moe_route (on-GPU router), gemv_f32_a8 (the f32 router
// projection), gemv_w4a8_moe / _wacc (indexed stacked-expert GEMVs), shared_gate_combine.
//
//go:embed testdata/moe.ptx
var moePTX []byte

// routerF32PTX: gemv_f32_f32 — pure-f32 router projection (NO activation quant) for Gemma-4's
// router, the discrete-failure path. Kept in its own module so adding it did not force a
// regeneration of the audited 12.6 moe.ptx at this box's 12.9 NVRTC. See cuda/router_f32.cu.
//
//go:embed testdata/router_f32.ptx
var routerF32PTX []byte

// fusedQKVPTX: K1 super-kernel — rmsnorm+quant folded into the Q/K/V GEMV via redundant
// per-block recompute, killing a GridX:1 glue kernel and 3 launches (spec §5.2).
//
//go:embed testdata/fused_qkv.ptx
var fusedQKVPTX []byte

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
	for i := 0; i < 8; i++ {
		nv := (w >> (4 * i)) & 0xf
		o |= nv << (4 * nibblePosFast(i))
	}
	return o
}

// f32tof16 encodes an IEEE float32 into a float16 bit pattern (round-to-nearest-even, simple).
func f32tof16(f float32) uint16 {
	b := math.Float32bits(f)
	s := uint16((b >> 16) & 0x8000)
	e := int32((b>>23)&0xff) - 127 + 15
	m := b & 0x7fffff
	if e <= 0 {
		return s
	}
	if e >= 0x1f {
		return s | 0x7c00
	}
	return s | uint16(e<<10) | uint16(m>>13)
}

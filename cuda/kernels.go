//go:build cuda

package cuda

import _ "embed"

// The production decode kernels (NVRTC→PTX→go:embed→driver-JIT via cuModuleLoadDataEx).
// These embeds live in a non-test file so the real backend (BuildResident/cudaResident)
// can load them; the standalone bandwidth/parity tests reference the same vars.

// gemvFwdPTX: forward GEMVs — gemv_w4a8_fwd (coalesced + 2× ILP unroll, f32 group scales,
// on-device aScale ptr, per-row bias) / gemv_w8a8_fwd / kv_store.
//
//go:embed testdata/gemv_fwd.ptx
var gemvFwdPTX []byte

// gluePTX: the per-token elementwise/attention glue — rmsnorm_quant, quant_vec, rope,
// attention (GQA online softmax), swiglu_quant, residual, argmax_reduce.
//
//go:embed testdata/glue.ptx
var gluePTX []byte

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

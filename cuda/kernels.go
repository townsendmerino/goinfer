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

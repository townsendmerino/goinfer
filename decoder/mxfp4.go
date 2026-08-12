package decoder

import (
	"fmt"
	"math"
)

// MXFP4 (OCP Microscaling FP4, GGML quant type 39) — used by the gpt-oss family. A block is 32
// elements: one e8m0 8-bit power-of-two scale byte, then 16 bytes each packing two e2m1 4-bit
// values (4.25 bits/weight). 17 bytes/block.
//
// The layout is NOT inferred — it is transcribed from the reference `gguf` Python library
// (gguf/quants.py MXFP4) and verified bit-for-bit against it on a real gpt-oss checkpoint
// (mxfp4_test.go), the same order cpubrrr used to get its forward pass right on the first run.
// e2m1 has 16 representable values; the table below is those values DOUBLED (so the lookup stays
// integer), paired with a HALF-scale e8m0 — d*half · value*2 == the true product, and keeping
// both integer is what lets the eventual SIMD kernel avoid a float dequant in the inner loop.
const (
	mxfp4BlockElems = 32 // elements per block
	mxfp4BlockBytes = 17 // 1 scale + 16 packed-nibble bytes
)

// mxfp4KValues is the e2m1 dequant table (values doubled), indexed by the 4-bit code.
// ref: gguf/quants.py MXFP4.kvalues; OCP MX v1.0 spec.
var mxfp4KValues = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// e8m0ToF32Half converts an 8-bit e8m0 block scale to float32, matching ggml's
// ggml_e8m0_to_fp32_half (gguf/quants.py MXFP4.e8m0_to_fp32_half): x<2 uses a subnormal bit
// pattern, else the exponent field is x-1. Replicated via the exact bit formula (not 2^(x-128))
// so the subnormal cases at x∈{0,1} are bit-identical to the reference.
func e8m0ToF32Half(x uint8) float32 {
	var bits uint32
	if x < 2 {
		bits = uint32(0x00200000) << uint32(x)
	} else {
		bits = uint32(x-1) << 23
	}
	return math.Float32frombits(bits)
}

// mxfp4DequantBlock dequantizes one 17-byte MXFP4 block into 32 float32 weights, in natural
// element order: element j (j<16) is the LOW nibble of byte j, element j+16 the HIGH nibble —
// matching gguf/quants.py's reshape((n,2,16)) split. dst must have length >= 32.
func mxfp4DequantBlock(block []byte, dst []float32) {
	d := e8m0ToF32Half(block[0])
	qs := block[1:mxfp4BlockBytes] // 16 bytes
	for j := range 16 {
		b := qs[j]
		dst[j] = d * float32(mxfp4KValues[b&0x0F])
		dst[j+16] = d * float32(mxfp4KValues[b>>4])
	}
}

// mxfp4Dequant dequantizes n contiguous MXFP4 blocks (n*17 bytes) into n*32 float32 weights.
// Returns an error rather than panicking on a short/misaligned buffer (hostile-input discipline).
func mxfp4Dequant(raw []byte, nBlocks int) ([]float32, error) {
	if nBlocks < 0 || len(raw) < nBlocks*mxfp4BlockBytes {
		return nil, fmt.Errorf("mxfp4: buffer too short — %d bytes for %d blocks (need %d)",
			len(raw), nBlocks, nBlocks*mxfp4BlockBytes)
	}
	out := make([]float32, nBlocks*mxfp4BlockElems)
	for i := range nBlocks {
		mxfp4DequantBlock(raw[i*mxfp4BlockBytes:], out[i*mxfp4BlockElems:])
	}
	return out, nil
}

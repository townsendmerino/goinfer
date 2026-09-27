//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"
)

// TestMC3SimdSumTree: EXPLORATORY, for MC3's bit-identical matrix-unit GEMV. Does this GPU's simd_sum add its 32
// lanes in a fixed butterfly order? If it equals the xor tree (16, 8, 4, 2, 1) or (1, 2, 4, 8, 16) exactly, a kernel
// can reproduce the decode GEMV's reduction by computing part of that tree in registers instead of one simd_sum per
// output. Random f32 inputs spanning many exponents, so different trees round differently.
//
//	GOINFER_METAL_MC3=1 go test -count=1 -run '^TestMC3SimdSumTree$' -v ./metal/
func TestMC3SimdSumTree(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseAll()
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void sstree(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    float v = in[gid];
    float a = v, b = v;
    a += simd_shuffle_xor(a, 16); a += simd_shuffle_xor(a, 8); a += simd_shuffle_xor(a, 4);
    a += simd_shuffle_xor(a, 2); a += simd_shuffle_xor(a, 1);
    b += simd_shuffle_xor(b, 1); b += simd_shuffle_xor(b, 2); b += simd_shuffle_xor(b, 4);
    b += simd_shuffle_xor(b, 8); b += simd_shuffle_xor(b, 16);
    float s = simd_sum(v);
    if (lane == 0) { uint w = gid >> 5; out[w*3] = s; out[w*3+1] = a; out[w*3+2] = b; }
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "sstree")
	if err != nil {
		t.Fatal(err)
	}
	const W = 1 << 16 // simdgroups
	in := make([]float32, W*32)
	seed := uint32(12345)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	for i := range in {
		m := float64(rnd())/float64(1<<32) - 0.5
		in[i] = float32(math.Ldexp(m, int(rnd()%24)-12))
	}
	ib, ob := NewBufferFloats(d, in), d.NewBufferLen(W*3)
	cq := d.NewCommandQueue()
	e := cq.Begin()
	e.DispatchTG(p, W*32, 256, 0, ib, ob)
	e.End()
	o := ob.Floats()[:W*3]
	// CPU sequential order as a control that SHOULD differ
	seqDiff, aDiff, bDiff, abDiff := 0, 0, 0, 0
	for w := 0; w < W; w++ {
		var sq float32
		for l := 0; l < 32; l++ {
			sq += in[w*32+l]
		}
		s, a, b := o[w*3], o[w*3+1], o[w*3+2]
		if math.Float32bits(s) != math.Float32bits(sq) {
			seqDiff++
		}
		if math.Float32bits(s) != math.Float32bits(a) {
			aDiff++
		}
		if math.Float32bits(s) != math.Float32bits(b) {
			bDiff++
		}
		if math.Float32bits(a) != math.Float32bits(b) {
			abDiff++
		}
	}
	t.Logf("%d simdgroups: simd_sum differs from sequential %d, from xor(16..1) %d, from xor(1..16) %d; the two xor trees differ from each other on %d",
		W, seqDiff, aDiff, bDiff, abDiff)
}

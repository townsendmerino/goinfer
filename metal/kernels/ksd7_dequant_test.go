//go:build darwin

package kernels

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// ksd7Src computes, for every f16 scale (65,536) and nibble (16), the staging dequant the prefill GEMM uses
// (gemm_stage_w: half(float(n-8) * sc)) and KS-D7's half2 form ((nibble pair | 0x64006400) as half2, minus 1032, times the
// scale), each lane of the pair holding the same nibble. out[2*i] is the current form's bits, out[2*i+1] and out[2*i+2] the
// half2 form's two lanes; i = scale*16 + nibble.
const ksd7Src = `
#include <metal_stdlib>
using namespace metal;
kernel void ksd7_dequant(device ushort* out[[buffer(0)]], uint i[[thread_position_in_grid]]) {
    if (i >= 65536u*16u) return;
    uint s = i >> 4, n = i & 15u;
    half sh = as_type<half>((ushort)s);
    float sc = float(sh);
    half cur = half(float(int(n) - 8) * sc);
    uint word = n | (n << 16);
    half2 v = (as_type<half2>((word & 0x000F000Fu) | 0x64006400u) - half2(1032.0h)) * half2(sh);
    out[3*i+0] = as_type<ushort>(cur);
    out[3*i+1] = as_type<ushort>(v.x);
    out[3*i+2] = as_type<ushort>(v.y);
}
`

// TestKSD7_half2DequantExhaustive is KS-D7's first measurement (docs/tasks/task-kernel-survey-2026-10.md): every f16 scale
// times every nibble, the half2 dequant against the current one, bit for bit, compiled as the prefill library is (fast math).
// It reports the differing cases by class; it does not change any kernel.
func TestKSD7_half2DequantExhaustive(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(ksd7Src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "ksd7_dequant")
	if err != nil {
		t.Fatal(err)
	}
	const n = 65536 * 16
	out := NewBufferU16s(d, make([]uint16, 3*n))
	q := d.NewCommandQueue()
	e := q.Begin()
	e.Dispatch(p, n, 256, out)
	e.End()
	got := out.U16s()
	var differ, finiteDiffer, subnormalScale, subnormalResult, lanes int
	firstFinite := -1
	for i := range n {
		cur, a, b := got[3*i], got[3*i+1], got[3*i+2]
		if a != b {
			lanes++
		}
		if a == cur {
			continue
		}
		differ++
		s := uint16(i >> 4)
		f := f16ToF32(s)
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			continue
		}
		finiteDiffer++
		if firstFinite < 0 {
			firstFinite = i
		}
		if s&0x7C00 == 0 && s&0x03FF != 0 {
			subnormalScale++
		}
		if cur&0x7C00 == 0 && cur&0x03FF != 0 {
			subnormalResult++
		}
	}
	fmt.Fprintf(os.Stderr, "[KS-D7] %d cases: %d differ (%d with a finite scale: %d of them a subnormal scale, %d a subnormal current result); the half2 lanes disagree in %d\n",
		n, differ, finiteDiffer, subnormalScale, subnormalResult, lanes)
	if firstFinite >= 0 {
		i := firstFinite
		fmt.Fprintf(os.Stderr, "[KS-D7] first finite case: scale 0x%04x (%g), nibble %d: current 0x%04x, half2 0x%04x\n", i>>4, f16ToF32(uint16(i>>4)), i&15, got[3*i], got[3*i+1])
	}
}

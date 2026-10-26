//go:build darwin

package gpudev

import "testing"

const smokeSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* out    [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}`

// TestMetal_smokeAdd proves the cgo-free Metal path works end to end from aikit:
// reach the device, compile MSL at >=3.1, build a pipeline, upload buffers, dispatch,
// and read the result back off the shared (UMA) buffer.
func TestMetal_smokeAdd(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	t.Logf("device: %s", d.Name())

	lib, err := d.CompileLibrary(smokeSrc, MSL3_1)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "vadd")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	a := NewBufferOf(d, []float32{1, 2, 3, 4, 5})
	b := NewBufferOf(d, []float32{10, 20, 30, 40, 50})
	out := d.NewBufferLen(5)
	q.Run1D(p, 5, 5, a, b, out)

	got := out.Floats()
	want := []float32{11, 22, 33, 44, 55}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	t.Logf("vadd result: %v", got)
}

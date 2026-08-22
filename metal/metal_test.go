//go:build darwin

package metal

import "testing"

// TestLayerA_deviceAndCompiler is the Phase-1 go/no-go on the binding risk: reach
// Metal, and drive the runtime MSL compiler — cgo-free. The bfloat kernel is the
// landmine detector: bfloat is MSL >=3.1 only, so if the LC_BUILD_VERSION landmine
// (risk #7) silently pinned us to 2.4, this FAILS to compile. Compiling it proves
// the languageVersion fix actually works, not just that the option reads back.
func TestLayerA_deviceAndCompiler(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	t.Logf("Metal device (cgo-free, purego): %q", d.Name())

	// A trivial dense-decode-shaped kernel using bfloat — the >=3.1 canary.
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void vadd_bf(device const bfloat* a [[buffer(0)]],
                    device const bfloat* b [[buffer(1)]],
                    device float* out       [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    out[i] = float(a[i]) + float(b[i]);
}`
	if _, err := d.CompileLibrary(src, MSL3_1); err != nil {
		t.Fatalf("compile at MSL 3.1 (bfloat canary): %v", err)
	}
	t.Logf("compiled bfloat MSL kernel at 3.1 — landmine defused (risk #7)")
}

// TestLayerA_vectorAddDispatch completes the binding proof: compile → queue → buffers
// → encode → dispatch → wait → read back, with a CORRECT result. This retires the
// last Layer-A risks (MTLSize struct-by-value, autoreleasepool discipline, the full
// compute-encoder sequence) — all cgo-free.
func TestLayerA_vectorAddDispatch(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* out      [[buffer(2)]],
                 uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile vadd: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "vadd")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const n = 4096
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(2 * i)
	}
	q := d.NewCommandQueue()
	bufA := NewBufferFloats(d, a)
	bufB := NewBufferFloats(d, b)
	bufOut := d.NewBufferLen(n)

	q.Run1D(pipe, n, 256, bufA, bufB, bufOut)

	out := bufOut.Floats()
	for i := range n {
		if want := a[i] + b[i]; out[i] != want {
			t.Fatalf("out[%d] = %v, want %v (GPU compute wrong)", i, out[i], want)
		}
	}
	t.Logf("vector-add %d elems on Metal GPU (cgo-free): out[0]=%v out[%d]=%v — CORRECT",
		n, out[0], n-1, out[n-1])
}

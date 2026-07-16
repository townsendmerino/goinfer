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

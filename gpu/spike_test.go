//go:build gpu

package gpu

import (
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
)

// TestSpike_capabilities is the GPU Stage-1 "binding spike": it reports the
// adapter the WebGPU binding selects and whether cogentcore/webgpu exposes the
// features the W8A8 plan needs — packed int8 dot (dot4I8Packed) and timestamp
// queries — plus the storage-buffer binding limit (the sharding constraint).
// Informational; it never fails (a missing feature is a finding, not a bug).
func TestSpike_capabilities(t *testing.T) {
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	info := ctx.adapter.GetInfo()
	t.Logf("adapter: %q (%s) | backend=%s type=%s | driver=%q",
		info.Name, info.VendorName, info.BackendType, info.AdapterType, info.DriverDescription)

	lim := ctx.adapter.GetLimits().Limits
	t.Logf("maxStorageBufferBindingSize = %d MiB | maxBufferSize = %d MiB | maxComputeWorkgroupStorageSize = %d KiB",
		lim.MaxStorageBufferBindingSize/(1<<20), lim.MaxBufferSize/(1<<20), lim.MaxComputeWorkgroupStorageSize/(1<<10))

	// Timestamp queries (for kernel profiling in the microbenchmark stage).
	t.Logf("timestamp-query feature: adapter=%v device=%v",
		hasFeature(ctx.adapter.EnumerateFeatures(), wgpu.FeatureNameTimestampQuery),
		ctx.device.HasFeature(wgpu.FeatureNameTimestampQuery))

	// dot4I8Packed: try compiling a shader that uses it. Success ⇒ the packed
	// int8 fast path is available; failure ⇒ we ship the unpacked int8 fallback.
	const dotShader = `
@group(0) @binding(0) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = dot4I8Packed(0x01020304u, 0x05060708u);
}`
	sm, derr := ctx.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "dot4I8Packed-probe",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: dotShader},
	})
	if derr != nil {
		t.Logf("dot4I8Packed: NOT supported (%v) — W8A8 must use the unpacked int8 fallback", derr)
	} else {
		sm.Release()
		t.Logf("dot4I8Packed: SUPPORTED — W8A8 fast path available")
	}
}

func hasFeature(fs []wgpu.FeatureName, want wgpu.FeatureName) bool {
	for _, f := range fs {
		if f == want {
			return true
		}
	}
	return false
}

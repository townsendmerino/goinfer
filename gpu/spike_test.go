//go:build gpu

package gpu

import (
	"slices"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// TestSpike_capabilities is the GPU Stage-1 "binding spike": it reports the
// adapter the WebGPU binding selects and whether the wgpu binding exposes the
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
		info.Device, info.Vendor, info.BackendType, info.AdapterType, info.Description)

	lim := ctx.adapter.GetLimits()
	t.Logf("maxStorageBufferBindingSize = %d MiB | maxBufferSize = %d MiB | maxComputeWorkgroupStorageSize = %d KiB",
		lim.MaxStorageBufferBindingSize/(1<<20), lim.MaxBufferSize/(1<<20), lim.MaxComputeWorkgroupStorageSize/(1<<10))

	// Timestamp queries (for kernel profiling in the microbenchmark stage).
	t.Logf("timestamp-query feature: adapter=%v device=%v",
		hasFeature(ctx.adapter.GetFeatures(), wgpu.FeatureNameTimestampQuery),
		ctx.device.HasFeature(wgpu.FeatureNameTimestampQuery))

	// dot4I8Packed: New() already probed this (probeDP4A, gpu.go) and cached it as
	// ctx.hasDP4A — ensureTiled uses it to pick the DP4A tiled-GEMM kernel over the
	// scalar-unpack fallback. Report the cached result rather than re-probing.
	if ctx.hasDP4A {
		t.Logf("dot4I8Packed: SUPPORTED — W8A8 tiled-GEMM fast path available")
	} else {
		t.Logf("dot4I8Packed: NOT supported — W8A8 must use the unpacked int8 fallback")
	}
}

func hasFeature(fs []wgpu.FeatureName, want wgpu.FeatureName) bool {
	return slices.Contains(fs, want)
}

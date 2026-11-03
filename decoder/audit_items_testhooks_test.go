//go:build goinfer_testhooks

package decoder

import "testing"

// TestResidentKV_webgpuSharedLayersOwnNone holds the plan's KV on WebGPU to the caches its build makes (webgpuLayerCaches, which a shared
// layer has none of), on a Gemma 4 with KV-shared layers.
func TestResidentKV_webgpuSharedLayersOwnNone(t *testing.T) {
	m := loadSharedKVFixture(t)
	const ctx = 8
	var built int64
	for l := range m.w.arch.NumLayers {
		for _, b := range m.webgpuLayerCaches(l, ctx, false, false) {
			built += b
		}
	}
	if got := m.ResidentKVBytes("webgpu", ctx, false, false); got != built {
		t.Errorf("priced %d bytes of KV at %d positions, the build makes %d", got, ctx, built)
	}
}

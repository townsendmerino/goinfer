//go:build goinfer_testhooks

package decoder

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
)

// TestResidentKV_webgpuSharedLayersOwnNone holds the plan's KV on WebGPU to the caches its build makes (WebGPULayerCaches, which a shared
// layer has none of), on a Gemma 4 with KV-shared layers.
func TestResidentKV_webgpuSharedLayersOwnNone(t *testing.T) {
	m := loadSharedKVFixture(t)
	const ctx = 8
	var built int64
	for l := range m.w.Arch.NumLayers {
		bufs, kv := memprice.WebGPULayerCaches(m.w, m.LayerBuffers, l, ctx, false, false)
		if !kv {
			continue // recurrent state: not context
		}
		for _, b := range bufs {
			built += b
		}
	}
	if got := m.ResidentKVBytes("webgpu", ctx, false, false); got != built {
		t.Errorf("priced %d bytes of KV at %d positions, the build makes %d", got, ctx, built)
	}
}

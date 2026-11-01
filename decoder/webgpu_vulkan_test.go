//go:build goinfer_testhooks

package decoder

import "testing"

// TestWebGPUDeviceBuffers_sumToTheModel holds webgpuDeviceBuffers to the model's own figures: its matrices are the dense bytes less the
// host-side tables (as cuda prices them) plus the routed experts, and its caches every layer's KV on fixtures with no shared or
// recurrent layer.
func TestWebGPUDeviceBuffers_sumToTheModel(t *testing.T) {
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/glm-tiny"} {
		m, err := Load(dir, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		const ctx = 512
		_, slotKV, weights := m.webgpuDeviceBuffers(ctx, false, false)
		var kv int64
		for _, n := range slotKV {
			kv += n
		}
		if want := m.ResidentDenseWeightBytesFor("cuda") + m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes(); weights != want {
			t.Errorf("%s: the list's matrices %d, the model's device matrices %d", dir, weights, want)
		}
		if want := m.ResidentKVBytes("webgpu", ctx, false, false); kv != want || kv != m.webgpuKVBytes(ctx, false, false) {
			t.Errorf("%s: the list's caches %d, ResidentKVBytes %d", dir, kv, want)
		}
		m.Close()
	}
}

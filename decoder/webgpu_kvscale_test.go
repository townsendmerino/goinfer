package decoder

import "testing"

// TestWebGPUKV_int8HoldsOneScaleAPositionAKVHead holds WebGPU's plan at int8 KV to the buffers its build allocates: K and V at one byte an
// element, and a K and a V scale buffer of one f32 a KV head a position (backend/webgpu's kvBytesPerPosition is the same figure). The rate
// Plan reads (kvBytesPerPositionAllLayers) and the per-buffer ledger (webgpuLayerCaches) must each equal it, on a narrow head, a wide
// one, and a family whose recurrent layers hold no K/V (and so no scale).
func TestWebGPUKV_int8HoldsOneScaleAPositionAKVHead(t *testing.T) {
	const ctx = 64
	for _, dir := range []string{"llama-tiny", "gemma4-moe-kv-tiny", "olmo_hybrid-tiny", "qwen3moe-tiny"} {
		t.Run(dir, func(t *testing.T) {
			m, err := Load("../testdata/"+dir, Options{Backend: "cpu", DirectLoad: true})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			_, layers, _, _, _, _, _ := m.Dims()
			var want, ledger int64
			for l := range layers {
				bufs, kv := m.webgpuLayerCaches(l, ctx, false, true)
				if !kv {
					continue
				}
				for _, b := range bufs {
					ledger += b
				}
				if m.w.arch.hasNoAttentionKVAt(l) || m.w.arch.gemma4KVSrcAt(l) != l || m.w.arch.mla != nil {
					continue
				}
				want += 2 * int64(ctx) * (int64(m.w.arch.kvDimAt(l)) + 4*int64(m.KVHeadsAtResident(l)))
			}
			if want == 0 {
				t.Fatal("VOID: no attention layer holds K/V")
			}
			if got := m.kvBytesPerPositionAllLayers(false, true) * ctx; got != want {
				t.Errorf("Plan's int8 rate at %d positions is %d, the geometry holds %d", ctx, got, want)
			}
			if ledger != want {
				t.Errorf("webgpuLayerCaches sums to %d at int8, the geometry holds %d", ledger, want)
			}
		})
	}
}

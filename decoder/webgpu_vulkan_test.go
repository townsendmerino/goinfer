package decoder

import "testing"

// TestWebGPUDeviceBuffers_sumToTheModel holds webgpuDeviceBuffers' context caches (K and V, MLA's latent; not the recurrent state)
// to ResidentKVBytes("webgpu") on a fixture of each mixer family, so the block rounding the plan adds is over the caches it counts.
// The list's matrices are sized as WebGPU uploads them, not as the planner's walk counts them; a WebGPU build holds them to its own
// buffers (backend/webgpu TestWebGPUDeviceBuffers_inTheBuildsOrder).
func TestWebGPUDeviceBuffers_sumToTheModel(t *testing.T) {
	for _, dir := range []string{
		"../testdata/llama-tiny",
		"../testdata/glm-tiny",
		"../testdata/qwen35-tiny",         // DeltaNet and gated attention
		"../testdata/deepseek-tiny",       // MLA
		"../testdata/nemotron-tiny",       // Mamba-2
		"../testdata/lfm2-tiny",           // the short convolution
		"../testdata/bailing_hybrid-tiny", // Kimi Delta Attention
		"../testdata/gemma4-moe-kv-tiny",  // Gemma 4's MoE block
	} {
		m, err := Load(dir, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", dir, err)
		}
		const ctx = 512
		_, slotKV, _ := m.webgpuDeviceBuffers(ctx, false, false)
		var kv int64
		for _, n := range slotKV {
			kv += n
		}
		if want := m.ResidentKVBytes("webgpu", ctx, false, false); kv != want {
			t.Errorf("%s: the list's caches %d, ResidentKVBytes %d", dir, kv, want)
		}
		m.Close()
	}
}

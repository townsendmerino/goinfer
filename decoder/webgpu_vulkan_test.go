package decoder

import "testing"

// TestWebGPUDeviceBuffers_sumToTheModel holds webgpuDeviceBuffers to the planner's own figures on a fixture of each mixer family: the
// list's matrices are the dense bytes less the host-side tables plus the routed experts (residentWeightBytesSplit's walk), and its
// context caches (K and V, MLA's latent; not the recurrent state) are ResidentKVBytes("webgpu"), so the block rounding the plan adds
// is over the buffers it counts.
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
		_, slotKV, weights := m.webgpuDeviceBuffers(ctx, false, false)
		if want := m.ResidentDenseWeightBytesFor("webgpu") + m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes() - m.denseVectorBytes() - 2*m.expertBiasBytes(); weights != want { // a matrix walk: no vectors or biases
			t.Errorf("%s: the list's matrices %d, the planner's walk %d", dir, weights, want)
		}
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

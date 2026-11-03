//go:build goinfer_testhooks

package decoder

import "testing"

// TestWebGPUDeviceBuffers_sumToTheModel holds webgpuDeviceBuffers' matrix walk to the planner's own (residentWeightBytesSplit): on a
// fixture of each mixer family the list's matrices are the dense bytes less the host-side tables (as cuda prices them) plus the routed
// experts. On the attention-only fixtures the list's caches are also ResidentKVBytes("webgpu").
func TestWebGPUDeviceBuffers_sumToTheModel(t *testing.T) {
	for _, c := range []struct {
		dir      string
		attnOnly bool
	}{
		{"../testdata/llama-tiny", true},
		{"../testdata/glm-tiny", true},
		{"../testdata/qwen35-tiny", false},         // DeltaNet and gated attention
		{"../testdata/deepseek-tiny", false},       // MLA
		{"../testdata/nemotron-tiny", false},       // Mamba-2
		{"../testdata/lfm2-tiny", false},           // the short convolution
		{"../testdata/bailing_hybrid-tiny", false}, // Kimi Delta Attention
		{"../testdata/gemma4-moe-kv-tiny", false},  // Gemma 4's MoE block
	} {
		m, err := Load(c.dir, Options{Backend: "cpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("%s: %v", c.dir, err)
		}
		const ctx = 512
		_, slotKV, weights := m.webgpuDeviceBuffers(ctx, false, false)
		if want := m.ResidentDenseWeightBytesFor("cuda") + m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes() - m.denseVectorBytes() - 2*m.expertBiasBytes(); weights != want { // a matrix walk: no vectors or biases
			t.Errorf("%s: the list's matrices %d, the planner's walk %d", c.dir, weights, want)
		}
		if c.attnOnly {
			var kv int64
			for _, n := range slotKV {
				kv += n
			}
			if want := m.ResidentKVBytes("webgpu", ctx, false, false); kv != want {
				t.Errorf("%s: the list's caches %d, ResidentKVBytes %d", c.dir, kv, want)
			}
		}
		m.Close()
	}
}

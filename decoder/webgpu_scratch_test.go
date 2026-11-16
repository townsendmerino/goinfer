package decoder

import "testing"

// TestMM5b_webgpuDecodeScratchNamesNoFamily is G-MS4w (docs/tasks/task-forward-and-memory-2026-11.md, "MM5b, WebGPU's decode scratch:
// registration"): the figure is priced from the operations' per-row declarations, so its pricing names no family, mixer or
// FFN form. denseAttnLayer, which names the operation kinds of the one layer shape the figure is derived for, is the coverage
// test and not the pricing; it is not scanned.
func TestMM5b_webgpuDecodeScratchNamesNoFamily(t *testing.T) {
	banned := []string{"ResidentLayerOps", "Mixer", "KVSrcAtResident", "HeadDimAtResident", "KVHeadsAtResident",
		"Qwen35", "Nemotron", "MLA", "Gemma4", "GptOss", "DeltaNet", "Mamba", "KDA", "ShortConv", "PLE", "MoEResident", "QKNormWhole",
		"NonGatedMLP", "OpAttention", "OpGatedAttn", "Ffn", ".qwen35", ".lfm2", ".kda", ".granite", ".nemotron", ".gemma4", ".mla"}
	scanPricingBodies(t, map[string][]string{"webgpu_scratch.go": {"WebGPUDecodeScratch", "WebGPUDecodeScratchBytes"}}, banned,
		"WebGPU's decode scratch is priced from the operations' per-row declarations, not the family")
}

// TestWebGPUDecodeScratch_handTable is G-MS3w: the figure for llama-tiny, written as arithmetic on its config (hidden 64, 4 layers, 4 heads
// of 16 over 2 KV heads, FFN 128, vocabulary 256), term by term, at each KV precision; and no figure where a layer is not the plain shape.
func TestWebGPUDecodeScratch_handTable(t *testing.T) {
	const H, L, V, qW, kvW, ffn = 64, 4, 256, 64, 32, 128
	rmsq := int64(H + 4 + 16)   // the int8 row, its scale, its uniform: two a layer and the final norm
	quant := int64(qW + 4 + 16) // the context's
	act := int64(ffn + 4 + 16)  // the gated activation's
	gemv := int64(L*(4*(qW+kvW+kvW+ffn+ffn)+5*16) + 4*V + 16)
	base := map[string]int64{
		"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V,
		"rmsQuant": (2*L + 1) * rmsq, "quant": L * quant, "gemv": gemv, "gemvAdd": L * 2 * 16, "actQuantPad": L * act,
		"attnLayer": L * (4*qW + 16), "geomFor": 16 + 32,
	}
	for _, c := range []struct {
		name string
		o    Options
		rope map[string]int64
		geom int64
	}{
		{"f32", Options{}, map[string]int64{"qkvFinUniFor": 32}, 48},
		{"f16", Options{KVPrecision: "f16"}, map[string]int64{"ropeQUniFor": 32, "ropeKUniFor": 32}, 48},
		{"i8", Options{KVPrecision: "i8"}, map[string]int64{"ropeQUniFor": 32, "ropeKUniFor": 32}, 80},
	} {
		m, err := Load("../testdata/llama-tiny", c.o)
		if err != nil {
			t.Fatal(err)
		}
		terms, ok := m.WebGPUDecodeScratch()
		if !ok {
			t.Fatalf("%s: no figure for llama-tiny", c.name)
		}
		want := map[string]int64{}
		for k, v := range base {
			want[k] = v
		}
		want["geomFor"] = c.geom
		for k, v := range c.rope {
			want[k] = v
		}
		got := map[string]int64{}
		for _, tm := range terms {
			got[tm.Role] += tm.Bytes
		}
		for k, v := range want {
			if got[k] != v {
				t.Errorf("%s: %s = %d B, hand arithmetic says %d B", c.name, k, got[k], v)
			}
		}
		if len(got) != len(want) {
			t.Errorf("%s: %d terms, the hand table has %d", c.name, len(got), len(want))
		}
		m.Close()
	}
	m, err := Load("../testdata/qwen35-tiny", Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.WebGPUDecodeScratch(); ok {
		t.Error("qwen35-tiny has delta-net layers, whose scratch is not derived: the plan must give no figure")
	}
}

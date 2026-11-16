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

// webgpuScratchHand is the figure for a plain-attention model written as arithmetic on its config: hidden h, l layers, vocabulary v, the
// query, key and FFN widths, a QK-norm per layer, a norm on each sublayer's output, at a KV precision, and the q/k/v bias an int4 or int8 weight carries.
func webgpuScratchHand(h, l, v, qW, kvW, ffn int, qkNorm, post bool, kv, bias string) map[string]int64 {
	pk := func(k int) int64 { return int64((k + 31) &^ 31) }
	H, L, V := int64(h), int64(l), int64(v)
	want := map[string]int64{
		"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V,
		"rmsQuant":    (2*L + 1) * (pk(h) + 4 + 16),
		"quant":       L * (pk(qW) + 4 + 16),
		"gemv":        L*(4*int64(qW+kvW+kvW+ffn+ffn)+5*16) + 4*V + 16,
		"gemvAdd":     L * 2 * 16,
		"actQuantPad": L * (pk(ffn) + 4 + 16),
		"attnLayer":   L * (4*int64(qW) + 16),
		"geomFor":     16 + 32,
	}
	if kv == "i8" {
		want["geomFor"] += 32
	}
	if kv == "f16" || kv == "i8" {
		want["ropeQUniFor"], want["ropeKUniFor"] = 32, 32
	} else {
		want["qkvFinUniFor"] = 32
	}
	switch bias { // q, k and v's bias: a separate add each on an int4 weight, folded into the projection on an int8 one
	case "int4":
		want["biasAdd"] = L * 3 * 16
	case "int8":
		q3 := L * (4*int64(qW+2*kvW) + 3*16)
		want["gemv"] -= q3
		want["gemvBias"] = q3
	}
	if qkNorm {
		want["qkNorm"] = L * 2 * 16
	}
	if post { // both sublayers: a bare gemv, a norm and a separate residual add in place of the fused projection
		delete(want, "gemvAdd") // no sublayer's output projection is fused
		want["gemv"] += L * 2 * (4*H + 16)
		want["rmsnormF32"] = L * 2 * (4*H + 16)
		want["biasAdd"] += L * 2 * 16
	}
	return want
}

// TestWebGPUDecodeScratch_handTable is G-MS3w: the figure for llama-tiny at each KV precision, qwen3asr-tiny (a QK-norm) and glm-ocr-tiny
// (a norm on each sublayer's output), term by term as arithmetic on the config; and no figure where a layer is not a plain shape.
func TestWebGPUDecodeScratch_handTable(t *testing.T) {
	for _, c := range []struct {
		name, fx string
		o        Options
		want     map[string]int64
	}{
		{"llama f32", "llama-tiny", Options{}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "f32", "")},
		{"llama f16", "llama-tiny", Options{KVPrecision: "f16"}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "f16", "")},
		{"llama i8", "llama-tiny", Options{KVPrecision: "i8"}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "i8", "")},
		{"qwen3asr (qk-norm)", "qwen3asr-tiny", Options{}, webgpuScratchHand(24, 2, 160, 24, 12, 48, true, false, "f32", "")},
		{"qwen25vl int4 (bias)", "qwen25vl-tiny", Options{Quant: "int4"}, webgpuScratchHand(64, 2, 300, 64, 32, 128, false, false, "f32", "int4")},
		{"qwen25vl int8 (bias)", "qwen25vl-tiny", Options{Quant: "int8int8"}, webgpuScratchHand(64, 2, 300, 64, 32, 128, false, false, "f32", "int8")},
		{"glm-ocr (post norms)", "glm-ocr-tiny", Options{}, webgpuScratchHand(48, 3, 160, 128, 64, 144, false, true, "f32", "")},
	} {
		m, err := Load("../testdata/"+c.fx, c.o)
		if err != nil {
			t.Fatal(err)
		}
		terms, ok := m.WebGPUDecodeScratch()
		if !ok {
			t.Fatalf("%s: no figure", c.name)
		}
		got := map[string]int64{}
		for _, tm := range terms {
			got[tm.Role] += tm.Bytes
		}
		for k, v := range c.want {
			if got[k] != v {
				t.Errorf("%s: %s = %d B, hand arithmetic says %d B", c.name, k, got[k], v)
			}
		}
		if len(got) != len(c.want) {
			t.Errorf("%s: %d terms, the hand table has %d", c.name, len(got), len(c.want))
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

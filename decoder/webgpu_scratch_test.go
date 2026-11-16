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
	scanPricingBodies(t, map[string][]string{"webgpu_scratch.go": {"WebGPUDecodeScratch", "WebGPUDecodeScratchBytes", "webgpuDecodeScratch", "WebGPUDecodeScratchWhy"}}, banned,
		"WebGPU's decode scratch is priced from the operations' per-row declarations, not the family")
}

// webgpuScratchHand is the figure for a plain-attention model written as arithmetic on its config: hidden h, l layers, vocabulary v, the
// query, key and FFN widths, a QK-norm per layer, a norm on each sublayer's output, at a KV precision, and the q/k/v bias an int4 or int8 weight carries.
func webgpuScratchHand(h, l, v, qW, kvW, ffn int, qkNorm, post bool, kv, bias string, window, noPre bool) map[string]int64 {
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
	if kv == "i8" || window { // the int8 V store's uniform; a sliding window's second attention uniform
		want["geomFor"] += 32
	}
	if noPre { // norms after the sublayer only: the raw residual is quantized in place of each pre-norm's fused norm
		want["rmsQuant"] = pk(h) + 20
		want["quant"] += 2 * L * (pk(h) + 4 + 16)
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

// webgpuScratchHandMoE is the figure for an f32-KV model with no bias or post norms, layer by layer: each layer's attention, then its FFN, dense
// (ffn[l] > 0, that width) or routed (ffn[l] == 0, with nE experts of width inter, top k, and a shared expert of width shared, 0 for none,
// sigmoid-gated when gated).
func webgpuScratchHandMoE(h, v, qW, kvW int, ffn []int, nE, k, inter, shared int, gated, qkNorm bool) map[string]int64 {
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	H, V := int64(h), int64(v)
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "geomFor": 16 + 32, "qkvFinUniFor": 32,
		"rmsQuant": pk(h) + 20, "gemv": 4*V + 16}
	add := func(role string, n int64) { want[role] += n }
	anyMoE := false
	for _, f := range ffn {
		add("rmsQuant", 2*(pk(h)+20))
		add("gemv", 4*int64(qW+2*kvW)+3*16)
		add("quant", pk(qW)+20)
		add("attnLayer", 4*int64(qW)+16)
		add("gemvAdd", 16) // the output projection
		if qkNorm {
			add("qkNorm", 2*16)
		}
		if f > 0 {
			add("gemv", 4*int64(2*f)+2*16)
			add("actQuantPad", pk(f)+20)
			add("gemvAdd", 16)
			continue
		}
		anyMoE = true
		add("gemv", 4*int64(nE)+16)                  // the router's logits
		add("moeFFN", 4*int64(2*k)+4*int64(2*inter)) // the selection rows, the experts' gate and up rows
		add("moeRoute", 32)                          // the router's uniform
		add("moeExpert", 3*int64(k)*16)              // gate, up and down, a uniform each, per slot
		add("swigluQuant", int64(k)*(pk(inter)+20))  // per slot: the int8 row, its scale, its uniform
		if shared > 0 {
			add("gemv", 4*int64(2*shared)+2*16)
			add("swigluQuant", pk(shared)+20)
			if gated {
				add("gemv", 4*H+16+4+16)
				add("sharedGatedCombine", 16)
			} else {
				add("gemvAdd", 16)
			}
		}
	}
	if anyMoE {
		want["moeZeroBias"] = 4 * int64(nE)
	}
	return want
}

// webgpuScratchHandQwen35Tiny is qwen35-tiny's figure from its config (layer_types: three linear-attention layers then a full-attention one;
// hidden 64, vocabulary 256; 2 key heads of 16 and 4 value heads of 16; 4 query heads of 16 over 2 KV heads, with an output gate and a
// QK-norm; every FFN routed: 4 experts, top 2, width 32, and a sigmoid-gated shared expert of width 32).
func webgpuScratchHandQwen35Tiny() map[string]int64 {
	const H, V, L, nE, K, inter, shared = 64, 256, 4, 4, 2, 32, 32
	pk := func(n int) int64 { return int64((n + 31) &^ 31) }
	keyDim, valDim := 2*16, 4*16
	conv := 2*keyDim + valDim // the mixed projection's rows
	qW, kvW := 4*16, 2*16
	want := map[string]int64{"build": 4 + 4*256, "residual": 4 * H, "logitsStaging": 4 * V, "geomFor": 16 + 32, "qkvFinUniFor": 32,
		"rmsQuant": pk(H) + 20, "gemv": 4*V + 16, "deltaSetup": 16 + 16 + 32 + 32 + 32, "moeZeroBias": 4 * nE}
	add := func(role string, n int64) { want[role] += n }
	for l := 0; l < L; l++ {
		add("rmsQuant", 2*(pk(H)+20))
		add("gemvAdd", 16)
		if l < 3 { // linear attention: the mixed, beta, alpha and z projections; the conv, q, k, head-parameter, core and gated rows
			add("gemv", 4*int64(conv+4+4+valDim)+4*16)
			add("deltaLayer", 4*int64(conv+2*keyDim+2*4+valDim+valDim))
			add("quant", pk(valDim)+20)
		} else { // full attention: the wide query projection, k, v, the split, the gate, the QK-norm, the context
			add("gemv", 4*int64(2*qW+2*kvW)+3*16)
			add("qSplit", 4*2*int64(qW)+16)
			add("attnGate", 16)
			add("qkNorm", 2*16)
			add("attnLayer", 4*int64(qW)+16)
			add("quant", pk(qW)+20)
		}
		add("gemv", 4*int64(nE)+16)
		add("moeFFN", 4*2*K+4*2*inter)
		add("moeRoute", 32)
		add("moeExpert", 3*K*16)
		add("swigluQuant", int64(K)*(pk(inter)+20)+pk(shared)+20)
		add("gemv", 4*2*shared+2*16+4*H+16+4+16)
		add("sharedGatedCombine", 16)
	}
	return want
}

func withTerm(m map[string]int64, role string, b int64) map[string]int64 {
	m[role] = b
	return m
}

// TestWebGPUDecodeScratch_handTable is G-MS3w: the figure for llama-tiny at each KV precision, qwen3asr-tiny (a QK-norm) and glm-ocr-tiny
// (a norm on each sublayer's output), term by term as arithmetic on the config; and no figure where a layer is not a plain shape.
func TestWebGPUDecodeScratch_handTable(t *testing.T) {
	for _, c := range []struct {
		name, fx string
		o        Options
		want     map[string]int64
	}{
		{"llama f32", "llama-tiny", Options{}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "f32", "", false, false)},
		{"llama f16", "llama-tiny", Options{KVPrecision: "f16"}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "f16", "", false, false)},
		{"llama i8", "llama-tiny", Options{KVPrecision: "i8"}, webgpuScratchHand(64, 4, 256, 64, 32, 128, false, false, "i8", "", false, false)},
		{"qwen3asr (qk-norm)", "qwen3asr-tiny", Options{}, webgpuScratchHand(24, 2, 160, 24, 12, 48, true, false, "f32", "", false, false)},
		{"qwen25vl int4 (bias)", "qwen25vl-tiny", Options{Quant: "int4"}, webgpuScratchHand(64, 2, 300, 64, 32, 128, false, false, "f32", "int4", false, false)},
		{"qwen25vl int8 (bias)", "qwen25vl-tiny", Options{Quant: "int8int8"}, webgpuScratchHand(64, 2, 300, 64, 32, 128, false, false, "f32", "int8", false, false)},
		{"olmo3 (window, qk-norm, post-only norms)", "olmo3-tiny", Options{}, withTerm(webgpuScratchHand(64, 4, 512, 64, 64, 128, true, true, "f32", "", true, true), "qkvFinUniFor", 2*32)}, // two rope scales: the sliding layers' and the full ones'
		{"mistral window", "mistral-tiny-window", Options{}, webgpuScratchHand(128, 3, 128, 128, 64, 256, false, false, "f32", "", true, false)},
		{"mixtral (routed FFN)", "mixtral-tiny", Options{}, webgpuScratchHandMoE(64, 512, 64, 64, []int{0, 0, 0, 0}, 8, 2, 128, 0, false, false)},
		{"qwen3moe (routed FFN, qk-norm)", "qwen3moe-tiny", Options{}, webgpuScratchHandMoE(64, 512, 64, 16, []int{0, 0, 0, 0}, 8, 2, 32, 0, false, true)},
		{"glm (dense prefix, ungated shared expert, qk-norm)", "glm-tiny", Options{}, webgpuScratchHandMoE(64, 256, 64, 32, []int{128, 0, 0}, 8, 2, 32, 32, false, true)},
		{"qwen35 (delta-net, gated attention, routed FFN)", "qwen35-tiny", Options{}, webgpuScratchHandQwen35Tiny()},
		{"glm-ocr (post norms)", "glm-ocr-tiny", Options{}, webgpuScratchHand(48, 3, 160, 128, 64, 144, false, true, "f32", "", false, false)},
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
	m, err := Load("../testdata/nemotron-tiny", Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.WebGPUDecodeScratch(); ok {
		t.Error("nemotron-tiny has Mamba-2 layers, whose scratch is not derived: the plan must give no figure")
	}
}

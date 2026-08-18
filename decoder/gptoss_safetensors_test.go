//go:build realckpt

// gpt-oss safetensors loader gate — diffed against the ALREADY-T3-VALIDATED GGUF path.
//
// This family is the rare case where a new loader can be checked against a validated
// reader of the SAME MODEL rather than needing a fresh HF oracle. That matters here more
// than usual, because the two on-disk layouts disagree in ways nothing else would catch:
//
//   - MXFP4 nibbles are SEQUENTIAL in safetensors (byte j = elements 2j, 2j+1) but j/j+16
//     in GGML. Measured: cosine 1.000000 vs 0.081.
//   - gate_up_proj is INTERLEAVED (row 2k = gate k, row 2k+1 = UP k), where llama.cpp's
//     converter has already split them into ffn_gate_exps / ffn_up_exps.
//
// Both mistakes produce correct shapes, finite values and plausible magnitudes. Only a
// comparison against known-good weights separates them from success.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_GPTOSS_ST=~/models/gpt-oss-20b-hf \
//	  GOINFER_GPTOSS_GGUF=~/models/gpt-oss-20b-MXFP4.gguf \
//	  go test -tags realckpt ./decoder/ -run TestGptOssSafetensors -v -timeout 120m
package decoder

import (
	"testing"
)

func TestGptOssSafetensors_vsGGUF(t *testing.T) {
	requireHeavyModel(t)
	stDir := assetPath(t, "GOINFER_GPTOSS_ST")
	ggufPath := assetPath(t, "GOINFER_GPTOSS_GGUF")

	// int8 on both sides: the comparison is loader-vs-loader, so the quantization must be
	// identical or the diff measures the quantizer instead.
	ms, err := Load(stDir, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load safetensors: %v", err)
	}
	defer ms.Close()

	a := ms.w.arch
	if a.Name != "gpt-oss" || a.gptoss == nil {
		t.Fatalf("arch = %q (gptoss=%v), want gpt-oss", a.Name, a.gptoss != nil)
	}
	if a.NumLayers != 24 || a.HiddenDim != 2880 || a.NumHeads != 64 || a.NumKVHeads != 8 {
		t.Errorf("geometry = %dL/%dh/%dq/%dkv, want 24/2880/64/8",
			a.NumLayers, a.HiddenDim, a.NumHeads, a.NumKVHeads)
	}
	if a.MoE == nil || a.MoE.NumExperts != 32 || a.MoE.TopK != 4 {
		t.Errorf("MoE = %v, want 32 experts top-4", a.MoE)
	}
	// Per-head sinks and per-expert biases must both have loaded — they are what the GGUF
	// path also carries, and a nil here would silently change the forward.
	if got := len(ms.w.Layers[0].AttnSinks); got != a.NumHeads {
		t.Errorf("layer 0 sinks len = %d, want %d (one per head)", got, a.NumHeads)
	}
	if got := len(ms.w.Layers[0].Experts[0].GateBias); got != a.MoE.IntermediateDim {
		t.Errorf("expert 0 gate bias len = %d, want %d", got, a.MoE.IntermediateDim)
	}
	if got := len(ms.w.Layers[0].RouterBias); got != a.MoE.NumExperts {
		t.Errorf("router bias len = %d, want %d", got, a.MoE.NumExperts)
	}

	mg, err := Load(ggufPath, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load gguf: %v", err)
	}
	defer mg.Close()

	// Same prompt through both readers. The forward, the quantizer and the architecture are
	// identical; only the on-disk layout differs, so agreement is a statement about the
	// loader alone.
	prompt := []int{15496, 11, 616, 1438, 318}
	fwd := func(m *Model) []float32 {
		c := m.NewCache(len(prompt) + 1)
		var lg []float32
		var e error
		for _, id := range prompt {
			if lg, e = m.forward(id, c); e != nil {
				t.Fatalf("forward: %v", e)
			}
		}
		return lg
	}
	ls, lg := fwd(ms), fwd(mg)
	cos := logitCosine(ls, lg)
	as, ag := argmax(ls), argmax(lg)
	t.Logf("gpt-oss safetensors vs GGUF: argmax %d vs %d | logit cosine %.6f", as, ag, cos)
	if as != ag {
		t.Errorf("argmax differs: safetensors=%d gguf=%d — the loaders disagree about the weights", as, ag)
	}
	// Not bit-identical by construction: the two files quantize from independently-rounded
	// MXFP4 sources through the same int8 path, so tiny differences are expected. A wrong
	// nibble order or a mis-split gate/up does not land near 1.0 — it lands near zero.
	if cos < 0.999 {
		t.Errorf("logit cosine %.6f < 0.999 — too far apart for the same weights read two ways", cos)
	}
}

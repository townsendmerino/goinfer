package decoder

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// TestGemma4Config_realGGUF runs the real E2B GGUF through ggufConfig +
// resolveArchitecture (the Increment-1 config/descriptor path) and asserts the
// parsed descriptor matches the verified metadata. Weight loading is guarded off
// (the next increment), so this stops at the descriptor. Skips without the asset
// (set the file at ~/models/gemma-4-E2B_q4_0-it.gguf).
func TestGemma4Config_realGGUF(t *testing.T) {
	path := os.Getenv("HOME") + "/models/gemma-4-E2B_q4_0-it.gguf"
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		t.Skipf("no E2B gguf (%v)", err)
	}
	defer g.Close()

	cfg, err := ggufConfig(g)
	if err != nil {
		t.Fatalf("ggufConfig: %v", err)
	}
	if cfg.ModelType != "gemma4" {
		t.Fatalf("ModelType = %q, want gemma4", cfg.ModelType)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}

	eq := func(name string, got, want any) {
		if got != want {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	eq("Name", arch.Name, "gemma4")
	eq("HiddenDim", arch.HiddenDim, 1536)
	eq("NumLayers", arch.NumLayers, 35)
	eq("NumHeads", arch.NumHeads, 8)
	eq("NumKVHeads", arch.NumKVHeads, 1)
	eq("HeadDim(local)", arch.HeadDim, 256)
	eq("RMSAddOne", arch.RMSAddOne, false)
	eq("AttnScale", arch.AttnScale, 1.0)
	eq("NormPlacement", arch.NormPlacement, NormSandwich4)
	eq("Act", arch.Act, ActGeluTanh)
	eq("QKNorm", arch.QKNorm, true)
	eq("TiedLMHead", arch.TiedLMHead, true)
	eq("FinalLogitSoftcap", arch.FinalLogitSoftcap, float64(30))
	eq("RoPELocalBase", arch.RoPELocalBase, float64(10000))
	eq("RoPEGlobalBase", arch.RoPEGlobalBase, float64(1000000))
	if want := math.Sqrt(1536); arch.EmbedScale != want {
		t.Errorf("EmbedScale = %v, want %v", arch.EmbedScale, want)
	}

	g4 := arch.gemma4
	if g4 == nil {
		t.Fatal("arch.gemma4 nil")
	}
	eq("GlobalHeadDim", g4.GlobalHeadDim, 512)
	eq("GlobalRotaryDim", g4.GlobalRotaryDim, 128) // 0.25 * 512
	eq("SharedKVLayers", g4.SharedKVLayers, 20)
	eq("HiddenSizePerLayerInput", g4.HiddenSizePerLayerInput, 256)
	eq("VocabSizePerLayerInput", g4.VocabSizePerLayerInput, 262144)
	eq("KVShared(E2B=false)", g4.KVShared, false)

	// Variable FFN: 6144 for the first 15 layers, 12288 thereafter.
	if len(g4.FFNPerLayer) != 35 {
		t.Fatalf("FFNPerLayer len = %d, want 35", len(g4.FFNPerLayer))
	}
	eq("ffnAt(0)", arch.ffnAt(0), 6144)
	eq("ffnAt(14)", arch.ffnAt(14), 6144)
	eq("ffnAt(15)", arch.ffnAt(15), 12288)
	eq("ffnAt(34)", arch.ffnAt(34), 12288)

	// 4 sliding : 1 global (global at i where (i+1)%5==0).
	eq("isGlobal(0)", arch.isGlobalLayer(0), false)
	eq("isGlobal(4)", arch.isGlobalLayer(4), true)
	eq("isGlobal(34)", arch.isGlobalLayer(34), true)
	// Per-layer head_dim follows the pattern.
	eq("headDimAt(0)", arch.headDimAt(0), 256)
	eq("headDimAt(4)", arch.headDimAt(4), 512)
}

// TestGemma4Load is Increment 2: the full weight load. It exercises the gemma4
// loader against the real E2B GGUF — per-layer head_dim/FFN, the KV-shared tail
// (layers ≥ 15 carry no k/v), and the PLE tensors. int4 keeps it light on 16 GB.
// (Forward pass is Increment 3, so this checks the bundle, not generation.)
func TestGemma4Load(t *testing.T) {
	path := os.Getenv("HOME") + "/models/gemma-4-E2B_q4_0-it.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no E2B gguf (%v)", err)
	}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	w := m.w
	if len(w.Layers) != 35 {
		t.Fatalf("layers = %d, want 35", len(w.Layers))
	}
	// Layer 0: sliding, owns its KV; has the PLE branch + a layer scalar.
	if w.Layers[0].KVShared {
		t.Error("layer 0 should not be KV-shared")
	}
	if w.Layers[0].KProj.Rows() == 0 || w.Layers[0].VProj.Rows() == 0 {
		t.Error("layer 0 missing K/V projection")
	}
	if w.Layers[0].PLEGate.Rows() == 0 || w.Layers[0].PLEProj.Rows() == 0 {
		t.Error("layer 0 missing PLE gate/proj")
	}
	if w.Layers[0].PostPLENorm == nil || w.Layers[0].LayerScalar == 0 {
		t.Error("layer 0 missing PLE norm / layer scalar")
	}
	// Layer 20 (≥ 35−20 = 15): KV-shared → no own k/v.
	if !w.Layers[20].KVShared {
		t.Error("layer 20 should be KV-shared")
	}
	if w.Layers[20].KProj.Rows() != 0 || w.Layers[20].VProj.Rows() != 0 {
		t.Error("layer 20 (KV-shared) should carry no K/V")
	}
	if w.Layers[20].QProj.Rows() == 0 {
		t.Error("layer 20 still needs its own Q")
	}
	// Model-level PLE inputs.
	if w.PerLayerTokenEmbed.Rows() == 0 || w.PerLayerModelProj.Rows() == 0 {
		t.Error("missing model-level PLE tensors")
	}
	if len(w.PerLayerProjNorm) != 256 {
		t.Errorf("PerLayerProjNorm len = %d, want 256", len(w.PerLayerProjNorm))
	}
	t.Logf("loaded E2B int4: 35 layers; layer0 KProj rows=%d PLEGate rows=%d scalar=%.3f; layer20 KVShared=%v",
		w.Layers[0].KProj.Rows(), w.Layers[0].PLEGate.Rows(), w.Layers[0].LayerScalar, w.Layers[20].KVShared)
}

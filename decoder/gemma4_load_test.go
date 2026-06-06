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

package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// The real Mellum2.1 config.json and generation_config.json (JetBrains/Mellum2.1-12B-A2.5B-Thinking @ 92ddae9f, 2026-10-09), no weights.
// 2.1 differs from 2.0's config in two keys: eos_token_id 0 -> 28 and a new use_sliding_window:true (which mellum ignores; windowing comes
// from layer_types). The architecture must resolve as before and the stop id must be <|im_end|>.
func TestMellum21_configResolves(t *testing.T) {
	dir := filepath.Join("..", "testdata", "mellum21-config")
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	global := 0
	for l := 0; l < arch.NumLayers; l++ {
		if arch.layerIsGlobal(l) {
			global++
		}
	}
	type shape struct{ layers, global, window, heads, kv, headDim, experts, topK int }
	got := shape{arch.NumLayers, global, arch.SlidingWindow, arch.NumHeads, arch.NumKVHeads, arch.HeadDim, arch.MoE.NumExperts, arch.MoE.TopK}
	want := shape{28, 7, 1024, 32, 4, 128, 64, 8}
	if got != want {
		t.Errorf("resolved shape %+v, want %+v", got, want)
	}
	if ids := resolveEOSIDs(dir, cfg); !reflect.DeepEqual(ids, []int{28}) {
		t.Errorf("resolved stop ids %v, want [28] (<|im_end|>)", ids)
	}
}

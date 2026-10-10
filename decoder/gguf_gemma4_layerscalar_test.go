package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// loadG4 must assign l.LayerScalar (1, or blk.{i}.layer_output_scale.weight) BEFORE building the layer's gemma4MoEWeights,
// which copies it into its own layerScalar. Every MoE layer's output is multiplied by that copy (forward_gemma4_moe.go: out =
// (h + comb) * layerScalar), so a zero copy zeroes the layer and the argmax falls to token 0 (direct GGUF load of the 26B-A4B
// produced nothing but <pad>: docs/measurements/transcode-streaming-gemma4-2026-09-24.md). The .giw reader and the
// safetensors loader set LayerScalar first; Metal's resident build copies gm.layerScalar (residency.go) and would break the
// same way on a direct GGUF load.
//
// Pinned on the synthetic 26B-shaped GGUF (every layer MoE, layer_output_scale present and non-zero): the MoE copy must equal
// the layer's own scalar, on every layer, and must not be zero.
func TestGemma4GGUF_moeLayerScalarMatchesLayer(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gemma4-26b.gguf")
	if err := os.WriteFile(path, tinyGemma4GGUF(gemma4Variants[0]), 0o644); err != nil {
		t.Fatal(err)
	}
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	cfg, err := ggufConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := buildWeightsFromGGUF(cfg, arch, g, quantInt4, false, false, false, nil, "", nil) // the direct load
	if err != nil {
		t.Fatalf("direct build: %v", err)
	}
	for i := range w.Layers {
		l := &w.Layers[i]
		if l.gemma4moe == nil {
			t.Fatalf("layer %d: no MoE branch — the fixture is not exercising the 26B path", i)
		}
		if l.LayerScalar == 0 {
			t.Fatalf("layer %d: LayerScalar is 0 — the fixture's layer_output_scale did not load", i)
		}
		if got := l.gemma4moe.layerScalar; got != l.LayerScalar {
			t.Errorf("layer %d: MoE branch scales its output by %v, the layer's own scalar is %v — a stale copy "+
				"taken before LayerScalar was assigned zeroes every MoE layer (the direct-load <pad> bug)", i, got, l.LayerScalar)
		}
	}
}

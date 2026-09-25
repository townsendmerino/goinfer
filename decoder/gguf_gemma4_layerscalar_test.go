package decoder

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// A direct GGUF load of the gemma4 26B-A4B produced nothing but <pad> (token 0) on CPU, on HEAD, while the sidecar
// .giw built from the same file generated text (docs/measurements/transcode-streaming-gemma4-2026-09-24.md). The
// cause: loadG4 built the layer's gemma4MoEWeights — which copies l.LayerScalar into its own layerScalar — BEFORE
// it assigned l.LayerScalar (1, or blk.{i}.layer_output_scale.weight). Every MoE layer's output is multiplied by
// that copy (forward_gemma4_moe.go: out = (h + comb) * layerScalar), so it was zeroed, and the argmax fell to
// token 0. The .giw reader and the safetensors loader both set LayerScalar first, which is why only the direct GGUF
// path broke — and why Metal's resident build (residency.go copies gm.layerScalar) would break the same way on a
// direct GGUF load.
//
// Pinned on the synthetic 26B-shaped GGUF (every layer MoE, layer_output_scale present and non-zero): the MoE copy
// must equal the layer's own scalar, on every layer, and must not be zero.
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

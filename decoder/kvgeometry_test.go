package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"slices"
	"testing"
)

func kvGeometryArchs() map[string]*darch.Architecture {
	everySixth := func(i int) bool { return i%6 == 5 }
	return map[string]*darch.Architecture{
		"dense":   {Name: "llama", NumLayers: 4, NumKVHeads: 2, HeadDim: 64},
		"gemma3":  {Name: "gemma3", NumLayers: 12, NumKVHeads: 4, HeadDim: 256, SlidingWindow: 1024, LayerIsGlobal: everySixth},
		"gemma4e": {Name: "gemma4", NumLayers: 12, NumKVHeads: 1, HeadDim: 256, SlidingWindow: 512, LayerIsGlobal: everySixth, Gemma4: &archparts.Gemma4Params{GlobalHeadDim: 512, NumGlobalKVHeads: 2, KVShared: true, SharedKVLayers: 4}},
		"mla":     {Name: "deepseek", NumLayers: 3, NumKVHeads: 8, HeadDim: 64, MLA: &archparts.MLAParams{KVLoRARank: 512, QKRopeHeadDim: 64}},
		"llama4":  {Name: "llama4", NumLayers: 4, NumKVHeads: 8, HeadDim: 128, Llama4: &archparts.Llama4Params{ChunkSize: 8192, UseRope: []bool{true, true, true, false}}},
		"qwen35":  {Name: "qwen3_5", NumLayers: 8, NumKVHeads: 2, HeadDim: 256, LayerIsLinear: func(i int) bool { return i%4 != 3 }},
		"granite": {Name: "granite_hybrid", NumLayers: 6, NumKVHeads: 4, HeadDim: 64, LayerIsMamba: func(i int) bool { return i%3 != 2 }},
		"lfm2":    {Name: "lfm2", NumLayers: 6, NumKVHeads: 8, HeadDim: 64, LayerIsConv: func(i int) bool { return i%2 == 0 }},
		"nemotron": {Name: "nemotron_h", NumLayers: 5, NumKVHeads: 8, HeadDim: 128,
			Nemotron: &archparts.NemotronParams{BlockKind: []uint8{archparts.NemoMamba, archparts.NemoAttn, archparts.NemoMLP, archparts.NemoMamba, archparts.NemoMoE}}},
		"ling": {Name: "bailing_hybrid", NumLayers: 4, NumKVHeads: 4, HeadDim: 64, MLA: &archparts.MLAParams{KVLoRARank: 256, QKRopeHeadDim: 32},
			KDA: &archparts.KDAParams{HeadDim: 64, NumHeads: 4}, LayerIsLinear: func(i int) bool { return i%2 == 0 }},
	}
}

// The geometry and the pricing describe the same layers: a layer is in a group exactly when it is priced, the group's
// row is the priced width, and a window group is exactly the layers whose priced positions stop growing at the window.
func TestKVGeometry_agreesWithThePricing(t *testing.T) {
	const ctx = 1 << 20
	for name, a := range kvGeometryArchs() {
		g := a.KVGeometry()
		seen := map[int]bool{}
		for gi, grp := range g.Groups {
			if len(grp.Layers) == 0 || !slices.IsSorted(grp.Layers) {
				t.Errorf("%s: group %d has layers %v", name, gi, grp.Layers)
			}
			for _, l := range grp.Layers {
				if seen[l] {
					t.Errorf("%s: layer %d is in two groups", name, l)
				}
				seen[l] = true
				if grp.RowWidth != a.KVDimAt(l) {
					t.Errorf("%s: layer %d is in a group of row width %d, priced at %d", name, l, grp.RowWidth, a.KVDimAt(l))
				}
				if windowed := a.KVPositionsAt(l, ctx) < ctx; windowed != (grp.Retention == darch.KVRetainWindow) || (windowed && grp.Span != a.KVPositionsAt(l, ctx)) {
					t.Errorf("%s: layer %d is priced at %d positions of %d and its group is %v(%d)", name, l, a.KVPositionsAt(l, ctx), ctx, grp.Retention, grp.Span)
				}
				if chunk := a.AttnChunkStart(l, ctx+1) > 0; chunk != (grp.Retention == darch.KVRetainChunk) {
					t.Errorf("%s: layer %d: chunked attention %v, group retention %v", name, l, chunk, grp.Retention)
				}
			}
		}
		for l := 0; l < a.NumLayers; l++ {
			_, shared := g.Shared[l]
			if want := a.KVDimAt(l) > 0 && a.Gemma4KVSrcAt(l) == l; seen[l] != want {
				t.Errorf("%s: layer %d: in a group %v, want %v", name, l, seen[l], want)
			}
			if shared != (a.Gemma4KVSrcAt(l) != l) || (shared && g.Shared[l] != a.Gemma4KVSrcAt(l)) {
				t.Errorf("%s: layer %d reads layer %d's KV; the geometry says shared=%v -> %d", name, l, a.Gemma4KVSrcAt(l), shared, g.Shared[l])
			}
			if shared && g.GroupOf(l) != g.GroupOf(g.Shared[l]) {
				t.Errorf("%s: shared layer %d is not in its source's group", name, l)
			}
		}
	}
}

func TestKVGeometry_shapes(t *testing.T) {
	archs := kvGeometryArchs()
	for _, tc := range []struct {
		name   string
		groups int
		states []string
		check  func(darch.KVGeometry) bool
		what   string
	}{
		{"dense", 1, nil, func(g darch.KVGeometry) bool {
			return g.Groups[0].Retention == darch.KVRetainFull && g.Groups[0].RowWidth == 128 && len(g.Groups[0].Layers) == 4
		}, "one full group of four layers"},
		{"gemma3", 2, nil, func(g darch.KVGeometry) bool {
			return g.Groups[0].Retention == darch.KVRetainWindow && g.Groups[0].Span == 1024 && len(g.Groups[0].Layers) == 10 && len(g.Groups[1].Layers) == 2
		}, "ten window layers, two full"},
		{"gemma4e", 2, nil, func(g darch.KVGeometry) bool {
			return len(g.Shared) == 4 && g.Groups[1].KEqualsV && !g.Groups[0].KEqualsV && g.Groups[1].RowWidth == 1024 && g.Groups[0].RowWidth == 256
		}, "the last four layers read an earlier layer; the global group is K = V and wider"},
		{"mla", 1, nil, func(g darch.KVGeometry) bool { return g.Groups[0].Latent && g.Groups[0].RowWidth == 576 }, "one latent group"},
		{"llama4", 2, nil, func(g darch.KVGeometry) bool {
			return g.Groups[0].Retention == darch.KVRetainChunk && g.Groups[0].Span == 8192 && len(g.Groups[0].Layers) == 3 && g.Groups[1].Retention == darch.KVRetainFull
		}, "three chunked layers, one full"},
		{"qwen35", 1, []string{"deltanet"}, func(g darch.KVGeometry) bool { return len(g.Groups[0].Layers) == 2 && len(g.States[0].Layers) == 6 }, "two attention layers, six DeltaNet"},
		{"granite", 1, []string{"mamba2"}, func(g darch.KVGeometry) bool { return len(g.States[0].Layers) == 4 }, "four Mamba-2 layers"},
		{"lfm2", 1, []string{"shortconv"}, func(g darch.KVGeometry) bool { return len(g.States[0].Layers) == 3 }, "three conv layers"},
		{"nemotron", 1, []string{"mamba2"}, func(g darch.KVGeometry) bool {
			return slices.Equal(g.Groups[0].Layers, []int{1}) && slices.Equal(g.States[0].Layers, []int{0, 3})
		}, "one attention layer, two Mamba-2"},
		{"ling", 1, []string{"kda"}, func(g darch.KVGeometry) bool { return g.Groups[0].Latent && len(g.States[0].Layers) == 2 }, "two MLA layers, two KDA"},
	} {
		g := archs[tc.name].KVGeometry()
		var kinds []string
		for _, s := range g.States {
			kinds = append(kinds, s.Kind)
		}
		if len(g.Groups) != tc.groups || !slices.Equal(kinds, tc.states) || !tc.check(g) {
			t.Errorf("%s: want %s; got %d groups %+v, states %+v, shared %v", tc.name, tc.what, len(g.Groups), g.Groups, g.States, g.Shared)
		}
		if g.Recurrent() != (len(tc.states) > 0) {
			t.Errorf("%s: Recurrent() = %v", tc.name, g.Recurrent())
		}
	}
	if g := (*Model)(nil).KVGeometry(); len(g.Groups) != 0 {
		t.Error("a nil Model has a geometry")
	}
}

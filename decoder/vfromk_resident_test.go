package decoder

import "testing"

// VFromKResident feeds every resident backend's K=V decision (Metal, CUDA, WebGPU). A GGUF-derived
// Gemma 4 marks a K=V layer only per layer (VFromK, by attn_v.weight's absence) and leaves the
// config-level attention_k_eq_v false, so the config flag alone mis-reported it and Metal fused an
// empty V matrix. These pin each source on its own, and the cases that must stay false.
func TestVFromKResident_sources(t *testing.T) {
	global5 := func(i int) bool { return i >= 0 && (i+1)%6 == 0 } // gemma4's 5 local : 1 global
	for _, tc := range []struct {
		name string
		arch *Architecture
		vfk  []bool // per-layer LayerWeights.VFromK
		want []bool
	}{
		{"GGUF-derived: per-layer flag only (config flag false)",
			&Architecture{gemma4: &gemma4Params{}, layerIsGlobal: global5},
			[]bool{false, false, false, false, false, true}, []bool{false, false, false, false, false, true}},
		{"config.json-derived: config flag marks the global layer only, per-layer flag unset (older bundle)",
			&Architecture{gemma4: &gemma4Params{KVShared: true}, layerIsGlobal: global5},
			[]bool{false, false, false, false, false, false}, []bool{false, false, false, false, false, true}},
		{"neither source: an ordinary layer stays false",
			&Architecture{gemma4: &gemma4Params{}, layerIsGlobal: global5},
			[]bool{false, false, false, false, false, false}, []bool{false, false, false, false, false, false}},
		{"not gemma4: per-layer flag is ignored",
			&Architecture{},
			[]bool{true, true, true, true, true, true}, []bool{false, false, false, false, false, false}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			layers := make([]LayerWeights, len(tc.vfk))
			for i, v := range tc.vfk {
				layers[i].VFromK = v
			}
			m := &Model{w: &Weights{arch: tc.arch, Layers: layers}}
			for i, want := range tc.want {
				if got := m.VFromKResident(i); got != want {
					t.Errorf("layer %d: VFromKResident = %v, want %v", i, got, want)
				}
			}
			if m.VFromKResident(-1) || m.VFromKResident(len(layers)+3) {
				t.Errorf("an out-of-range layer index must report false, not panic or read past Layers")
			}
		})
	}
}

package decoder

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestBackendModelView_FlintTinyParity(t *testing.T) {
	const ckpt = "../testdata/flint-tiny"
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", ckpt)
	}

	m, err := Load(ckpt, Options{Backend: "cpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	bv := m.BackendView()

	h1, l1, nh1, nkv1, hd1, i1, v1 := bv.Dims()
	h2, l2, nh2, nkv2, hd2, i2, v2 := m.Dims()
	if h1 != h2 || l1 != l2 || nh1 != nh2 || nkv1 != nkv2 || hd1 != hd2 || i1 != i2 || v1 != v2 {
		t.Fatalf("Dims mismatch: view=(%d,%d,%d,%d,%d,%d,%d) model=(%d,%d,%d,%d,%d,%d,%d)",
			h1, l1, nh1, nkv1, hd1, i1, v1, h2, l2, nh2, nkv2, hd2, i2, v2)
	}
	if bv.NumLayers() != l2 {
		t.Fatalf("NumLayers = %d, want %d", bv.NumLayers(), l2)
	}
	if bv.Spec().MoE == nil || m.ResidentSpec().MoE == nil || *bv.Spec().MoE != *m.ResidentSpec().MoE {
		t.Fatalf("Spec().MoE = %+v, want %+v", bv.Spec().MoE, m.ResidentSpec().MoE)
	}
	if bv.Weights() != m.Weights() {
		t.Fatal("Weights pointer mismatch")
	}
	if bv.NormEps() != m.NormEps() {
		t.Fatalf("NormEps = %v, want %v", bv.NormEps(), m.NormEps())
	}
	if bv.AttnScale() != m.AttnScale() {
		t.Fatalf("AttnScale = %v, want %v", bv.AttnScale(), m.AttnScale())
	}
	if bv.SlidingWindow() != m.SlidingWindowResident() {
		t.Fatalf("SlidingWindow = %d, want %d", bv.SlidingWindow(), m.SlidingWindowResident())
	}
	if !slices.Equal(bv.RopeInvFreq(), m.RopeInvFreq()) {
		t.Fatalf("RopeInvFreq mismatch")
	}
	if bv.HasAttnSinks() != m.HasAttnSinksResident() {
		t.Fatalf("HasAttnSinks = %v, want %v", bv.HasAttnSinks(), m.HasAttnSinksResident())
	}
	if bv.ProjsInt4() != m.ResidentProjsInt4() {
		t.Fatalf("ProjsInt4 = %v, want %v", bv.ProjsInt4(), m.ResidentProjsInt4())
	}

	for i := range bv.NumLayers() {
		lv := bv.Layer(i)
		if lv.Index() != i {
			t.Fatalf("layer %d: Index() = %d", i, lv.Index())
		}
		mix1, ffn1 := lv.Ops()
		mix2, ffn2 := m.BackendView().Layer(i).Ops()
		if mix1 != mix2 || ffn1 != ffn2 {
			t.Fatalf("layer %d: Ops() = (%v,%v), want (%v,%v)", i, mix1, ffn1, mix2, ffn2)
		}
		if lv.HeadDim() != m.BackendView().Layer(i).HeadDim() {
			t.Fatalf("layer %d: HeadDim() = %d, want %d", i, lv.HeadDim(), m.BackendView().Layer(i).HeadDim())
		}
		if lv.KVHeads() != m.BackendView().Layer(i).KVHeads() {
			t.Fatalf("layer %d: KVHeads() = %d, want %d", i, lv.KVHeads(), m.BackendView().Layer(i).KVHeads())
		}
		if lv.RotaryDim() != m.BackendView().Layer(i).RotaryDim() {
			t.Fatalf("layer %d: RotaryDim() = %d, want %d", i, lv.RotaryDim(), m.BackendView().Layer(i).RotaryDim())
		}
		if lv.KVSrc() != m.BackendView().Layer(i).KVSrc() {
			t.Fatalf("layer %d: KVSrc() = %d, want %d", i, lv.KVSrc(), m.BackendView().Layer(i).KVSrc())
		}
		if lv.IsLocal() != m.BackendView().Layer(i).IsLocal() {
			t.Fatalf("layer %d: IsLocal() = %v, want %v", i, lv.IsLocal(), m.BackendView().Layer(i).IsLocal())
		}
		if !slices.Equal(lv.RopeInvFreq(), m.BackendView().Layer(i).RopeInvFreq()) {
			t.Fatalf("layer %d: RopeInvFreq() mismatch", i)
		}
		if lv.RopeMscale() != m.BackendView().Layer(i).RopeMscale() {
			t.Fatalf("layer %d: RopeMscale() = %v, want %v", i, lv.RopeMscale(), m.BackendView().Layer(i).RopeMscale())
		}
		if !slices.Equal(lv.AttnSinks(), m.BackendView().Layer(i).AttnSinks()) {
			t.Fatalf("layer %d: AttnSinks() mismatch", i)
		}
	}
}

func TestBackendModelView_HybridLayerOps(t *testing.T) {
	m := &Model{
		w: &Weights{
			arch: &Architecture{
				HiddenDim:  64,
				NumLayers:  4,
				NumHeads:   4,
				NumKVHeads: 2,
				HeadDim:    16,
				nemotron: &nemotronParams{
					blockKind: []uint8{nemoMamba, nemoAttn, nemoMLP, nemoMoE},
				},
			},
			Layers: make([]LayerWeights, 4),
		},
	}

	bv := m.BackendView()
	wantOps := []struct {
		mixer ResidentMixer
		ffn   bool
	}{
		{MixerMamba, false},
		{MixerAttention, false},
		{MixerNone, true},
		{MixerNone, true},
	}
	for i, want := range wantOps {
		lv := bv.Layer(i)
		mix, ffn := lv.Ops()
		if mix != want.mixer || ffn != want.ffn {
			t.Errorf("layer %d: Ops() = (%v,%v), want (%v,%v)", i, mix, ffn, want.mixer, want.ffn)
		}
	}
}

// TestBackendModelView_familyParams holds each per-family geometry method to the structured ResidentSpec it reads: the values for a
// model that carries the subsystem, ok=false and zeros for one that does not.
func TestBackendModelView_familyParams(t *testing.T) {
	moe := &MoEConfig{NumExperts: 8, TopK: 2, IntermediateDim: 24, SharedIntermediateDim: 16, RouterSigmoid: true, NormTopKProb: true, RoutedScale: 2.5, NGroup: 4, TopkGroup: 2}
	m := &Model{w: &Weights{arch: &Architecture{HiddenDim: 64, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 16, MoE: moe}, Layers: make([]LayerWeights, 1)}}
	bv := m.BackendView()

	nE, k, inter, shared, sigmoid, norm, ungated, scale, nGroup, topkGroup, ok := bv.MoEParams()
	if !ok || nE != 8 || k != 2 || inter != 24 || shared != 16 || !sigmoid || !norm || ungated || scale != 2.5 || nGroup != 4 || topkGroup != 2 {
		t.Errorf("MoEParams = (%d,%d,%d,%d,%v,%v,%v,%v,%d,%d,%v), want the config's values with ok", nE, k, inter, shared, sigmoid, norm, ungated, scale, nGroup, topkGroup, ok)
	}
	if _, _, _, _, _, _, _, ok := bv.NemotronParams(); ok {
		t.Error("NemotronParams ok for a model with no Nemotron subsystem")
	}
	if _, _, _, _, _, _, _, _, _, ok := bv.GraniteParams(); ok {
		t.Error("GraniteParams ok for a model with no Granite subsystem")
	}
	if _, _, _, _, _, _, ok := bv.Qwen35Params(); ok {
		t.Error("Qwen35Params ok for a model with no DeltaNet subsystem")
	}
	if _, _, _, _, _, _, _, _, ok := bv.MLAParams(); ok {
		t.Error("MLAParams ok for a model with no MLA subsystem")
	}

	// The forwarders delegate to the view, so what the view returns is what the model returns.
	a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1 := m.MoEResidentParams()
	if a1 != nE || b1 != k || c1 != inter || d1 != shared || e1 != sigmoid || f1 != norm || g1 != ungated || h1 != scale || i1 != nGroup || j1 != topkGroup || k1 != ok {
		t.Error("Model.MoEResidentParams differs from the view's MoEParams")
	}
}

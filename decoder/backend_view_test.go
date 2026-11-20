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

	h2, l2, nh2, nkv2, hd2, i2, v2 := m.Dims()
	if d, want := bv.Dims(), (ResidentDims{Hidden: h2, Layers: l2, Heads: nh2, KVHeads: nkv2, HeadDim: hd2, Inter: i2, Vocab: v2}); d != want {
		t.Fatalf("Dims mismatch: view=%+v model=%+v", d, want)
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
	if bv.SlidingWindow() != m.BackendView().SlidingWindow() {
		t.Fatalf("SlidingWindow = %d, want %d", bv.SlidingWindow(), m.BackendView().SlidingWindow())
	}
	if !slices.Equal(bv.RopeInvFreq(), m.BackendView().RopeInvFreq()) {
		t.Fatalf("RopeInvFreq mismatch")
	}
	if bv.HasAttnSinks() != m.BackendView().HasAttnSinks() {
		t.Fatalf("HasAttnSinks = %v, want %v", bv.HasAttnSinks(), m.BackendView().HasAttnSinks())
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

	want := ResidentMoESpec{NumExperts: 8, TopK: 2, IntermediateDim: 24, SharedIntermediateDim: 16, RouterSigmoid: true,
		NormTopKProb: true, RoutedScale: 2.5, NGroup: 4, TopkGroup: 2}
	if mo, ok := bv.MoE(); !ok || mo != want {
		t.Errorf("MoE = %+v, %v; want %+v with ok", mo, ok, want)
	}
	if _, ok := bv.Nemotron(); ok {
		t.Error("Nemotron ok for a model with no Nemotron subsystem")
	}
	if _, ok := bv.Granite(); ok {
		t.Error("Granite ok for a model with no Granite subsystem")
	}
	if _, ok := bv.Qwen35(); ok {
		t.Error("Qwen35 ok for a model with no DeltaNet subsystem")
	}
	if _, ok := bv.MLA(); ok {
		t.Error("MLA ok for a model with no MLA subsystem")
	}

}

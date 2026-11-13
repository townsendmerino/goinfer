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
	if bv.HasQKNorm() != m.HasQKNorm() {
		t.Fatalf("HasQKNorm = %v, want %v", bv.HasQKNorm(), m.HasQKNorm())
	}
	if bv.QKNormWhole() != m.QKNormWholeResident() {
		t.Fatalf("QKNormWhole = %v, want %v", bv.QKNormWhole(), m.QKNormWholeResident())
	}
	if bv.RotaryDim() != m.RotaryDimResident() {
		t.Fatalf("RotaryDim = %d, want %d", bv.RotaryDim(), m.RotaryDimResident())
	}
	if !slices.Equal(bv.RopeInvFreq(), m.RopeInvFreq()) {
		t.Fatalf("RopeInvFreq mismatch")
	}
	if bv.HasAttnSinks() != m.HasAttnSinksResident() {
		t.Fatalf("HasAttnSinks = %v, want %v", bv.HasAttnSinks(), m.HasAttnSinksResident())
	}
	if bv.MoEExpertKind() != m.MoEExpertKind() {
		t.Fatalf("MoEExpertKind = %q, want %q", bv.MoEExpertKind(), m.MoEExpertKind())
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
		mix2, ffn2 := m.ResidentLayerOps(i)
		if mix1 != mix2 || ffn1 != ffn2 {
			t.Fatalf("layer %d: Ops() = (%v,%v), want (%v,%v)", i, mix1, ffn1, mix2, ffn2)
		}
		if lv.HeadDim() != m.HeadDimAtResident(i) {
			t.Fatalf("layer %d: HeadDim() = %d, want %d", i, lv.HeadDim(), m.HeadDimAtResident(i))
		}
		if lv.KVHeads() != m.KVHeadsAtResident(i) {
			t.Fatalf("layer %d: KVHeads() = %d, want %d", i, lv.KVHeads(), m.KVHeadsAtResident(i))
		}
		if lv.RotaryDim() != m.RotaryDimAtResident(i) {
			t.Fatalf("layer %d: RotaryDim() = %d, want %d", i, lv.RotaryDim(), m.RotaryDimAtResident(i))
		}
		if lv.KVSrc() != m.KVSrcAtResident(i) {
			t.Fatalf("layer %d: KVSrc() = %d, want %d", i, lv.KVSrc(), m.KVSrcAtResident(i))
		}
		if lv.IsLocal() != m.LayerIsLocalResident(i) {
			t.Fatalf("layer %d: IsLocal() = %v, want %v", i, lv.IsLocal(), m.LayerIsLocalResident(i))
		}
		if lv.RopeGlobal() != m.LayerRopeGlobal(i) {
			t.Fatalf("layer %d: RopeGlobal() = %v, want %v", i, lv.RopeGlobal(), m.LayerRopeGlobal(i))
		}
		if !slices.Equal(lv.RopeInvFreq(), m.RopeInvFreqLayerResident(i)) {
			t.Fatalf("layer %d: RopeInvFreq() mismatch", i)
		}
		if lv.RopeMscale() != m.RopeMscaleLayer(i) {
			t.Fatalf("layer %d: RopeMscale() = %v, want %v", i, lv.RopeMscale(), m.RopeMscaleLayer(i))
		}
		if !slices.Equal(lv.AttnSinks(), m.GptOssSinksResident(i)) {
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
		if lv.NemotronBlockKind() != m.NemotronBlockKind(i) {
			t.Errorf("layer %d: NemotronBlockKind() = %d, want %d", i, lv.NemotronBlockKind(), m.NemotronBlockKind(i))
		}
	}
}

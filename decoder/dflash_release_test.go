package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// writeTinyDFlash writes a BF16 DFlash v1 checkpoint (2 layers, hidden 8) to a temp dir and returns the dir.
func writeTinyDFlash(t *testing.T) string {
	t.Helper()
	const h, heads, kv, hd, inter, layers = 8, 2, 1, 4, 16, 2
	dir := t.TempDir()
	cfg := fmt.Sprintf(`{"hidden_size": %d, "num_hidden_layers": %d, "num_attention_heads": %d, "num_key_value_heads": %d,
	  "head_dim": %d, "intermediate_size": %d, "rms_norm_eps": 1e-6, "rope_theta": 10000,
	  "dflash_config": {"block_size": 4, "mask_token_id": 7, "target_layer_ids": [0, 1]}}`, h, layers, heads, kv, hd, inter)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	seed := 0
	bf := func(shape ...int) stTyped {
		n := 1
		for _, d := range shape {
			n *= d
		}
		raw := make([]byte, 2*n)
		for i := range n {
			seed++
			u := linalg.F32ToBF16(float32(seed%13+1) / 16)
			raw[2*i], raw[2*i+1] = byte(u), byte(u>>8)
		}
		return stTyped{"BF16", shape, raw}
	}
	ts := map[string]stTyped{"fc.weight": bf(h, 2*h), "hidden_norm.weight": bf(h), "norm.weight": bf(h)}
	for i := range layers {
		p := fmt.Sprintf("layers.%d.", i)
		ts[p+"self_attn.q_proj.weight"] = bf(heads*hd, h)
		ts[p+"self_attn.k_proj.weight"] = bf(kv*hd, h)
		ts[p+"self_attn.v_proj.weight"] = bf(kv*hd, h)
		ts[p+"self_attn.o_proj.weight"] = bf(h, heads*hd)
		ts[p+"mlp.gate_proj.weight"] = bf(inter, h)
		ts[p+"mlp.up_proj.weight"] = bf(inter, h)
		ts[p+"mlp.down_proj.weight"] = bf(h, inter)
		ts[p+"self_attn.q_norm.weight"] = bf(hd)
		ts[p+"self_attn.k_norm.weight"] = bf(hd)
		ts[p+"input_layernorm.weight"] = bf(h)
		ts[p+"post_attention_layernorm.weight"] = bf(h)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), buildSafetensors(ts), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestDFlashDrafter_releaseHostWeights: after ReleaseHostWeights the trunk's matrices and norms are gone and the mapping is
// closed, what a BlockSpec reads afterwards (BlockSize, MaskTokenID, TargetLayerIDs, DrafterGeometry) is unchanged, and a
// second release or a Close is harmless.
func TestDFlashDrafter_releaseHostWeights(t *testing.T) {
	d, err := LoadDFlashDrafter(writeTinyDFlash(t))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if d.HostWeightsReleased() || d.DrafterFC().Rows() == 0 || d.DrafterLayer(1).Down.Rows() == 0 || d.st == nil {
		t.Fatal("the fixture loaded without weights or without its mapping")
	}
	geo, bs, mask, taps := d.DrafterGeometry(), d.BlockSize(), d.MaskTokenID(), d.TargetLayerIDs()

	if err := d.ReleaseHostWeights(); err != nil {
		t.Fatal(err)
	}
	if !d.HostWeightsReleased() {
		t.Error("HostWeightsReleased is false after the release")
	}
	if d.st != nil {
		t.Error("the mapping is still open after the release of a drafter with no own head")
	}
	if d.DrafterFC().Rows() != 0 || d.DrafterHiddenNorm() != nil || d.DrafterFinalNorm() != nil {
		t.Error("fc or a trunk norm survived the release")
	}
	for i := range geo.Layers {
		l := d.DrafterLayer(i)
		for _, m := range []*linalg.WeightMat{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down} {
			if m.Rows() != 0 {
				t.Errorf("layer %d: a %d-row matrix survived the release", i, m.Rows())
			}
		}
		if l.QNorm != nil || l.KNorm != nil || l.InputNorm != nil || l.PostAttnNorm != nil {
			t.Errorf("layer %d: a norm survived the release", i)
		}
	}
	if g := d.DrafterGeometry(); g.Layers != geo.Layers || g.Hidden != geo.Hidden || g.HeadDim != geo.HeadDim || !slices.Equal(g.InvFreq, geo.InvFreq) {
		t.Errorf("geometry changed: %+v, want %+v", g, geo)
	}
	if d.BlockSize() != bs || d.MaskTokenID() != mask || !slices.Equal(d.TargetLayerIDs(), taps) {
		t.Errorf("block %d mask %d taps %v after the release, want %d %d %v", d.BlockSize(), d.MaskTokenID(), d.TargetLayerIDs(), bs, mask, taps)
	}
	if err := d.ReleaseHostWeights(); err != nil {
		t.Errorf("a second release: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Errorf("Close after the release: %v", err)
	}
}

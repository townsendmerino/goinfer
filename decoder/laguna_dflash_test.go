//go:build realckpt

// Laguna DFlash pairing — poolside/Laguna-XS.2-speculator.dflash.
//
// This is the first VENDOR-BLESSED drafter goinfer has paired with (every prior
// DFlash/DSpark pairing was third-party), and it differs structurally from those:
// it ships its OWN embed_tokens and a REDUCED-vocab lm_head (32000 rows against a
// 100352-token target) plus d2t/t2d, where z-lab's drafters borrow the target's.
// Its config is also a fourth dialect — vLLM "speculators" v0.5, with the layer
// geometry under transformer_layer_config and the taps under
// aux_hidden_state_layer_ids.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_LAGUNA_DFLASH=~/models/laguna-xs2-dflash \
//	  go test -tags realckpt ./decoder/ -run TestLagunaDFlash -v
package decoder

import "testing"

func TestLagunaDFlash_load(t *testing.T) {
	requireHeavyModel(t)
	dir := assetPath(t, "GOINFER_LAGUNA_DFLASH")

	d, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter(%s): %v", dir, err)
	}
	defer d.Close()

	// Geometry comes from transformer_layer_config, not the top level.
	g := d.DrafterGeometry()
	if g.Layers != 5 || g.Hidden != 2048 || g.NumHeads != 16 || g.NumKVHeads != 8 || g.HeadDim != 128 {
		t.Errorf("geometry = %dL/h%d/%dq/%dkv/hd%d, want 5/2048/16/8/128",
			g.Layers, g.Hidden, g.NumHeads, g.NumKVHeads, g.HeadDim)
	}
	if g.Intermediate != 8192 {
		t.Errorf("intermediate = %d, want 8192", g.Intermediate)
	}
	// Five taps, from aux_hidden_state_layer_ids.
	if ids := d.TargetLayerIDs(); len(ids) != 5 || ids[0] != 1 || ids[4] != 39 {
		t.Errorf("taps = %v, want [1 9 17 36 39]", ids)
	}
	if d.BlockSize() != 8 {
		t.Errorf("block_size = %d, want 8", d.BlockSize())
	}
	// mask_token_id is TOP-LEVEL in this dialect. Getting it wrong is silent and
	// expensive: the drafter still runs and still produces lossless output, just
	// badly — P10 measured a known-good pairing fall from 1.60x to 0.66x on exactly
	// this mistake.
	if d.MaskTokenID() != 12 {
		t.Errorf("mask_token_id = %d, want 12", d.MaskTokenID())
	}
	// The reduced-vocab head and its mapping table.
	if !d.HasOwnHead() {
		t.Fatal("HasOwnHead() = false — this drafter ships lm_head + d2t and must not borrow the target's")
	}
	if d.draftVocab != 32000 {
		t.Errorf("draftVocab = %d, want 32000", d.draftVocab)
	}
	if got := d.lmHead.Rows(); got != 32000 {
		t.Errorf("lm_head rows = %d, want 32000", got)
	}
	if got := d.embed.Rows(); got != 100352 {
		t.Errorf("embed_tokens rows = %d, want 100352 (the TARGET vocab)", got)
	}
	if len(d.d2t) != 32000 {
		t.Errorf("d2t len = %d, want 32000", len(d.d2t))
	}
	// d2t must be a strictly-usable mapping into the target vocab; the loader already
	// range-checks it, so this pins that it is not the identity (which would mean the
	// tensor was misread as zeros and every draft id would be wrong-but-in-range).
	nonZero := 0
	for _, v := range d.d2t {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Error("d2t is all zeros — read as an identity mapping, which would silently draft wrong ids")
	}
	t.Logf("laguna DFlash: %d layers, taps %v, block %d, mask %d, draft vocab %d (%d non-identity d2t entries)",
		g.Layers, d.TargetLayerIDs(), d.BlockSize(), d.MaskTokenID(), d.draftVocab, nonZero)
}

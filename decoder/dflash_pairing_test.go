package decoder

import (
	"testing"
)

// TestDFlash_secondPairingLoads is the CONFIG-DIALECT gate, and it exists because the first
// pairing's spelling is not the only one z-lab ships.
//
// z-lab/Qwen3.6-35B-A3B-DFlash differs from Qwen3-4B-DFlash-b16 in every config-driven
// dimension — 8 taps not 5, 6 trunk layers not 5, block 16, hidden 2048, vocab 248320 — and
// in TWO SPELLINGS the loader originally could not read:
//
//   - block_size NESTED in dflash_config (top-level on the 4B)
//   - RoPE as rope_parameters (flat rope_theta on the 4B)
//
// One publisher, two dialects, and the 4B-only loader reported them as "block_size must be
// >= 2, got 0" — a supported pairing looking broken. Third instance of this class in P10
// after granite's flat-only rope and nemotron's hybrid_override_pattern, which is why it is
// gated rather than fixed quietly.
//
// It asserts the SHAPE the loader derived, not just that loading succeeded: a loader that
// silently defaulted a dimension would still "load".
func TestDFlash_secondPairingLoads(t *testing.T) {
	dir := assetPath(t, "GOINFER_DFLASH_35B")
	d, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter(%s): %v", dir, err)
	}
	defer d.Close()

	for _, c := range []struct {
		what      string
		got, want int
	}{
		{"block_size (nested spelling)", d.BlockSize(), 16},
		{"trunk layers", len(d.layers), 6},
		{"hidden", d.hidden, 2048},
		{"taps", len(d.TargetLayerIDs()), 8},
		{"fc input width", d.fc.Cols(), 8 * 2048},
		{"mask token", d.MaskTokenID(), 248077},
	} {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.what, c.got, c.want)
		}
	}

	// The SHARED trunk must run at this geometry — 8 taps and 6 layers are both new.
	be := &cpuBackend{}
	ctx := make([][]float32, 12)
	for i := range ctx {
		ctx[i] = make([]float32, d.fc.Cols())
		for j := range ctx[i] {
			ctx[i][j] = float32((i*7+j)%13) / 13
		}
	}
	fused, err := d.FuseContext(be, ctx)
	if err != nil {
		t.Fatalf("FuseContext: %v", err)
	}
	blk := make([][]float32, d.BlockSize())
	for i := range blk {
		blk[i] = make([]float32, d.hidden)
		for j := range blk[i] {
			blk[i][j] = float32((i*3+j)%11) / 11
		}
	}
	out, err := d.DraftBlock(be, fused, blk)
	if err != nil {
		t.Fatalf("DraftBlock: %v", err)
	}
	if len(out) != d.BlockSize() || len(out[0]) != d.hidden {
		t.Fatalf("trunk out %dx%d, want %dx%d", len(out), len(out[0]), d.BlockSize(), d.hidden)
	}
	t.Logf("second pairing OK: block=%d taps=%v layers=%d hidden=%d heads=%d/%d",
		d.BlockSize(), d.TargetLayerIDs(), len(d.layers), d.hidden, d.nHeads, d.nKV)
}

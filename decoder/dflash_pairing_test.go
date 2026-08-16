package decoder

import (
	"os"
	"testing"
)

// The CONFIG-DIALECT gate, and it exists because one publisher ships more than one spelling.
//
// z-lab's four DFlash drafters differ in every config-driven dimension — taps, trunk depth,
// block width, hidden, vocab, mask id — and in TWO SPELLINGS the 4B-only loader could not read:
//
//   - block_size NESTED in dflash_config (top-level on the other three)
//   - RoPE as rope_parameters (flat rope_theta on the other three)
//
// One publisher, two dialects, and the 4B-only loader reported the 35B as "block_size must be
// >= 2, got 0" — a supported pairing looking broken. Third instance of this class in P10 after
// granite's flat-only rope and nemotron's hybrid_override_pattern, which is why it is gated
// rather than fixed quietly.
//
// TABLE, NOT A ONE-OFF. This started as a single 35B test. It is a table now because the 35B
// turned out to be the OUTLIER — gpt-oss and gemma-4 both spell block_size at top level — and a
// single-case gate would have left that as an assumption. The point of the table is that a
// future drafter is one row, and the row records the dialect it exercises.
//
// Every case asserts the SHAPE the loader derived, not merely that loading succeeded: a loader
// that silently defaulted a dimension would still "load".
func TestDFlash_pairingDialects(t *testing.T) {
	cases := []struct {
		name    string
		env     string
		dialect string
		// expectations, all read from the published config rather than guessed
		block, layers, hidden, taps, mask int
	}{
		{
			name:    "qwen3.6-35b-a3b",
			env:     "GOINFER_DFLASH_35B",
			dialect: "block_size NESTED in dflash_config; rope_parameters (nested)",
			block:   16, layers: 6, hidden: 2048, taps: 8, mask: 248077,
		},
		{
			name:    "gpt-oss-20b",
			env:     "GOINFER_DFLASH_GPTOSS",
			dialect: "block_size top-level; flat rope_theta",
			block:   8, layers: 8, hidden: 2880, taps: 5, mask: 200000,
		},
		{
			name:    "gemma-4-26b-a4b-it",
			env:     "GOINFER_DFLASH_GEMMA4",
			dialect: "block_size top-level; flat rope_theta",
			block:   16, layers: 5, hidden: 2816, taps: 6, mask: 4,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := os.Getenv(c.env)
			if dir == "" {
				dir = assetPath(t, c.env)
			}
			d, err := LoadDFlashDrafter(dir)
			if err != nil {
				t.Fatalf("LoadDFlashDrafter(%s): %v", dir, err)
			}
			defer d.Close()
			t.Logf("dialect: %s", c.dialect)

			for _, k := range []struct {
				what      string
				got, want int
			}{
				{"block_size", d.BlockSize(), c.block},
				{"trunk layers", len(d.layers), c.layers},
				{"hidden", d.hidden, c.hidden},
				{"taps", len(d.TargetLayerIDs()), c.taps},
				{"fc input width", d.fc.Cols(), c.taps * c.hidden},
				{"mask token", d.MaskTokenID(), c.mask},
			} {
				if k.got != k.want {
					t.Errorf("%s = %d, want %d", k.what, k.got, k.want)
				}
			}

			// The SHARED trunk must actually RUN at this geometry — every one of these
			// pairings has a tap count and trunk depth the others do not.
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
			t.Logf("OK: block=%d taps=%v layers=%d hidden=%d heads=%d/%d",
				d.BlockSize(), d.TargetLayerIDs(), len(d.layers), d.hidden, d.nHeads, d.nKV)
		})
	}
}

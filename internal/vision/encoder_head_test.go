package vision

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
)

// TestLoadEncoderHead_matchesLoadEncoder: a head-only encoder holds no blocks until something needs them, and is then
// the same encoder. Its CPU Forward is bit-identical to LoadEncoder's at both precisions (the blocks load on the first
// call), its streamed blocks are exactly Weights()' blocks, and its head is exactly the full encoder's.
func TestLoadEncoderHead_matchesLoadEncoder(t *testing.T) {
	const ckpt = "testdata/siglip-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no siglip-tiny checkpoint (%v)", err)
	}
	raw, err := os.ReadFile("testdata/siglip_vision_golden.json")
	if err != nil {
		t.Skipf("no golden (%v)", err)
	}
	var g struct {
		PixelValues []float32 `json:"pixel_values"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	for _, quant := range []bool{false, true} {
		full, err := LoadEncoder(ckpt, quant)
		if err != nil {
			t.Fatal(err)
		}
		head, err := LoadEncoderHead(ckpt, quant)
		if err != nil {
			t.Fatal(err)
		}
		if head.layers != nil {
			t.Fatalf("quant=%v: LoadEncoderHead loaded the blocks", quant)
		}
		pw, pb, pe, np := head.SiglipHead()
		if !slices.Equal(pw, full.patchW) || !slices.Equal(pb, full.patchB) || !slices.Equal(pe, full.posEmb) || np != full.numPatches ||
			!slices.Equal(head.postLNw, full.postLNw) || !slices.Equal(head.postLNb, full.postLNb) {
			t.Errorf("quant=%v: the head differs from the full encoder's", quant)
		}
		want, err := full.Forward(g.PixelValues)
		if err != nil {
			t.Fatal(err)
		}
		got, err := head.Forward(g.PixelValues)
		if err != nil {
			t.Fatal(err)
		}
		if head.layers == nil {
			t.Errorf("quant=%v: the CPU Forward did not load the blocks", quant)
		}
		for i := range want {
			if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
				t.Fatalf("quant=%v: Forward differs at %d: %v vs %v", quant, i, got[i], want[i])
			}
		}
	}
	full, err := LoadEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := full.Weights()
	if err != nil {
		t.Fatal(err)
	}
	head, err := LoadEncoderHead(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	err = head.ForEachSiglipBlock(func(l int, b SiglipBlock) error {
		f := w.Blocks[l]
		for _, pair := range [][2][]float32{{b.LN1W, f.LN1W}, {b.LN1B, f.LN1B}, {b.LN2W, f.LN2W}, {b.LN2B, f.LN2B},
			{b.Q.W, f.Q.W}, {b.Q.B, f.Q.B}, {b.K.W, f.K.W}, {b.K.B, f.K.B}, {b.V.W, f.V.W}, {b.V.B, f.V.B},
			{b.O.W, f.O.W}, {b.O.B, f.O.B}, {b.FC1.W, f.FC1.W}, {b.FC1.B, f.FC1.B}, {b.FC2.W, f.FC2.W}, {b.FC2.B, f.FC2.B}} {
			if len(pair[0]) == 0 || !slices.Equal(pair[0], pair[1]) {
				t.Errorf("block %d: a streamed tensor differs from Weights()'", l)
			}
		}
		if b.FC1.Out != f.FC1.Out || b.FC1.In != f.FC1.In || b.Q.Out != f.Q.Out {
			t.Errorf("block %d: shapes differ", l)
		}
		n++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if n != len(w.Blocks) || head.layers != nil {
		t.Errorf("streamed %d blocks of %d; the head loaded its own blocks: %v", n, len(w.Blocks), head.layers != nil)
	}
}

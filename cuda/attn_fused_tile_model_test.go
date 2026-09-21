//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttnFusedTile_defaultBitIdenticalWholeModel is gate 2 of attn-fused-tile128-default-PREREGISTERED.md: a whole chunked
// prefill with the DEFAULT selector equals the same prefill with the 64x64 kernel forced (attnTile = -1), bit for bit on
// the last-row logits. Every earlier row's K/V reach the last row's attention, so an earlier-chunk difference would surface
// there. It also asserts WHICH kernel ran (tile128Launches): the hd128 model must use the 128-row tile on every layer of
// every chunk by default and never when forced; the hd64 model must not use it at all (measured ~5% slower there).
func TestAttnFusedTile_defaultBitIdenticalWholeModel(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1")
	}
	for _, mdl := range []struct {
		file     string
		hd       int
		rows     []int
		wantTile bool
	}{
		{"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 128, []int{1300, 3900}, true},
		{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 64, []int{1300}, false},
	} {
		path := modelPath(mdl.file)
		if _, err := os.Stat(path); err != nil {
			t.Skipf("no fixture at %s", path)
		}
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load %s: %v", mdl.file, err)
		}
		rf, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || rf.bAttnBM128hd128 == (Pipeline{}) || rf.bAttnFused64 == (Pipeline{}) {
			m.Close()
			t.Skip("resident declined or fast/tile kernels not loaded")
		}
		_, _, _, _, _, _, vocab := m.Dims()
		layers := len(rf.layers)
		for _, M := range mdl.rows {
			embs := make([][]float32, M)
			var s uint32 = uint32(M)*7 + 1
			for i := range embs {
				s = s*1664525 + 1013904223
				embs[i] = append([]float32(nil), m.EmbedResidentForTest(int(s>>8)%(vocab-1))...)
			}
			run := func(tile int) ([]float32, int) {
				rf.attnTile, rf.tile128Launches = tile, 0
				out, e := rf.PrefillLast(context.Background(), embs, 0)
				if e != nil {
					t.Fatalf("%s M=%d tile=%d: %v", mdl.file, M, tile, e)
				}
				return append([]float32(nil), out...), rf.tile128Launches
			}
			run(-1) // warm
			forced, nForced := run(-1)
			def, nDef := run(0)
			chunks := (M + 511) / 512
			wantDef := 0
			if mdl.wantTile {
				wantDef = layers * chunks
			}
			if nForced != 0 || nDef != wantDef {
				t.Errorf("%s M=%d: 128-row tile launches forced=%d (want 0) default=%d (want %d = %d layers x %d chunks)", mdl.file, M, nForced, nDef, wantDef, layers, chunks)
			}
			if len(def) != len(forced) {
				t.Fatalf("logit length %d vs %d", len(def), len(forced))
			}
			diff := 0
			for i := range def {
				if math.Float32bits(def[i]) != math.Float32bits(forced[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("%s M=%d: %d of %d logits differ between the default tile and 64x64", mdl.file, M, diff, len(def))
			}
			t.Logf("%s hd%d M=%d (%d chunks): default 128-tile launches=%d, 0/%d logits differ", mdl.file, mdl.hd, M, chunks, nDef, len(def))
		}
		rf.attnTile = 0
		m.Close()
	}
}

//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestB35SplitPrefill is B35's first question on CUDA (docs/queue-engineering.md, B35, step 5): does a Gated-DeltaNet prompt
// prefilled in two PrefillLast calls, [0,b) then [b,n), leave what one call over [0,n) leaves? It compares the last row's logits,
// every DeltaNet layer's conv ring and recurrent state, and every full-attention layer's K/V rows, bit for bit.
//
// A pass under 16 rows runs decode's GEMV, a longer one the batched GEMM (TestCUDADeltaNetPrefill_matchesPerToken), so:
//   - gated: a split whose passes are all 16 rows or more must be bit-identical. A restore at b followed by the rest equals a cold
//     prefill only if a row's GEMM result does not depend on where its pass starts.
//   - reported, not gated: a split with a pass under 16 rows (a 2-row head, a 4-token tail, B35's empty think block). These are
//     expected to differ; B35 makes every cold prefill cut at the tail too, so the restore and the cold prompt take the same classes.
func TestB35SplitPrefill(t *testing.T) {
	requireCUDADevice(t)
	path := filepath.Join("..", "testdata", "qwen35-tiny")
	const n = 61
	type arm struct {
		logits      []float32
		win, state  [][]float32
		kRows, vRow [][]float32
	}
	run := func(zeroBetween bool, cuts ...int) arm {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
		}
		_, _, _, _, _, _, vocab := m.Dims()
		emb := func(lo, hi int) [][]float32 {
			rows := make([][]float32, 0, hi-lo)
			for i := lo; i < hi; i++ {
				rows = append(rows, m.EmbedResidentForTest((i*37+5)%vocab))
			}
			return rows
		}
		var a arm
		from := 0
		for _, to := range append(cuts, n) {
			lg, err := r.PrefillLast(context.Background(), emb(from, to), from)
			if err != nil {
				t.Fatalf("PrefillLast [%d,%d): %v", from, to, err)
			}
			a.logits = append([]float32(nil), lg...)
			from = to
			if zeroBetween && from < n {
				for l := range r.layers {
					if Ly := &r.layers[l]; Ly.isDeltaNet() {
						if err := gpu.Upload(Ly.dnState, make([]float32, Ly.dnState.Len())); err != nil {
							t.Fatal(err)
						}
					}
				}
			}
		}
		dl := func(b gpu.Buffer, n int) []float32 {
			out := make([]float32, n)
			if err := gpu.Download(b.At(0), out); err != nil {
				t.Fatalf("download: %v", err)
			}
			return out
		}
		for l := range r.layers {
			Ly := &r.layers[l]
			switch {
			case Ly.isDeltaNet():
				a.win = append(a.win, dl(Ly.dnWin, Ly.dnWin.Len()))
				a.state = append(a.state, dl(Ly.dnState, Ly.dnState.Len()))
			case Ly.kvHalf || Ly.kvShared || Ly.kvDim == 0:
				// f16 or aliased K/V: the logits and the later layers' state already depend on it.
			default:
				a.kRows = append(a.kRows, dl(r.kc[l], n*Ly.kvDim))
				a.vRow = append(a.vRow, dl(r.vc[l], n*Ly.kvDim))
			}
		}
		return a
	}
	diff := func(a, b [][]float32) (int, float64) {
		first, worst := -1, 0.0
		for i := range a {
			for j := range a[i] {
				if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
					if first < 0 {
						first = i
					}
					worst = max(worst, math.Abs(float64(a[i][j]-b[i][j])))
				}
			}
		}
		return first, worst
	}
	ref := run(false)
	if len(ref.state) == 0 || len(ref.kRows) == 0 {
		t.Fatalf("compared %d DeltaNet and %d f32 attention layers: the fixture must have both", len(ref.state), len(ref.kRows))
	}
	for _, c := range []struct {
		name  string
		cuts  []int
		gated bool
		plant bool // the recurrent state zeroed between the passes: must differ, or the comparison sees nothing
	}{
		{"passes of 30 and 31 rows", []int{30}, true, false},
		{"passes of 16, 16 and 29 rows", []int{16, 32}, true, false},
		{"a 2-row head pass", []int{2}, false, false},
		{"a 4-token tail (B35's shape)", []int{n - 4}, false, false},
		{"plant: passes of 30 and 31 rows, state zeroed between", []int{30}, false, true},
	} {
		got := run(c.plant, c.cuts...)
		parts := []struct {
			what string
			a, b [][]float32
		}{
			{"logits", [][]float32{ref.logits}, [][]float32{got.logits}},
			{"conv ring", ref.win, got.win},
			{"recurrent state", ref.state, got.state},
			{"K rows", ref.kRows, got.kRows},
			{"V rows", ref.vRow, got.vRow},
		}
		same := true
		for _, p := range parts {
			if first, worst := diff(p.a, p.b); first >= 0 {
				same = false
				t.Logf("%s: %s differs (first at layer index %d, max |diff| %.3g)", c.name, p.what, first, worst)
			}
		}
		t.Logf("%s: bit-identical to one pass: %v", c.name, same)
		if c.gated && !same {
			t.Errorf("%s: a split whose passes all take the batched GEMM must be bit-identical to one pass", c.name)
		}
		if c.plant && same {
			t.Errorf("%s: zeroing the state between the passes changed nothing, so the comparison cannot see a difference", c.name)
		}
	}
}

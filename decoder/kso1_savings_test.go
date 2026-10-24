package decoder

import (
	"slices"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// TestKSO1_metalSavingsCountsOnlyTheLane: metalKQuantSavings prices only what Metal's q4k lane uploads natively. A recorded
// weight in a layer projection counts on a q4k load for metal; the same record on the embedding table (not a lane projection)
// does not; and nothing counts for another backend or when the load is not q4k (the lane off).
func TestKSO1_metalSavingsCountsOnlyTheLane(t *testing.T) {
	mk := func() linalg.WeightMat { return linalg.QuantizeInt8(make([]float32, 4*256), 4, 256, true) }
	w := &Weights{Embed: mk(), Layers: []LayerWeights{{QProj: mk(), DownProj: mk()}}}
	w.kquant = map[*int8]kquantSrc{}
	note := func(wm *linalg.WeightMat, int8B, nativeB int64) {
		q8, _, _, _ := wm.Int8()
		w.kquant[unsafe.SliceData(q8)] = kquantSrc{kind: "q6k", int8Bytes: int8B, nativeBytes: nativeB}
	}
	note(&w.Layers[0].QProj, 1040, 840)
	note(&w.Layers[0].DownProj, 2000, 1500)
	note(&w.Embed, 9000, 1000)
	for _, c := range []struct {
		quant, backend string
		want           int64
	}{{"q4k", "metal", 200 + 500}, {"q4k", "cuda", 0}, {"int8int8", "metal", 0}} {
		m := &Model{w: w, quant: c.quant}
		if got := m.metalKQuantSavings(c.backend); got != c.want {
			t.Errorf("quant %s backend %s: saves %d, want %d", c.quant, c.backend, got, c.want)
		}
	}
}

// TestKSO1_kquantRows: kquantRows keeps a run (a plain load, a fused split's offset) as its start alone and records a permuted
// mapping (a RoPE un-permutation) row by row, and kquantSrc.src reads both back.
func TestKSO1_kquantRows(t *testing.T) {
	const in = 256
	for _, c := range []struct {
		name string
		src  []int
	}{{"plain", []int{0, 1, 2, 3}}, {"fused split", []int{6, 7, 8, 9}}, {"permuted", []int{0, 2, 1, 3}}} {
		start, perm := kquantRows(len(c.src), in, func(r int) int { return c.src[r] * in })
		if (perm != nil) != (c.name == "permuted") {
			t.Errorf("%s: perm %v", c.name, perm)
		}
		s := kquantSrc{start: start, perm: perm}
		var got []int
		for r := range c.src {
			got = append(got, s.src(r))
		}
		if !slices.Equal(got, c.src) {
			t.Errorf("%s: rows %v, want %v", c.name, got, c.src)
		}
	}
}

package decoder

import (
	"errors"
	"slices"
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// TestKSO1_savingsCountOnlyTheLane: kquantSavings prices only what a backend's q4k lane uploads natively. On metal a recorded
// layer projection counts and the head does not; on cuda the layer projections and the untied head count, only under per-32
// activations, and a K=V layer's v is not uploaded so does not count; a record on the embedding table never counts; nothing
// counts for another backend or a load that is not q4k.
func TestKSO1_savingsCountOnlyTheLane(t *testing.T) {
	mk := func() linalg.WeightMat { return linalg.QuantizeInt8(make([]float32, 4*256), 4, 256, true) }
	q4 := func() linalg.WeightMat {
		wm, err := linalg.WrapQ4K(make([]byte, 4*144), 4, 256)
		if err != nil {
			t.Fatal(err)
		}
		return wm
	}
	build := func(kEqV bool) *Weights {
		w := &Weights{Embed: mk(), LMHead: mk(), Layers: []LayerWeights{{QProj: mk(), VProj: mk(), DownProj: mk(), GateProj: q4()}}, arch: &Architecture{NumLayers: 1}}
		if kEqV {
			w.arch.gemma4 = &gemma4Params{}
			w.Layers[0].VFromK = true
		}
		w.kquant = map[*int8]kquantSrc{}
		note := func(wm *linalg.WeightMat, int8B, nativeB int64) {
			q8, _, _, _ := wm.Int8()
			w.kquant[unsafe.SliceData(q8)] = kquantSrc{kind: "q6k", int8Bytes: int8B, nativeBytes: nativeB}
		}
		note(&w.Layers[0].QProj, 1040, 840)
		note(&w.Layers[0].VProj, 1100, 1000)
		note(&w.Layers[0].DownProj, 2000, 1500)
		note(&w.LMHead, 5000, 4000)
		note(&w.Embed, 9000, 1000)
		return w
	}
	for _, c := range []struct {
		name           string
		quant, backend string
		group          int
		kEqV           bool
		want           int64
	}{
		{"metal", "q4k", "metal", 32, false, 200 + 100 + 500},
		{"cuda", "q4k", "cuda", 32, false, 200 + 100 + 500 + 1000},
		{"cuda, a K=V layer", "q4k", "cuda", 32, true, 200 + 500 + 1000},
		{"cuda, per-vector activations", "q4k", "cuda", 0, false, 0},
		{"webgpu", "q4k", "webgpu", 32, false, 0},
		{"not q4k", "int8int8", "metal", 32, false, 0},
	} {
		m := &Model{w: build(c.kEqV), quant: c.quant, actGroup: c.group}
		if got := m.kquantSavings(c.backend); got != c.want {
			t.Errorf("%s: saves %d, want %d", c.name, got, c.want)
		}
		for _, wm := range []*linalg.WeightMat{&m.w.Embed} {
			if k := m.KQuantNative(c.backend, wm); k != "" {
				t.Errorf("%s: the embedding table is native %q", c.name, k)
			}
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

// TestKSO1_kquantServesRejects: kquantServes, the check noteKQuant records by and KQuantReader.Rows serves by, accepts a plain
// load, a fused split's offset and an in-range permutation, and rejects each way a source could fail Rows: an accessor error, a
// tensor of another kind, dims that are not two, a column mismatch, columns not a multiple of 256, no rows, a negative start, too
// few rows (plain or offset), a permuted row past the end or negative, and a permutation whose length is not the row count.
func TestKSO1_kquantServesRejects(t *testing.T) {
	plain, off, perm, permOut := kquantSrc{start: 0}, kquantSrc{start: 6}, kquantSrc{perm: []int32{3, 0, 2, 1}}, kquantSrc{perm: []int32{3, 0, 4, 1}}
	for _, c := range []struct {
		name string
		dims []int
		ok   bool
		err  error
		in   int
		src  kquantSrc
		rows int
		want bool
	}{
		{"plain", []int{512, 4}, true, nil, 512, plain, 4, true},
		{"fused offset", []int{512, 10}, true, nil, 512, off, 4, true},
		{"permutation", []int{512, 4}, true, nil, 512, perm, 4, true},
		{"accessor error", []int{512, 4}, true, errors.New("bad"), 512, plain, 4, false},
		{"another kind", nil, false, nil, 512, plain, 4, false},
		{"three dims", []int{512, 4, 1}, true, nil, 512, plain, 4, false},
		{"column mismatch", []int{768, 4}, true, nil, 512, plain, 4, false},
		{"columns not a multiple of 256", []int{300, 4}, true, nil, 300, plain, 4, false},
		{"too few rows", []int{512, 3}, true, nil, 512, plain, 4, false},
		{"offset past the end", []int{512, 9}, true, nil, 512, off, 4, false},
		{"permuted row past the end", []int{512, 4}, true, nil, 512, permOut, 4, false},
		{"a negative permuted row", []int{512, 4}, true, nil, 512, kquantSrc{perm: []int32{0, -1, 2, 3}}, 4, false},
		{"a permutation shorter than the rows", []int{512, 4}, true, nil, 512, kquantSrc{perm: []int32{0, 1, 2}}, 4, false},
		{"a permutation longer than the rows", []int{512, 4}, true, nil, 512, kquantSrc{perm: []int32{0, 1, 2, 3}}, 3, false},
		{"a negative start", []int{512, 4}, true, nil, 512, kquantSrc{start: -1}, 4, false},
		{"no rows", []int{512, 4}, true, nil, 512, plain, 0, false},
	} {
		if got := kquantServes(c.dims, c.ok, c.err, c.in, c.src, c.rows); got != c.want {
			t.Errorf("%s: kquantServes %v, want %v", c.name, got, c.want)
		}
	}
}

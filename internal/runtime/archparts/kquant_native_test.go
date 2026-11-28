package archparts

import (
	"slices"
	"testing"
)

// TestKSO1_kquantRows: KQuantRows keeps a run (a plain load, a fused split's offset) as its start alone and records a permuted
// mapping (a RoPE un-permutation) row by row, and KQuantSrc.Src reads both back.
func TestKSO1_kquantRows(t *testing.T) {
	const in = 256
	for _, c := range []struct {
		name string
		src  []int
	}{{"plain", []int{0, 1, 2, 3}}, {"fused split", []int{6, 7, 8, 9}}, {"permuted", []int{0, 2, 1, 3}}} {
		start, perm := KQuantRows(len(c.src), in, func(r int) int { return c.src[r] * in })
		if (perm != nil) != (c.name == "permuted") {
			t.Errorf("%s: perm %v", c.name, perm)
		}
		s := KQuantSrc{Start: start, Perm: perm}
		var got []int
		for r := range c.src {
			got = append(got, s.Src(r))
		}
		if !slices.Equal(got, c.src) {
			t.Errorf("%s: rows %v, want %v", c.name, got, c.src)
		}
	}
}

// TestKSO1_kquantServesRejects: KQuantServes, the check noteKQuant records by and KQuantReader.Rows serves by, accepts a plain
// load, a fused split's offset and an in-range permutation, and rejects each way a source could fail Rows: a tensor of another
// kind, dims that are not two, a column mismatch, no columns, columns not a multiple of 256, no rows, a negative start, too
// few rows (plain or offset), a permuted row past the end or negative, and a permutation whose length is not the row count.
func TestKSO1_kquantServesRejects(t *testing.T) {
	plain, off, perm, permOut := KQuantSrc{Start: 0}, KQuantSrc{Start: 6}, KQuantSrc{Perm: []int32{3, 0, 2, 1}}, KQuantSrc{Perm: []int32{3, 0, 4, 1}}
	for _, c := range []struct {
		name string
		dims []int
		ok   bool
		in   int
		src  KQuantSrc
		rows int
		want bool
	}{
		{"plain", []int{512, 4}, true, 512, plain, 4, true},
		{"fused offset", []int{512, 10}, true, 512, off, 4, true},
		{"permutation", []int{512, 4}, true, 512, perm, 4, true},
		{"no columns", []int{0, 4}, true, 0, plain, 4, false},
		{"another kind", nil, false, 512, plain, 4, false},
		{"three dims", []int{512, 4, 1}, true, 512, plain, 4, false},
		{"column mismatch", []int{768, 4}, true, 512, plain, 4, false},
		{"columns not a multiple of 256", []int{300, 4}, true, 300, plain, 4, false},
		{"too few rows", []int{512, 3}, true, 512, plain, 4, false},
		{"offset past the end", []int{512, 9}, true, 512, off, 4, false},
		{"permuted row past the end", []int{512, 4}, true, 512, permOut, 4, false},
		{"a negative permuted row", []int{512, 4}, true, 512, KQuantSrc{Perm: []int32{0, -1, 2, 3}}, 4, false},
		{"a permutation shorter than the rows", []int{512, 4}, true, 512, KQuantSrc{Perm: []int32{0, 1, 2}}, 4, false},
		{"a permutation longer than the rows", []int{512, 4}, true, 512, KQuantSrc{Perm: []int32{0, 1, 2, 3}}, 3, false},
		{"a negative start", []int{512, 4}, true, 512, KQuantSrc{Start: -1}, 4, false},
		{"no rows", []int{512, 4}, true, 512, plain, 0, false},
	} {
		if got := KQuantServes(c.dims, c.ok, c.in, c.src, c.rows); got != c.want {
			t.Errorf("%s: KQuantServes %v, want %v", c.name, got, c.want)
		}
	}
}

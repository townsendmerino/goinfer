//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestArgmaxRowsPart_matchesHost is E-P08's kernel gate: argmax_rows_part then argmax_finish over a logits row gives
// argmaxF32's id (the first maximum; 0 for a row that starts with NaN, which the caller handles) on rows built to break a
// reduction: ties at the maximum (the lowest index must win), all -inf, all equal, NaN at element 0 and elsewhere, a
// maximum in the last element, and lengths that are not a multiple of the grid.
func TestArgmaxRowsPart_matchesHost(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	part, err := d.NewComputePipeline(lib, "argmax_rows_part")
	if err != nil {
		t.Fatal(err)
	}
	finish, err := d.NewComputePipeline(lib, "argmax_finish")
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(8))
	nan, ninf := float32(math.NaN()), float32(math.Inf(-1))
	type rowCase struct {
		name string
		row  []float32
	}
	var cases []rowCase
	for _, V := range []int{151936, 152064, 32000, 1000, 257, 1} {
		r := make([]float32, V)
		for i := range r {
			r[i] = float32(rng.NormFloat64())
		}
		cases = append(cases, rowCase{"random", r})
		tie := append([]float32(nil), r...)
		mx := float32(10)
		for _, i := range []int{V - 1, V / 2, V / 3, 7 % V} {
			tie[i] = mx
		}
		cases = append(cases, rowCase{"ties", tie})
		last := append([]float32(nil), r...)
		last[V-1] = 20
		cases = append(cases, rowCase{"max last", last})
		inf := make([]float32, V)
		for i := range inf {
			inf[i] = ninf
		}
		cases = append(cases, rowCase{"all -inf", inf})
		eq := make([]float32, V)
		for i := range eq {
			eq[i] = 0.5
		}
		cases = append(cases, rowCase{"all equal", eq})
		n0 := append([]float32(nil), r...)
		n0[0] = nan
		cases = append(cases, rowCase{"NaN first", n0})
		nm := append([]float32(nil), last...)
		for i := 1; i < V; i += 97 {
			nm[i] = nan
		}
		cases = append(cases, rowCase{"NaN scattered", nm})
	}
	uP := NewBufferU32(d, batchArgmaxParts)
	parts, tok := d.NewBufferLen(2*batchArgmaxParts), d.NewBufferLen(1)
	for _, c := range cases {
		V := len(c.row)
		buf, uV := NewBufferFloats(d, c.row), NewBufferU32(d, uint32(V))
		e := q.Begin()
		e.DispatchTG(part, batchArgmaxParts*256, 256, 0, buf, parts, uV, uP)
		e.DispatchTG(finish, 256, 256, 0, parts, tok, uP)
		e.End()
		got := int(tok.U32s()[0])
		if math.IsNaN(float64(c.row[0])) {
			got = 0 // the caller's rule (forwardMultiInto's devID)
		}
		if want := argmaxF32(c.row); got != want {
			t.Errorf("%s, V=%d: device argmax %d, host %d", c.name, V, got, want)
		}
	}
	t.Logf("%d rows: device argmax equal to argmaxF32 on every one", len(cases))
}

//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"math"
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/sampling"
)

// TestArgmaxTieBreak is the C-14 gate: the on-device argmax_reduce must return the LOWEST index on an
// exact tie, matching the CPU reference (decoder.argmax / argmaxF, strict >). The tree reduction pairs
// by thread index, and a left thread's strided max can sit at a higher absolute index than the right
// thread's — so before the fix an exact tie returned the higher index, silently diverging greedy decode
// from the CPU golden. Crafted ties exercise the multi-thread reduction the real forward almost never
// hits. Heavy (needs a resident for the compiled kernel + buffers).
func TestArgmaxTieBreak(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 0.5B model for the compiled argmax kernel)")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	path := modelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		t.Skip("model did not go resident")
	}

	// build a V-length logits vector that is `lo` everywhere except `hi` at the given indices.
	craft := func(V int, at ...int) []float32 {
		v := make([]float32, V)
		for i := range v {
			v[i] = -1e9
		}
		for _, i := range at {
			v[i] = 0.0
		}
		return v
	}
	cases := []struct {
		name string
		V    int
		at   []int
		want int
	}{
		{"tie 1 vs 256 (left thread holds the higher index)", 512, []int{1, 256}, 1},
		{"tie 5 vs 300 across threads", 1000, []int{300, 5}, 5},
		{"three-way tie", 800, []int{600, 40, 271}, 40},
		{"all-equal picks 0", 300, allIdx(300), 0},
		{"single max control", 700, []int{423}, 423},
	}
	for _, c := range cases {
		got, err := rf.ArgmaxForTest(craft(c.V, c.at...))
		if err != nil {
			t.Fatalf("%s: ArgmaxForTest: %v", c.name, err)
		}
		if got != c.want {
			t.Errorf("%s: argmax = %d, want %d (lowest tied index)", c.name, got, c.want)
		}
	}
	// Rows argmax_reduce finds nothing on (no logit above its -1e30 start): the pick takes the host's argmax of the row.
	under := make([]float32, 700)
	for i := range under {
		under[i] = -4e30
	}
	under[423], under[500] = -2e30, -2e30 // a tie under -1e30: the host's strict > keeps the lower index
	allInf := make([]float32, 700)
	for i := range allInf {
		allInf[i] = float32(math.Inf(-1))
	}
	for _, c := range []struct {
		name string
		row  []float32
		want int
	}{{"all -inf picks 0, as the host does", allInf, 0}, {"every logit under -1e30, max tied at 423 and 500", under, 423}} {
		got, err := rf.ArgmaxForTest(c.row)
		if err != nil {
			t.Fatalf("%s: ArgmaxForTest: %v", c.name, err)
		}
		if host := sampling.Argmax(c.row); got != c.want || got != host {
			t.Errorf("%s: argmax = %d, host's %d, want %d", c.name, got, host, c.want)
		}
	}
	// argmax_rows through the resident's own argmaxRows (the verify tail's and the drafter head's readback), each row the
	// vocabulary wide: a finite tie, all -inf, all NaN, and every logit under -1e30 with a tie.
	V := rf.vocab
	fill := func(x float32) []float32 {
		row := make([]float32, V)
		for i := range row {
			row[i] = x
		}
		return row
	}
	tie := fill(-1e9)
	tie[V-3], tie[17] = 0, 0
	underRow := fill(-4e30)
	underRow[423], underRow[500] = -2e30, -2e30
	rows := [][]float32{tie, fill(float32(math.Inf(-1))), fill(float32(math.NaN())), underRow}
	wants := []int{17, 0, 0, 423}
	ids, err := rf.ArgmaxRowsForTest(rows)
	if err != nil {
		t.Fatalf("ArgmaxRowsForTest: %v", err)
	}
	for m, row := range rows {
		if host := sampling.Argmax(row); ids[m] != wants[m] || ids[m] != host {
			t.Errorf("argmax_rows row %d: %d, host's %d, want %d", m, ids[m], host, wants[m])
		}
	}
}

func allIdx(n int) []int {
	a := make([]int, n)
	for i := range a {
		a[i] = i
	}
	return a
}

// TestArgmaxID_nothingComparableTakesTheHostsArgmax: argmax_reduce reports -1 when no logit beats its -1e30 start, and
// argmaxID then returns the host's argmax over the row, so the device's greedy pick (ForwardArgmax, the batched step,
// gumbelPick's fallback) equals the host's on a row with nothing comparable. A found index is returned without a read.
func TestArgmaxID_nothingComparableTakesTheHostsArgmax(t *testing.T) {
	inf := float32(math.Inf(-1))
	nan := float32(math.NaN())
	for _, c := range []struct {
		name string
		row  []float32
		want int
	}{
		{"all -inf", []float32{inf, inf, inf, inf, inf}, 0},
		{"all NaN", []float32{nan, nan, nan}, 0},
		{"finite, every logit under -1e30", []float32{-4e30, -3e30, -2e30, -5e30, -2e30}, 2},
	} {
		got, err := argmaxID(-1, func() ([]float32, error) { return c.row, nil })
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if host := sampling.Argmax(c.row); got != host || got != c.want {
			t.Errorf("%s: argmaxID %d, host's argmax %d, want %d", c.name, got, host, c.want)
		}
	}
	got, err := argmaxID(7, func() ([]float32, error) { t.Fatal("read called for a found index"); return nil, nil })
	if err != nil || got != 7 {
		t.Errorf("a found index: argmaxID %d, %v; want 7", got, err)
	}
	boom := errors.New("download failed")
	if _, err := argmaxID(-1, func() ([]float32, error) { return nil, boom }); !errors.Is(err, boom) {
		t.Errorf("a failed read: argmaxID error %v, want %v", err, boom)
	}
}

// TestArgmaxRowIDs_readsOnlyTheRowsWithNothingComparable: argmaxRowIDs keeps argmax_rows' index for a row that has one,
// reads back only the rows that came back -1, and gives each of those the host's argmax.
func TestArgmaxRowIDs_readsOnlyTheRowsWithNothingComparable(t *testing.T) {
	inf := float32(math.Inf(-1))
	rows := map[int][]float32{1: {inf, inf, inf}, 3: {-3e30, -2e30, -2e30}}
	var read []int
	ids := make([]int, 4)
	err := argmaxRowIDs([]int32{5, -1, 9, -1}, ids, func(m int) ([]float32, error) {
		read = append(read, m)
		return rows[m], nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{5, 0, 9, 1}; ids[0] != want[0] || ids[1] != want[1] || ids[2] != want[2] || ids[3] != want[3] {
		t.Errorf("ids %v, want %v", ids, want)
	}
	if len(read) != 2 || read[0] != 1 || read[1] != 3 {
		t.Errorf("read rows %v, want [1 3]", read)
	}
	boom := errors.New("download failed")
	if err := argmaxRowIDs([]int32{2, -1}, make([]int, 2), func(int) ([]float32, error) { return nil, boom }); !errors.Is(err, boom) {
		t.Errorf("a failed read: error %v, want %v", err, boom)
	}
}

//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"
)

// TestTowerAttnKernel is S17 lever A's kernel gate on Metal (docs/tasks/task-multimodal-support-2026-10.md): tower_attn_hd
// {64,72,80} against a float64 softmax(q·kᵀ)·v on the host, for segment lengths that cover one row, a partial query block,
// a partial key block, several of each, and a tower-sized segment. Bar, set before the first run: every output element
// within 1e-5 of the segment's max |v| (an output is a convex combination of v rows, so max |v| bounds it; f32 accumulation
// and exp2 only). Each planted defect, compiled into the kernel by its dbg uniform, must miss it.
func TestTowerAttnKernel(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibraryPrecise(eg2MSL+gvMSL, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(17))
	type cs struct{ T, nH int }
	cases := []cs{{1, 2}, {7, 2}, {16, 3}, {33, 2}, {70, 2}, {257, 2}, {1100, 2}}
	run := func(hd int, c cs, dbg uint32) float64 {
		p, err := d.NewComputePipeline(lib, fmt.Sprintf("tower_attn_hd%d", hd))
		if err != nil {
			t.Fatal(err)
		}
		n := c.T * c.nH * hd
		qv, kv, vv := make([]float32, n), make([]float32, n), make([]float32, n)
		vmax := 0.0
		for i := range qv {
			qv[i] = float32(rng.NormFloat64() / math.Sqrt(float64(hd)) * 2) // pre-scaled, as the tower's q is
			kv[i] = float32(rng.NormFloat64())
			vv[i] = float32(rng.NormFloat64())
			vmax = math.Max(vmax, math.Abs(float64(vv[i])))
		}
		bq, bk, bv, bo := NewBufferFloats(d, qv), NewBufferFloats(d, kv), NewBufferFloats(d, vv), d.NewBufferLen(n)
		e := q.Begin()
		e.Dispatch(p, c.nH*((c.T+31)/32)*128, 128, bq, bk, bv, bo, NewBufferU32(d, uint32(c.T)), NewBufferU32(d, uint32(c.nH)),
			NewBufferU32(d, dbg))
		e.End()
		if err := e.Err(); err != nil {
			t.Fatal(err)
		}
		got := bo.Floats()
		stride := c.nH * hd
		worst := 0.0
		sc := make([]float64, c.T)
		for h := range c.nH {
			for i := range c.T {
				mx := math.Inf(-1)
				for j := range c.T {
					s := 0.0
					for x := range hd {
						s += float64(qv[i*stride+h*hd+x]) * float64(kv[j*stride+h*hd+x])
					}
					sc[j] = s
					mx = math.Max(mx, s)
				}
				sum := 0.0
				for j := range c.T {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				for x := range hd {
					ref := 0.0
					for j := range c.T {
						ref += sc[j] * float64(vv[j*stride+h*hd+x])
					}
					ref /= sum
					worst = math.Max(worst, math.Abs(float64(got[i*stride+h*hd+x])-ref)/vmax)
				}
			}
		}
		return worst
	}
	const bar = 1e-5
	for _, hd := range []int{64, 72, 80} {
		worst := 0.0
		for _, c := range cases {
			worst = math.Max(worst, run(hd, c, 0))
		}
		fmt.Printf("[S17 A kernel] hd %d: worst |err|/max|v| %.2e over T = 1..1100\n", hd, worst)
		if worst > bar {
			t.Errorf("hd %d: worst error %.2e, bar %.0e", hd, worst, bar)
		}
	}
	// Each planted defect, on head dim 72 at T values where it shows (the rescale needs a max that moves across key
	// blocks; the one-past key mask and the late V need a partial or several key blocks).
	for _, df := range []struct {
		name string
		dbg  uint32
	}{{"(1) the rescale skipped when the max moves", 1}, {"(2) the key mask one past the segment", 2}, {"(3) V read one key late", 3}} {
		worst := 0.0
		for _, c := range []cs{{33, 2}, {70, 2}, {257, 2}} {
			worst = math.Max(worst, run(72, c, df.dbg))
		}
		fmt.Printf("[S17 A kernel] planted %s: worst %.2e\n", df.name, worst)
		if worst <= bar {
			t.Errorf("planted defect %s left the bar green (%.2e)", df.name, worst)
		}
	}
	for _, hd := range []string{"64", "72", "80"} { // the kernels are named by token-pasting TOWER_ATTN(hd)
		if !strings.Contains(gvMSL, "TOWER_ATTN("+hd+")") {
			t.Fatalf("the kernel source instantiates no tower_attn_hd%s", hd)
		}
	}
}

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

// TestTowerGemmKernel is S17 lever B's kernel gate on Metal (docs/tasks/task-multimodal-support-2026-10.md, "S17's
// Metal lever B", registered before the code): tower_gemm_w32 and tower_gemm_w16 against a float64 A·Wᵀ + bias from the
// f32 inputs, at the towers' projection shapes (rows trimmed so the host reference stays fast) and at edge shapes (M, N
// and K off every tile multiple). Bars: every output within 1e-5 (w32: f32 accumulation order only) or 1e-3 (w16: the
// weights rounded to f16) of its own Σ|a·w| + |bias|.
// Weights at a tower's scale (σ 0.02), so a dropped bias is far over either bar. Each
// planted defect must miss it, on both kernels.
//
// tower_gemm_w8 is G-S18e (S18 on the Mac, registered before the code): int8 weights in groups of 32 with an f32 scale
// each, against a float64 reference over the same dequantized weights (float(q)·scale), so the bar is 1e-5 (only the
// accumulation differs); its fourth planted defect takes each group's scale from the next group.
func TestTowerGemmKernel(t *testing.T) {
	for _, k := range []struct {
		name string
		bar  float64
	}{{"tower_gemm_w32", 1e-5}, {"tower_gemm_w16", 1e-3}, {"tower_gemm_w8", 1e-5}} {
		t.Run(k.name, func(t *testing.T) { testTowerGemmKernel(t, k.name, k.bar) })
	}
}

func testTowerGemmKernel(t *testing.T, kernel string, bar float64) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibraryPrecise(eg2MSL+gvMSL, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, kernel)
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(23))
	type sh struct{ M, N, K int }
	shapes := []sh{{1, 7, 9}, {70, 33, 1176}, {65, 65, 17}, {128, 1152, 1152}, {96, 4304, 1152}, {96, 1152, 4304},
		{64, 1280, 1280}, {100, 3420, 1280}}
	run := func(s sh, dbg uint32) float64 {
		a, w, b := make([]float32, s.M*s.K), make([]float32, s.N*s.K), make([]float32, s.N)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		h := make([]uint16, len(w))
		for i := range w {
			w[i] = float32(0.02 * rng.NormFloat64())
			h[i] = f32ToF16(w[i])
		}
		bW, bS := NewBufferU16s(d, h), Buffer{}
		switch kernel {
		case "tower_gemm_w32":
			bW = NewBufferFloats(d, w)
		case "tower_gemm_w8":
			q8, sc := quantG32(w, s.N, s.K)
			bW, bS = NewBufferInt8(d, q8), NewBufferFloats(d, sc)
			w = dequantG32(q8, sc, s.N, s.K) // the reference is over the weights the kernel computes with
		}
		for i := range b {
			b[i] = float32(0.5 * rng.NormFloat64())
		}
		bA, bB, bC := NewBufferFloats(d, a), NewBufferFloats(d, b), d.NewBufferLen(s.M*s.N)
		e := q.Begin()
		args := []Buffer{bA, bW, bC, bB, NewBufferU32(d, uint32(s.M)), NewBufferU32(d, uint32(s.N)), NewBufferU32(d, uint32(s.K)),
			NewBufferU32(d, 1), NewBufferU32(d, dbg)}
		if bS != (Buffer{}) {
			args = append(args, bS)
		}
		e.Dispatch2D(p, (s.N+63)/64, (s.M+63)/64, 512, 1, args...)
		e.End()
		if err := e.Err(); err != nil {
			t.Fatal(err)
		}
		got := bC.Floats()
		worst := 0.0
		for m := range s.M {
			for n := range s.N {
				ref, mag := float64(b[n]), math.Abs(float64(b[n]))
				for k := range s.K {
					v := float64(a[m*s.K+k]) * float64(w[n*s.K+k])
					ref, mag = ref+v, mag+math.Abs(v)
				}
				worst = math.Max(worst, math.Abs(float64(got[m*s.N+n])-ref)/mag)
			}
		}
		return worst
	}
	worst := 0.0
	for _, s := range shapes {
		worst = math.Max(worst, run(s, 0))
	}
	fmt.Printf("[S17 B kernel] %s: worst |err|/(Σ|a·w|+|b|) %.2e over %d shapes (bar %.0e)\n", kernel, worst, len(shapes), bar)
	if worst > bar {
		t.Errorf("worst error %.2e, bar %.0e", worst, bar)
	}
	for _, df := range []struct {
		name string
		dbg  uint32
	}{{"(1) the bias dropped", 1}, {"(2) A staged one K column late", 2}, {"(3) the last quarter of each K step skipped", 3},
		{"(4) each group's scale from the next group", 4}} {
		if df.dbg == 4 && kernel != "tower_gemm_w8" {
			continue
		}
		w := 0.0
		for _, s := range []sh{{70, 33, 1176}, {128, 1152, 1152}} {
			w = math.Max(w, run(s, df.dbg))
		}
		fmt.Printf("[S17 B kernel] %s planted %s: worst %.2e\n", kernel, df.name, w)
		if w <= bar {
			t.Errorf("planted defect %s left the bar green (%.2e)", df.name, w)
		}
	}
}

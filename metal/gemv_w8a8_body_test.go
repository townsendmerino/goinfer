//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestGemvW8A8Body_matchesCPU is gate F1 of docs/tasks/task-metal-int8-2026-10.md: each W8A8 body kernel equals the
// CPU's W8A8 arithmetic bit-for-bit — float32(Σ int8·int8)·aScale·wScale (linalg's w8a8Span), then +bias, then
// +residual — on random int8 inputs at every decode GEMV shape of the 0.5B and 1.5B coder. The library is compiled
// as buildResident compiles it (CompileLibrary, fast math on), since that is what ships. The gate's fallback bar,
// should the build's fast math reorder the epilogue, is at most 1 ulp per output, with the cause recorded.
func TestGemvW8A8Body_matchesCPU(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	q := d.NewCommandQueue()
	// rows × K: qkv, o, gate|up, down for the 0.5B (H 896, I 4864, 14×64 q, 2×64 kv) and the 1.5B (H 1536, I 8960,
	// 12×128 q, 2×128 kv). Every row count is a multiple of 8, as the SA family's threadgroups need.
	shapes := []struct{ n, k int }{
		{1152, 896}, {896, 896}, {9728, 896}, {896, 4864},
		{2048, 1536}, {1536, 1536}, {17920, 1536}, {1536, 8960},
	}
	kernels := []struct {
		name               string
		sa, bias, residual bool
	}{
		{"gemv_w8a8_body", false, false, false},
		{"gemv_w8a8_body_resid", false, false, true},
		{"gemv_w8a8_body_resid_bias", false, true, true},
		{"gemv_w8a8_sa", true, false, false},
		{"gemv_w8a8_sa_bias", true, true, false},
		{"gemv_w8a8_sa_resid", true, false, true},
		{"gemv_w8a8_sa_bias_resid", true, true, true},
	}
	rng := rand.New(rand.NewSource(20261001))
	var checked, bitExact int
	for _, kc := range kernels {
		pipe, err := d.NewComputePipeline(lib, kc.name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", kc.name, err)
		}
		for _, sh := range shapes {
			n, k := sh.n, sh.k
			aq, bq := randInt8(rng, k), randInt8(rng, n*k)
			aSc := 0.001 + rng.Float32()*0.02
			bSc, bias, resid := make([]float32, n), make([]float32, n), make([]float32, n)
			for i := range n {
				bSc[i] = 0.0005 + rng.Float32()*0.01
				bias[i] = rng.Float32()*2 - 1
				resid[i] = rng.Float32()*8 - 4
			}
			want := make([]float32, n)
			for row := range n {
				var acc int32
				for i := range k {
					acc += int32(aq[i]) * int32(bq[row*k+i])
				}
				// Each float32() rounds, as the CPU does by storing the GEMV output before the bias and residual adds:
				// without them Go may fuse the product into the add on arm64, which the spec allows across statements.
				y := float32(float32(acc) * aSc * bSc[row])
				if kc.bias {
					y = float32(y + bias[row])
				}
				if kc.residual {
					y = float32(resid[row] + y)
				}
				want[row] = y
			}
			start := make([]float32, n)
			if kc.residual {
				copy(start, resid)
			}
			out := NewBufferFloats(d, start)
			bufs := []Buffer{NewBufferInt8(d, bq), NewBufferFloats(d, bSc), NewBufferInt8(d, aq), NewBufferFloats(d, []float32{aSc}), out}
			if kc.bias {
				bufs = append(bufs, NewBufferFloats(d, bias))
			}
			bufs = append(bufs, NewBufferU32(d, uint32(k)))
			e := q.Begin()
			if kc.sa {
				e.DispatchTG(pipe, n*32, 256, k*2, bufs...) // the shm the SA dispatch sites pass (2·K bytes)
			} else {
				e.Dispatch(pipe, n*32, 32, bufs...)
			}
			e.End()
			if err := e.Err(); err != nil {
				t.Fatalf("%s %dx%d: dispatch: %v", kc.name, n, k, err)
			}
			got := out.Floats()[:n]
			var exact, oneULP, worse int
			for row := range n {
				switch ulpDist(got[row], want[row]) {
				case 0:
					exact++
				case 1:
					oneULP++
				default:
					worse++
					if worse <= 3 {
						t.Errorf("%s %dx%d row %d: got %g, want %g", kc.name, n, k, row, got[row], want[row])
					}
				}
			}
			if worse > 0 {
				t.Errorf("%s %dx%d: %d of %d outputs more than 1 ulp from the CPU's W8A8", kc.name, n, k, worse, n)
			}
			if oneULP > 0 {
				t.Logf("%s %dx%d: %d exact, %d at 1 ulp (fast math contracting the epilogue)", kc.name, n, k, exact, oneULP)
			}
			checked += n
			bitExact += exact
		}
	}
	t.Logf("%d kernels x %d shapes: %d of %d outputs bit-identical to the CPU's W8A8", len(kernels), len(shapes), bitExact, checked)
}

func randInt8(rng *rand.Rand, n int) []int8 {
	v := make([]int8, n)
	for i := range v {
		v[i] = int8(rng.Intn(255) - 127)
	}
	return v
}

// ulpDist is the number of float32 steps between a and b (both finite).
func ulpDist(a, b float32) int64 {
	if a == b {
		return 0
	}
	ia, ib := int64(math.Float32bits(a)), int64(math.Float32bits(b))
	if ia < 0x80000000 != (ib < 0x80000000) { // opposite signs: count through zero
		return (ia & 0x7fffffff) + (ib & 0x7fffffff)
	}
	if ia > ib {
		return ia - ib
	}
	return ib - ia
}

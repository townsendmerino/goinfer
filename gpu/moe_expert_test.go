//go:build gpu && goinfer_testhooks

package gpu_test

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/gpu"
)

// TestMoEExpertGEMV_indexing gates C3b: the indexed sparse-expert GEMV must read the
// RIGHT expert's row out of the stacked buffer (the dynamic index comes from the routing
// buffer, not a record-time constant). For each expert e it runs the indexed GEMV with
// idx[0]=e and checks the result equals a direct int8 GEMV of expert e's weight — proving
// the stacked addressing + W8A8 math. (The full gate→up→SwiGLU→down combine is gated
// end-to-end on mixtral-tiny in C3c.)
func TestMoEExpertGEMV_indexing(t *testing.T) {
	c, err := gpu.New()
	if err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	defer c.Close()

	const nE, N, K = 6, 20, 48 // K multiple of 16 so kp==K (no padding ambiguity)
	// Distinct per-expert int8 weights + scales so a wrong index is caught.
	q8 := make([][]int8, nE)
	scales := make([][]float32, nE)
	for e := 0; e < nE; e++ {
		q8[e] = make([]int8, N*K)
		scales[e] = make([]float32, N)
		for n := 0; n < N; n++ {
			scales[e][n] = 0.01 * float32(e+1) * float32(n+1)
			for k := 0; k < K; k++ {
				q8[e][n*K+k] = int8(((e*7 + n*3 + k*5) % 17) - 8) // deterministic, in [-8,8]
			}
		}
	}
	st, err := c.UploadStackedExperts(q8, scales, nE, N, K)
	if err != nil {
		t.Fatalf("UploadStackedExperts: %v", err)
	}
	defer st.Release()

	// Activation: a fixed int8 vector + scale.
	aq := make([]int8, K)
	for k := 0; k < K; k++ {
		aq[k] = int8((k % 11) - 5)
	}
	const aScale = 0.0123

	// CPU reference int8 GEMV for expert e: out[n] = (Σ_k q8[e][n,k]·aq[k])·aScale·scale[e][n].
	ref := func(e int) []float32 {
		out := make([]float32, N)
		for n := 0; n < N; n++ {
			var acc int32
			for k := 0; k < K; k++ {
				acc += int32(q8[e][n*K+k]) * int32(aq[k])
			}
			out[n] = float32(acc) * aScale * scales[e][n]
		}
		return out
	}

	for e := 0; e < nE; e++ {
		got, err := c.IndexedGEMVForTest(st, aq, aScale, []int{e}, 0)
		if err != nil {
			t.Fatalf("expert %d: IndexedGEMVForTest: %v", e, err)
		}
		want := ref(e)
		var maxd float64
		for n := 0; n < N; n++ {
			if d := math.Abs(float64(got[n] - want[n])); d > maxd {
				maxd = d
			}
		}
		if maxd > 1e-3 {
			t.Errorf("expert %d: indexed GEMV maxΔ=%.5f (got[:3]=%v want[:3]=%v)", e, maxd, got[:3], want[:3])
		}
	}
	// Cross-check: selecting a different slot index picks a different expert (idx[1]=2).
	got, err := c.IndexedGEMVForTest(st, aq, aScale, []int{5, 2}, 1)
	if err != nil {
		t.Fatalf("slot-1 select: %v", err)
	}
	want := ref(2)
	for n := 0; n < N; n++ {
		if math.Abs(float64(got[n]-want[n])) > 1e-3 {
			t.Fatalf("slot-1 idx[1]=2 picked wrong expert at n=%d: got=%.5f want=%.5f", n, got[n], want[n])
		}
	}
}

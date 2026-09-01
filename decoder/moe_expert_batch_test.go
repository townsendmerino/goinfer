package decoder

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// Is batching the MoE expert matmul over rows worth anything? — the cheap check
// before anyone funds the restructuring.
//
// WHY THIS IS BEING ASKED AGAIN. The 2026-09-01 full-model profile put
// `swiGLUExpert` at 93.1% of moeMLP and moeMLP at 42.1% of prefill — so the
// expert weight matmuls are ~39% of prefill, and the largest single bucket now
// that A3 collapsed attention to 17.4%. At K=8192 the batched-prefill loop
// (forwardn.go) calls moeMLP once PER ROW, and swiGLUExpert issues its three
// matmuls at M=1, so an expert's weights are re-read for every token that
// routes to it.
//
// AND WHY THE EXISTING VERDICT MAY NOT COVER IT. docs/task-moe-streaming.md
// Lever 4 is PARKED on "expert-major MoE prefill batching is NOT a compute
// lever", measured 2026-08-28 as `uniform` (every row picks the same experts)
// against `varied` (real routing), with uniform called the CEILING. But BOTH
// arms of that experiment call moeMLP per row at M=1. What uniform changes is
// which weights get touched, so it captures the BANDWIDTH/locality half of
// batching — the weights stay cache-resident across rows — and not the
// M=1 -> M=N half, which is a different axis: a GEMV is latency/ILP-bound in a
// way a GEMM over hundreds of rows is not. The two coincide only if the
// workload is purely bandwidth-bound.
//
// So that result answers "does routing diversity cost much?" (no — and today's
// profile agrees, routeExperts is 1.7% of moeMLP) without bounding "would
// batching rows into GEMMs help?". This measures the second question directly,
// which is cheaper than arguing about the first.
//
// METHOD. Real Mellum2 expert shapes, real int4 W4A8 weights through the same
// `matmul(be, *WeightMat, ...)` entry point production uses. Same total rows in
// both arms:
//
//	M=1 arm    N separate calls, exactly what moeMLP does today
//	M=N arm    one call over N rows
//
// The comparison is TIME PER ROW. If M=N is not materially cheaper per row, the
// parked verdict covers this too and the lever really is closed.
func TestMoEExpertBatching_M1vsMN(t *testing.T) {
	if os.Getenv("GOINFER_MOE_BATCH_PROBE") == "" {
		t.Skip("set GOINFER_MOE_BATCH_PROBE=1")
	}
	// Mellum2: hidden 2304, intermediate 7168, 64 experts (config.json).
	const (
		hidden = 2304
		inter  = 7168
		group  = 32
	)
	be, err := NewBackend("cpu")
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	mk := func(rows, cols int, seed uint32) linalg.WeightMat {
		return linalg.QuantizeInt4(randF32(rows*cols, seed), rows, cols, group)
	}
	// The three matmuls swiGLUExpert issues, at their real shapes.
	gateW := mk(inter, hidden, 1) // [inter, hidden]
	upW := mk(inter, hidden, 2)   // [inter, hidden]
	downW := mk(hidden, inter, 3) // [hidden, inter]

	fmt.Fprintf(os.Stderr, "MoE expert batching: hidden=%d inter=%d int4/W4A8, Mellum2 shapes\n", hidden, inter)
	fmt.Fprintf(os.Stderr, "%-8s %14s %14s %10s\n", "rows", "M=1 per row", "M=N per row", "speedup")

	for _, N := range []int{8, 32, 64, 128, 256} {
		hIn := randF32(N*hidden, 10)
		gateOut := make([]float32, N*inter)
		upOut := make([]float32, N*inter)
		downOut := make([]float32, N*hidden)

		// ARM 1: N separate M=1 calls — today's shape.
		m1 := func() {
			for i := range N {
				h := hIn[i*hidden : (i+1)*hidden]
				g := gateOut[i*inter : (i+1)*inter]
				u := upOut[i*inter : (i+1)*inter]
				matmul(be, &gateW, h, g, 1)
				matmul(be, &upW, h, u, 1)
				for j := range g {
					g[j] = silu(g[j]) * u[j]
				}
				matmul(be, &downW, g, downOut[i*hidden:(i+1)*hidden], 1)
			}
		}
		// ARM 2: one M=N call each — what expert-major batching would issue.
		mN := func() {
			matmul(be, &gateW, hIn, gateOut, N)
			matmul(be, &upW, hIn, upOut, N)
			for j := range gateOut {
				gateOut[j] = silu(gateOut[j]) * upOut[j]
			}
			matmul(be, &downW, gateOut, downOut, N)
		}
		best := func(f func()) time.Duration {
			f() // warm
			b := time.Duration(1<<62 - 1)
			for range 3 {
				t0 := time.Now()
				f()
				if d := time.Since(t0); d < b {
					b = d
				}
			}
			return b
		}
		// Interleaved so drift cannot land on one arm.
		d1 := best(m1)
		dN := best(mN)
		d1b := best(m1)
		if d1b < d1 {
			d1 = d1b
		}
		per1 := float64(d1.Microseconds()) / float64(N)
		perN := float64(dN.Microseconds()) / float64(N)
		fmt.Fprintf(os.Stderr, "%-8d %11.1f µs %11.1f µs %9.2fx\n", N, per1, perN, per1/perN)
	}
}

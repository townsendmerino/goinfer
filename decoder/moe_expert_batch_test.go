package decoder

import (
	"fmt"
	"os"
	"sync/atomic"
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
	// Mellum2 config.json: hidden_size 2304, **moe_intermediate_size 896**,
	// 64 experts, top-k 8.
	//
	// 896 IS THE EXPERT WIDTH AND 7168 IS THE DENSE ONE. The first version of
	// this benchmark used 7168 -- `intermediate_size`, which the DENSE FFN uses
	// -- and so measured expert matmuls 8x wider than the ones swiGLUExpert
	// actually issues (it is called with moe.IntermediateDim). Wider matmuls
	// amortise per-call overhead better, so that error understated the batching
	// win rather than inventing one, but it was still the wrong shape.
	const (
		hidden = 2304
		inter  = 896
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

// TestMoEExpertMajor_bitIdentical is the gate that decides whether P18's
// expert-major path is usable at all.
//
// The win is worthless if it changes results: the whole point of running the
// MoE FFN expert-major is that it is a REORDERING of the same arithmetic, not
// a different computation. Two things have to survive, and a tolerance test
// would let both fail quietly:
//
//   - the matmuls must be M-invariant (a row computed alone equals the same row
//     computed inside a batch), which linalg documents as a contract; and
//   - the per-row fold must run in ROUTING RANK order, because float addition
//     is not associative and expert-major visits experts in a different order.
//
// So this asserts `!=` on every logit, through the real forward, with the flag
// on and off over the same tokens.
func TestMoEExpertMajor_bitIdentical(t *testing.T) {
	m, err := loadMoEBitIdentModel(t)
	if err != nil {
		t.Skipf("no MoE model (%v)", err)
	}
	ctx := deadlineCtx(t)
	const K = 600 // > moeExpertMajorChunk, so the chunk boundary is exercised
	if !m.canBatchN(K) {
		t.Skip("model has no batched prefill")
	}
	ids := make([]int, K)
	for i := range ids {
		ids[i] = 700 + i%97
	}
	run := func(on string) []float32 {
		t.Helper()
		t.Setenv("GOINFER_MOE_EXPERT_MAJOR", on)
		out, err := m.forwardLayersN(ctx, ids, m.NewCache(K+8), false)
		if err != nil {
			t.Fatalf("forward (expert-major=%q): %v", on, err)
		}
		return out
	}
	off := run("0")
	before := atomic.LoadInt64(&moeExpertMajorRuns)
	on := run("1")
	// NON-VACUITY: moeMLPBatch refuses for several legitimate reasons, and a
	// refusal makes both arms take the identical per-row path -- this test would
	// then pass while proving nothing about the batched path.
	if ran := atomic.LoadInt64(&moeExpertMajorRuns) - before; ran == 0 {
		t.Fatal("expert-major path never ran (moeMLPBatch refused) — this comparison proves nothing")
	} else {
		t.Logf("expert-major chunks executed: %d", ran)
	}
	if len(off) != len(on) {
		t.Fatalf("length %d vs %d", len(off), len(on))
	}
	diff := 0
	for i := range off {
		if off[i] != on[i] {
			if diff == 0 {
				t.Errorf("first divergence at logit %d: off=%v on=%v", i, off[i], on[i])
			}
			diff++
		}
	}
	if diff != 0 {
		t.Fatalf("expert-major is NOT bit-identical: %d/%d logits differ", diff, len(off))
	}
	// Non-vacuity: two buffers of zeros would satisfy the loop above perfectly.
	nz := 0
	for _, v := range off {
		if v != 0 {
			nz++
		}
	}
	if nz < len(off)/2 {
		t.Fatalf("output is mostly zeros (%d/%d) — the arms agree because nothing ran", nz, len(off))
	}
}

// loadMoEBitIdentModel resolves a real MoE checkpoint for the gate above.
func loadMoEBitIdentModel(t *testing.T) (*Model, error) {
	t.Helper()
	path := os.Getenv("GOINFER_MELLUM_CKPT")
	if path == "" {
		path = os.Getenv("HOME") + "/models/mellum2-unq"
	}
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	return Load(path, Options{Quant: "int4"})
}

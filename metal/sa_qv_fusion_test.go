//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestSAQVFusion_correctnessAndThroughput is the measurement item #5 of the 9-finding audit
// needs before it goes near a production dispatch site: does fusing quant_vec into
// gemv_w4a8_sa (gemv_w4a8_sa_qv) actually win, given every threadgroup the fused kernel
// launches redoes quant_vec's O(K) amax reduction independently (no cheap way for one
// threadgroup to hand a computed scale to another within one Metal dispatch)? Unlike item #4
// (the rope2 merge, a genuine reduction in total work), this trades one dispatch launch +
// one K-element device-memory round-trip against (N/8 - 1) redundant K-element reductions.
//
// VERDICT (measured, not inspected): roughly NEUTRAL, leaning slightly negative — NOT the win
// the audit's dispatch-count estimate implied. Four interleaved runs (see the interleaving
// comment below for why non-interleaved gave a false 1.28x win) at real dims: 0.974x, 0.949x,
// 0.992x, 0.972x speedup — a tight cluster around ~0.97x, i.e. the fused kernel is a few
// percent SLOWER, not faster. The redundant per-threadgroup reduction cost roughly cancels the
// removed-dispatch savings at this K/N. Kept in the tree as a correctness-proven (bit-identical
// to the two-dispatch path) but NOT-production-worthwhile experiment — do not wire this into
// model.go on the strength of the dispatch-count argument alone; the wall-clock number doesn't
// back it up here.
//
// Real dims (K=1536, N=1536): qwen2.5-coder-1.5b's o-proj (batched_verify_test.go:599 —
// hidden=1536, 12 heads, headDim=128, nH*hd=1536=K; o-proj output=hidden=1536=N).
func TestSAQVFusion_correctnessAndThroughput(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pQv, err := d.NewComputePipeline(lib, "quant_vec")
	if err != nil {
		t.Fatalf("pipeline quant_vec: %v", err)
	}
	pSA, err := d.NewComputePipeline(lib, "gemv_w4a8_sa")
	if err != nil {
		t.Fatalf("pipeline gemv_w4a8_sa: %v", err)
	}
	pFused, err := d.NewComputePipeline(lib, "gemv_w4a8_sa_qv")
	if err != nil {
		t.Fatalf("pipeline gemv_w4a8_sa_qv: %v", err)
	}

	const K, N = 1536, 1536 // qwen2.5-coder-1.5b o-proj

	rng := rand.New(rand.NewSource(41))
	x := make([]float32, K)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 3
	}
	// Packed int4 weight rows, same fixture pattern this package's other SA-family tests use.
	group := 32
	G := K / group
	wq := make([]uint32, N*(K/8))
	sct := make([]uint16, N*G)
	for row := 0; row < N; row++ {
		for gi := 0; gi < G; gi++ {
			var nibs [32]int8
			for i := range nibs {
				nibs[i] = int8(rng.Intn(16) - 8) // nibble range, pre-bias
			}
			base := row*(K/8) + gi*4
			for w4 := 0; w4 < 4; w4++ {
				var word uint32
				for n := 0; n < 8; n++ {
					v := uint32(nibs[w4*8+n]+8) & 0xF
					word |= v << (n * 4)
				}
				wq[base+w4] = word
			}
			sct[row*G+gi] = f32ToF16(rng.Float32()*0.05 + 0.001)
		}
	}

	q_ := d.NewCommandQueue()
	wqb := NewBufferUint32s(d, wq)
	sctb := NewBufferU16s(d, sct)
	xb := NewBufferFloats(d, x)
	uK := NewBufferU32(d, uint32(K))

	// --- correctness: fused vs two-dispatch, must match to int4/int8 quant precision ---
	twoDispatchOut := func() []float32 {
		aq := byteBuf(d, K)
		asc := NewBufferFloats(d, []float32{0})
		out := d.NewBufferLen(N)
		e := q_.Begin()
		e.Dispatch(pQv, 256, 256, xb, aq, asc, uK)
		e.DispatchTG(pSA, N*32, 256, K*2, wqb, sctb, aq, asc, out, uK)
		e.End()
		return out.Floats()
	}
	fusedOut := func() []float32 {
		out := d.NewBufferLen(N)
		e := q_.Begin()
		e.DispatchTG(pFused, N*32, 256, K*2, wqb, sctb, xb, out, uK)
		e.End()
		return out.Floats()
	}
	want := twoDispatchOut()
	got := fusedOut()
	var worst float64
	at := -1
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > worst {
			worst, at = diff, i
		}
	}
	mustFinite(t, "fused vs two-dispatch max|diff|", worst)
	if worst != 0 {
		t.Errorf("gemv_w4a8_sa_qv vs quant_vec+gemv_w4a8_sa: max|diff| %.3e at %d (got %v want %v) — want exactly 0, same math",
			worst, at, got[at], want[at])
	}
	t.Logf("correctness: fused vs two-dispatch max|diff|=%.3e over N=%d", worst, N)

	// --- throughput: warmup + best-of-N per config, INTERLEAVED (see below) rather than
	// TestPrefillGemmW4's "measure all of A, then all of B" shape — that ordering turned out to
	// be a real confound here (see the interleaving comment below).

	// reps=20: NOT the real per-token count (a real decode token issues far fewer than this per
	// layer) — chosen to stay clear of a real, separate, pre-existing issue: repeatedly calling
	// Encoder.Dispatch on the SAME reused buffers hundreds of times in one encoder (via the
	// general Dispatch path, which rebinds every buffer each call, unlike Run1DBatch's
	// bind-once-dispatch-many) hits a probabilistic crash unrelated to gemv_w4a8_sa_qv's
	// correctness (reproduces with the plain two-dispatch pattern alone, no fused kernel
	// involved, and isn't a hard threshold — it can still fire occasionally even at reps=20).
	// Worth its own investigation separately; out of scope here.
	const reps = 20
	aq := byteBuf(d, K)
	asc := NewBufferFloats(d, []float32{0})
	out1 := d.NewBufferLen(N)
	runTwoDispatch := func(r int) {
		e := q_.Begin()
		for range r {
			e.Dispatch(pQv, 256, 256, xb, aq, asc, uK)
			e.DispatchTG(pSA, N*32, 256, K*2, wqb, sctb, aq, asc, out1, uK)
		}
		e.End()
	}
	out2 := d.NewBufferLen(N)
	runFused := func(r int) {
		e := q_.Begin()
		for range r {
			e.DispatchTG(pFused, N*32, 256, K*2, wqb, sctb, xb, out2, uK)
		}
		e.End()
	}

	// INTERLEAVED, not two separate blocks: measuring "all of A then all of B" confounds the
	// comparison with whatever changes between the two blocks (thermal ramp, GPU contention
	// drift) — measured directly here: a first pass with A-then-B block order showed fused
	// WINNING 1.28x; two immediate re-runs of the same block order showed fused LOSING ~0.47x,
	// consistently with each other but not with the first run. That is a confound, not a real
	// effect, and it wouldn't have been visible without deliberately re-running. Alternating A/B
	// every sample makes drift affect both roughly equally instead of favoring whichever block
	// happens to run when conditions are better.
	for range 4 { // warmup, matches prof()'s own warmup count
		runTwoDispatch(reps)
		runFused(reps)
	}
	twoDispatchBest, fusedBest := time.Hour, time.Hour
	for range 15 {
		t0 := time.Now()
		runTwoDispatch(reps)
		if dt := time.Since(t0); dt < twoDispatchBest {
			twoDispatchBest = dt
		}
		t0 = time.Now()
		runFused(reps)
		if dt := time.Since(t0); dt < fusedBest {
			fusedBest = dt
		}
	}
	twoDispatchTime := twoDispatchBest / time.Duration(reps)
	fusedTime := fusedBest / time.Duration(reps)

	speedup := float64(twoDispatchTime) / float64(fusedTime)
	t.Logf("K=%d N=%d (interleaved): two-dispatch (quant_vec+gemv_w4a8_sa) %v/op, fused (gemv_w4a8_sa_qv) %v/op, speedup=%.3fx",
		K, N, twoDispatchTime, fusedTime, speedup)
	if speedup < 1.0 {
		t.Logf("VERDICT: fusion is a NET LOSS at these dims (%.3fx) — the redundant per-threadgroup amax reduction (~N/8=%d redundant O(K) passes) outweighs the removed dispatch + memory round-trip", speedup, N/8)
	} else {
		t.Logf("VERDICT: fusion is a net WIN at these dims (%.3fx)", speedup)
	}
}

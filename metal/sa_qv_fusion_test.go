//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestSAQVFusion_correctnessAndThroughput measures whether fusing quant_vec into gemv_w4a8_sa (gemv_w4a8_sa_qv) wins (item #5 of
// the 9-finding audit), given every threadgroup the fused kernel launches redoes quant_vec's O(K) amax reduction independently
// (one threadgroup cannot hand a computed scale to another within one Metal dispatch). Unlike the rope2 merge (item #4, a real
// reduction in total work), this trades one dispatch launch + one K-element device-memory round-trip against (N/8 - 1)
// redundant K-element reductions.
//
// VERDICT: roughly NEUTRAL, leaning slightly negative (a few percent slower, not faster; figures in
// docs/code-notes/metal.md#TestSAQVFusion_correctnessAndThroughput). Kept as a correctness-proven (bit-identical to the
// two-dispatch path) but NOT-production-worthwhile experiment: do not wire this into model.go on the strength of the
// dispatch-count argument alone.
//
// Real dims (K=1536, N=1536): qwen2.5-coder-1.5b's o-proj (hidden=1536, 12 heads, headDim=128, nH*hd=1536=K; o-proj output=hidden=1536=N).
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
	for row := range N {
		for gi := range G {
			var nibs [32]int8
			for i := range nibs {
				nibs[i] = int8(rng.Intn(16) - 8) // nibble range, pre-bias
			}
			base := row*(K/8) + gi*4
			for w4 := range 4 {
				var word uint32
				for n := range 8 {
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

	// reps=20: NOT the real per-token count. It is chosen to stay clear of a separate, pre-existing issue: repeatedly calling
	// Encoder.Dispatch on the same reused buffers hundreds of times in one encoder hits a probabilistic crash unrelated to
	// gemv_w4a8_sa_qv (it reproduces with the plain two-dispatch pattern alone and can still fire occasionally at reps=20). Not
	// investigated; the original note is docs/code-notes/metal.md#TestSAQVFusion_correctnessAndThroughput.reps
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

	// INTERLEAVED, not two separate blocks: measuring all of A then all of B confounds the comparison with whatever changes between
	// the blocks (thermal ramp, GPU contention drift), and a block-ordered pass here read fused WINNING where immediate re-runs of
	// the same order read it LOSING. Alternating A/B every sample makes drift affect both roughly equally instead of favoring
	// whichever block runs when conditions are better.
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

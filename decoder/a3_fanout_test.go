package decoder

import (
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// A3 fan-out: is the f32 prefill attention path single-threaded? attendBatchedHeads's f32 branch is serial
// across heads (its per-kv-group gather is shared mutable state), but its two matmuls are linalg.MatmulBT,
// which already fans out over output columns (parallelCols) above parThreshold = 1<<24 MACs. So the
// question is the fan-out LEVEL (head vs column); a serial-vs-serial kernel ratio must not be set against a
// profile share measured on the parallel production path.
//
// This measures the real attendBatchedHeads at Mellum2 geometry and reports utilization (CPU time / wall
// time) beside wall time: a single-threaded path cannot exceed ~1.0x. Arms, on identical inputs: acc64,
// f32-colpar (pre-A3: column-parallel inside MatmulBT), f32-headpar (A3) and f32-serial (linalg forced
// serial, the do-nothing arm).
//
// Reading, as coded (utilization of the f32-colpar arm): > 2.0x the premise is false (already parallel),
// < 1.3x it holds (head fan-out is real headroom), in between is ambiguous and parks.
// The premise's origin and the figures: docs/code-notes/decoder.md#TestA3FanoutUtilization.
func TestA3FanoutUtilization(t *testing.T) {
	if os.Getenv("GOINFER_A3_FANOUT") == "" {
		t.Skip("set GOINFER_A3_FANOUT=1 to run the A3 fan-out utilization measurement")
	}
	// Mellum2 geometry; one layer's attention is what attendBatchedHeads is called with.
	const (
		nH    = 32
		nKV   = 4
		hd    = 128
		kvDim = nKV * hd
		qDim  = nH * hd
		nKeys = 8192
		K     = 2048 // query rows per call; tile is 256, so 8 tiles/head
		reps  = 3
	)
	arch := &Architecture{NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, AttnScale: 1 / 11.3137}

	q := randF32(K*qDim, 1)
	keys := randF32(nKeys*kvDim, 2)
	vals := randF32(nKeys*kvDim, 3)
	ctx := make([]float32, K*qDim)
	cache := &KVCache{}

	nw := prefillAttnWorkers(K, nKeys, hd, nH)
	// wantFused=false: this pool is used under both useAcc64 states below, so it cannot promise fusedOK;
	// vt/scores must stay allocated for the acc64 arm's `scores` write.
	pool := newHeadWorkerPool(nw, K, nKeys, hd, false)
	fmt.Fprintf(os.Stderr, "A3 fan-out: nH=%d nKV=%d hd=%d nKeys=%d K=%d tile=%d workers=%d GOMAXPROCS=%d\n",
		nH, nKV, hd, nKeys, K, attnRowTile(K, nKeys), nw, runtime.GOMAXPROCS(0))

	// startPos = nKeys-K so the last query row attends the full depth.
	startPos := nKeys - K

	one := newHeadWorkerPool(1, K, nKeys, hd, false) // 1 slot => serial head loop

	run := func(name string, useAcc64 bool, forceSerial bool, p []headWorkerScratch) (time.Duration, float64) {
		if forceSerial {
			orig := linalg.ParallelThreshold()
			linalg.SetParallelThreshold(1 << 62)
			defer linalg.SetParallelThreshold(orig)
		}
		call := func() {
			attendBatchedHeads(q, ctx, keys, vals, 0, cache, 0, startPos, K, true, arch, useAcc64, p)
		}
		call() // warm
		best, bestUtil := time.Duration(1<<62-1), 0.0
		for range reps {
			c0 := cpuSeconds()
			t0 := time.Now()
			call()
			wall := time.Since(t0)
			cpu := cpuSeconds() - c0
			if wall < best {
				best, bestUtil = wall, cpu/wall.Seconds()
			}
		}
		fmt.Fprintf(os.Stderr, "  %-12s wall %8.1f ms   utilization %5.2fx\n",
			name, float64(best.Microseconds())/1000, bestUtil)
		return best, bestUtil
	}

	// Arms, interleaved so thermal or page-cache drift cannot line up with one arm. "f32-colpar" is the
	// pre-A3 shape: a 1-slot pool takes the serial head loop, whose matmul is the column-parallel
	// package-level MatmulBT. "f32-headpar" is A3: the pool fans out over heads with a serial matmul per worker.
	// Same function and inputs, bit-identical outputs (TestAttendF32Fanout_bitIdentical): only the fan-out
	// level moves. "f32-serial" forces linalg serial.
	wAcc, uAcc := run("acc64", true, false, pool)
	wCol, uCol := run("f32-colpar", false, false, one)
	wHead, uHead := run("f32-headpar", false, false, pool)
	wSer, uSer := run("f32-serial", false, true, one)
	_, _ = run("acc64 (2)", true, false, pool)
	wCol2, _ := run("f32-colpar (2)", false, false, one)
	wHead2, uHead2 := run("f32-headpar (2)", false, false, pool)

	uF32 := uCol // the pre-registered question was about the SHIPPING f32 path as it was
	fmt.Fprintf(os.Stderr, "\n  acc64 utilization:              %.2fx\n", uAcc)
	fmt.Fprintf(os.Stderr, "  f32 col-parallel vs serial:     %.2fx  <- what MatmulBT's internal fan-out already gave\n",
		float64(wSer)/float64(wCol))
	fmt.Fprintf(os.Stderr, "  A3: head-parallel vs col-only:  %.2fx / %.2fx  (utilization %.2fx -> %.2fx)\n",
		float64(wCol)/float64(wHead), float64(wCol2)/float64(wHead2), uCol, uHead)
	fmt.Fprintf(os.Stderr, "  head-parallel f32 vs acc64:     %.2fx\n", float64(wAcc)/float64(wHead))
	fmt.Fprintf(os.Stderr, "  utilizations: col %.2fx  head %.2fx / %.2fx  serial %.2fx\n", uCol, uHead, uHead2, uSer)

	switch {
	case uF32 > 2.0:
		fmt.Fprintf(os.Stderr, "\n  VERDICT: premise FALSE — the f32 path is already parallel (%.2fx cores).\n", uF32)
	case uF32 < 1.3:
		fmt.Fprintf(os.Stderr, "\n  VERDICT: premise HOLDS — f32 runs on ~one core; head fan-out is real headroom.\n")
	default:
		fmt.Fprintf(os.Stderr, "\n  VERDICT: AMBIGUOUS (%.2fx) — parked per the pre-registered band.\n", uF32)
	}
}

func randF32(n int, seed uint32) []float32 {
	s := seed*2654435761 + 1
	out := make([]float32, n)
	for i := range out {
		s ^= s << 13
		s ^= s >> 17
		s ^= s << 5
		out[i] = float32(int32(s)) / float32(1<<31) // [-1,1)
	}
	return out
}

// TestAttendF32Fanout_bitIdentical pins A3's central claim: the head-parallel f32 arm produces
// BYTE-IDENTICAL output to the serial arm. It is bit-identity rather than a tolerance because the two things
// a tolerance would let through are the two ways this change could be wrong: a worker reading another
// worker's kh/vt, and MatmulBT's column fan-out being width-sensitive. Both show up as small drift a cosine
// bar would wave through.
//
// The serial arm runs through a 1-slot pool with the column-parallel package-level MatmulBT, the parallel
// arm through a multi-slot pool with each worker's serial Workspace, so it gates the head split AND the
// claim that MatmulBT is numerically inert to fan-out width.
func TestAttendF32Fanout_bitIdentical(t *testing.T) {
	const (
		nH    = 8
		nKV   = 2
		hd    = 16
		kvDim = nKV * hd
		qDim  = nH * hd
		nKeys = 96
		K     = 48
	)
	arch := &Architecture{NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, AttnScale: 0.25}
	q := randF32(K*qDim, 11)
	keys := randF32(nKeys*kvDim, 12)
	vals := randF32(nKeys*kvDim, 13)
	cache := &KVCache{}
	startPos := nKeys - K

	serialCtx := make([]float32, K*qDim)
	parCtx := make([]float32, K*qDim)

	// wantFused=true: both arms always pass useAcc64=false against a treeMask-less cache, so this also
	// exercises the vt/scores elimination (GOINFER_FUSED_ATTENTION, default on) beside the fan-out identity.
	serialPool := newHeadWorkerPool(1, K, nKeys, hd, true)
	if len(serialPool) != 1 {
		t.Fatalf("serial arm wants exactly 1 slot, got %d", len(serialPool))
	}
	attendBatchedHeads(q, serialCtx, keys, vals, 0, cache, 0, startPos, K, true, arch, false, serialPool)

	parPool := newHeadWorkerPool(6, K, nKeys, hd, true)
	if len(parPool) < 2 {
		t.Fatalf("parallel arm needs >1 slot to exercise the fan-out, got %d", len(parPool))
	}
	attendBatchedHeads(q, parCtx, keys, vals, 0, cache, 0, startPos, K, true, arch, false, parPool)

	// Assert on the OUTPUT: this is the assertion behind the doc comment's bit-identity claim.
	for i := range serialCtx {
		if serialCtx[i] != parCtx[i] {
			t.Fatalf("f32 fan-out is not bit-identical: ctx[%d] serial=%v parallel=%v (head %d, row %d)",
				i, serialCtx[i], parCtx[i], (i%qDim)/hd, i/qDim)
		}
	}
	// Guard against passing on two buffers of zeros: a gather that never ran would satisfy the loop above.
	nonzero := 0
	for _, v := range serialCtx {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < len(serialCtx)/2 {
		t.Fatalf("output is mostly zeros (%d/%d nonzero) — the arms agree because nothing ran", nonzero, len(serialCtx))
	}
}

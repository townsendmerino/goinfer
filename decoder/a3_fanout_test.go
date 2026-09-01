package decoder

import (
	"fmt"
	"os"
	"runtime"
	"syscall"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// A3 fan-out — is the f32 prefill attention path actually single-threaded?
//
// The premise this test exists to check, quoted from G24's own harness
// (g24_attnkernel_test.go) and carried into the queue and the Atlas as a
// costed next move:
//
//	"the f32 branch in attendBatchedHeads is single-threaded by construction
//	 (its per-kv-group gather is shared mutable state)"
//
// That sentence is true about the HEAD LOOP and was then read as if it were
// true about the WORK. It is not. The f32 arm's two matmuls are
// linalg.MatmulBT, which fans out internally over its N output columns
// (parallelCols) above parThreshold = 1<<24 MACs. At the K=8192 tile shape
// each call is ~268M MACs — 16x over that line — so the f32 path is already
// using every core, just at the COLUMN level instead of the HEAD level.
//
// G24 measured its two arms with MatmulBT forced serial, deliberately and
// correctly, to get an arithmetic-plus-gather ratio. The error was downstream
// of it: that serial-vs-serial ratio (8.37x) was fed into an Amdahl estimate
// against a profile share measured on the PARALLEL production path, and the
// gap between them was booked as recoverable headroom worth ~13%. Two
// different parallelism states either side of one division sign.
//
// So this measures the real attendBatchedHeads, at real Mellum2 geometry, and
// reports UTILIZATION (CPU time / wall time) alongside wall time. Utilization
// is the discriminator: a genuinely single-threaded path cannot exceed ~1.0x
// no matter what its wall clock says, and no amount of Amdahl arithmetic can
// argue it up.
//
// Arms, all on identical inputs:
//
//	acc64            production default before 2026-08-31 (head-parallel)
//	f32              production default now (column-parallel inside MatmulBT)
//	f32-serial       f32 with linalg forced serial — the do-nothing arm, and
//	                 the ONLY arm the "single-threaded by construction" claim
//	                 actually describes
//
// Pre-registered reading (written before the first run):
//
//	If f32 utilization is ~1x, the premise holds and head fan-out is worth
//	building. If it is >2x, the premise is false, the ~13% is not there to
//	recover, and the queue item closes as already-parallel. The band between
//	1x and 2x is ambiguous and parks pending a profile.
func TestA3FanoutUtilization(t *testing.T) {
	if os.Getenv("GOINFER_A3_FANOUT") == "" {
		t.Skip("set GOINFER_A3_FANOUT=1 to run the A3 fan-out utilization measurement")
	}
	// Mellum2 geometry — the model both 8k f32 runs used. 28 layers, but one
	// layer's attention is what attendBatchedHeads is called with.
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
	pool := newHeadWorkerPool(nw, K, nKeys, hd)
	fmt.Fprintf(os.Stderr, "A3 fan-out: nH=%d nKV=%d hd=%d nKeys=%d K=%d tile=%d workers=%d GOMAXPROCS=%d\n",
		nH, nKV, hd, nKeys, K, attnRowTile(K, nKeys), nw, runtime.GOMAXPROCS(0))

	// startPos = nKeys-K so the last query row attends the full depth.
	startPos := nKeys - K

	one := newHeadWorkerPool(1, K, nKeys, hd) // 1 slot => serial head loop

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

	// Interleaved, not blocked: run the arms in an order that does not let a
	// thermal or page-cache drift line up with one arm (CLAUDE.md measurement
	// discipline; difference matched observations).
	// Arms, interleaved. "f32-colpar" is the PRE-A3 shape: a 1-slot pool takes
	// the serial head loop, whose matmul is the column-parallel package-level
	// MatmulBT. "f32-headpar" is A3: the 6-slot pool fans out over heads with a
	// serial matmul per worker. Same function, same inputs, bit-identical
	// outputs (TestAttendF32Fanout_bitIdentical) — only the fan-out LEVEL moves.
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

// cpuSeconds returns this process's user+sys CPU time. Utilization is
// cpu/wall, which is what separates "fast because parallel" from "fast".
func cpuSeconds() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	tv := func(t syscall.Timeval) float64 { return float64(t.Sec) + float64(t.Usec)/1e6 }
	return tv(ru.Utime) + tv(ru.Stime)
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

// TestAttendF32Fanout_bitIdentical pins A3's central claim: the head-parallel
// f32 arm produces BYTE-IDENTICAL output to the serial arm.
//
// It has to be bit-identity rather than a tolerance, because the two things a
// tolerance would let through are exactly the two ways this change could be
// wrong: a worker reading another worker's kh/vt (the race the old comment
// feared), and MatmulBT's column fan-out being width-sensitive after all. Both
// would show up as small numeric drift, which a cosine bar would wave through.
//
// The serial arm runs through a 1-slot pool with the column-parallel
// package-level MatmulBT; the parallel arm through a 6-slot pool with each
// worker's serial Workspace. So this simultaneously gates the head split AND
// the claim that MatmulBT is numerically inert to fan-out width — if aikit
// ever broke that contract, this goes red here rather than silently in a
// generated token.
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

	serialPool := newHeadWorkerPool(1, K, nKeys, hd)
	if len(serialPool) != 1 {
		t.Fatalf("serial arm wants exactly 1 slot, got %d", len(serialPool))
	}
	attendBatchedHeads(q, serialCtx, keys, vals, 0, cache, 0, startPos, K, true, arch, false, serialPool)

	parPool := newHeadWorkerPool(6, K, nKeys, hd)
	if len(parPool) < 2 {
		t.Fatalf("parallel arm needs >1 slot to exercise the fan-out, got %d", len(parPool))
	}
	attendBatchedHeads(q, parCtx, keys, vals, 0, cache, 0, startPos, K, true, arch, false, parPool)

	// Assert on the OUTPUT, not on a name: the doc comment above claims the two
	// arms agree bitwise, and this is the assertion that names that thing.
	for i := range serialCtx {
		if serialCtx[i] != parCtx[i] {
			t.Fatalf("f32 fan-out is not bit-identical: ctx[%d] serial=%v parallel=%v (head %d, row %d)",
				i, serialCtx[i], parCtx[i], (i%qDim)/hd, i/qDim)
		}
	}
	// Guard against the test passing on two buffers of zeros — a gather that
	// never ran would satisfy the loop above perfectly.
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

//go:build darwin

package metal

import (
	"runtime"
	"testing"
	"time"
)

// TestBatchKAmortization — the make-or-break speculation experiment. Does a batch-k W4A8 GEMM
// amortize the issue-bound nibble unpack, so ForwardN(k) on the GEMV costs ~<<k× Forward(1)?
// Times gemv_w4a8_sa_bk at k=1,2,4,8 on the gate/up matrix (17920×1536) and compares to k× the
// single-token cost. Sub-linear growth => speculation converts drafts into throughput.
func TestBatchKAmortization(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pBK, err := d.NewComputePipeline(lib, "gemv_w4a8_sa_bk")
	if err != nil {
		t.Fatalf("pipeline bk: %v", err)
	}
	pSA, err := d.NewComputePipeline(lib, "gemv_w4a8_sa")
	if err != nil {
		t.Fatalf("pipeline sa: %v", err)
	}

	const N, K = 17920, 1536 // gate/up
	wq := NewBufferUint32s(d, make([]uint32, N*(K/8)))
	sc := NewBufferU16s(d, make([]uint16, N*(K/32)))
	const kmax = 8
	aq := byteBuf(d, kmax*K)
	asc := NewBufferFloats(d, make([]float32, kmax))
	out := d.NewBufferLen(kmax * N)
	uK, uN := NewBufferU32(d, K), NewBufferU32(d, N)
	q := d.NewCommandQueue()

	prof := func(reps int, run func(int)) time.Duration {
		for range 4 {
			run(reps)
		}
		best := time.Hour
		for range 20 {
			t0 := time.Now()
			run(reps)
			if dt := time.Since(t0); dt < best {
				best = dt
			}
		}
		return best / time.Duration(reps)
	}

	// single-token Stage A baseline (the k=1 reference the loop path pays k× today).
	sa := prof(200, func(r int) { q.Run1DBatch(pSA, N*32, 256, r, wq, sc, aq, asc, out, uK) })
	t.Logf("single-token Stage A gate/up: %.1f us", float64(sa.Microseconds()))

	for _, k := range []int{1, 2, 4, 8} {
		uKK := NewBufferU32(d, uint32(k))
		tgBytes := k * K * 2 // k activation vectors × K shorts — only what k needs (occupancy)
		bk := prof(200, func(r int) { q.Run1DBatchTG(pBK, N*32, 256, r, tgBytes, wq, sc, aq, asc, out, uK, uN, uKK) })
		loop := time.Duration(k) * sa // what the current loop-over-Forward path costs for k tokens
		t.Logf("k=%d: batch-k %.1f us  vs  k×single %.1f us  =>  %.2fx speedup, %.2f us/token (%.1f%% of single)",
			k, float64(bk.Microseconds()), float64(loop.Microseconds()),
			float64(loop)/float64(bk), float64(bk.Microseconds())/float64(k), 100*float64(bk)/float64(k)/float64(sa))
	}
}

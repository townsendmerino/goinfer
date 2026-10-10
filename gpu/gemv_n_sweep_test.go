//go:build gpu

package gpu

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// TestGEMVNSweep times the GEMV at ONE N per process (GOINFER_GEMV_SWEEP_N), so each reading
// reflects a clean, isolated allocation the way a real model's load does: an in-process sweep
// over many N (allocating and releasing a differently-sized weight buffer for each, unlike a real
// model, which allocates its LM-head buffer once at load) read the same N up to 15x apart
// depending on what ran before it, an artifact of that method. This is the repo's
// separate-process-per-arm convention for exactly this contamination.
//
//	for n in 512 65536 100000 151936 200000; do GOINFER_GEMV_SWEEP_N=$n go test -tags gpu ./gpu/ -run TestGEMVNSweep -v; done
func TestGEMVNSweep(t *testing.T) {
	nStr := os.Getenv("GOINFER_GEMV_SWEEP_N")
	if nStr == "" {
		t.Skip("set GOINFER_GEMV_SWEEP_N (one N per process, deliberately -- see this test's own doc comment)")
	}
	var ns []int
	for part := range strings.SplitSeq(nStr, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			t.Fatalf("bad GOINFER_GEMV_SWEEP_N entry %q: %v", part, err)
		}
		ns = append(ns, n)
	}
	ctx, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer ctx.Close()

	const K = 1536
	act := randMat(K, 2)
	aq, aScales := linalg.QuantizeRowsInt8(act, 1, K)
	median := func(xs []float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	for _, n := range ns {
		weight := randMat(n*K, uint64(n+1))
		bq, bScales := linalg.QuantizeRowsInt8(weight, n, K)
		rm, err := ctx.UploadW8A8(bq, bScales, n, K)
		if err != nil {
			t.Fatalf("UploadW8A8 N=%d: %v", n, err)
		}
		for range 5 {
			if _, err := ctx.MatmulW8A8GEMV(aq, aScales[0], rm); err != nil {
				t.Fatalf("warmup N=%d: %v", n, err)
			}
		}
		const reps = 30
		times := make([]float64, reps)
		for i := range reps {
			t0 := time.Now()
			if _, err := ctx.MatmulW8A8GEMV(aq, aScales[0], rm); err != nil {
				t.Fatalf("N=%d: %v", n, err)
			}
			times[i] = float64(time.Since(t0).Microseconds())
		}
		med := median(times)
		gbps := float64(n) * float64(K) / (med * 1e-6) / 1e9
		fmt.Printf("=== GEMVSWEEP N=%d K=%d median_us=%.2f ns_per_wg=%.3f GBps=%.2f ===\n", n, K, med, med*1000/float64(n), gbps)
		rm.Release()
	}
}

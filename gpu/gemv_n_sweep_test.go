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

// TestGEMVNSweep is R10/G38's root-cause step. First run (in-process, many N in one binary) found a
// wild-looking cliff between N=65536 and N=100000 with the SAME N reading up to 15x apart depending
// on what ran before it -- which turned out to be an ARTIFACT of the sweep's own methodology
// (allocating and releasing a differently-sized weight buffer for every N in one process, unlike a
// real model, which allocates its LM-head buffer ONCE at load and never touches the allocator again).
// This variant runs ONE N per PROCESS (GOINFER_GEMV_SWEEP_N), matching this repo's own
// separate-process-per-arm convention for exactly this contamination reason, so each reading reflects
// a clean, isolated allocation the way a real model's own load does.
//
//	for n in 512 65536 100000 151936 200000; do GOINFER_GEMV_SWEEP_N=$n go test -tags gpu ./gpu/ -run TestGEMVNSweep -v; done
func TestGEMVNSweep(t *testing.T) {
	nStr := os.Getenv("GOINFER_GEMV_SWEEP_N")
	if nStr == "" {
		t.Skip("set GOINFER_GEMV_SWEEP_N (one N per process, deliberately -- see this test's own doc comment)")
	}
	var ns []int
	for _, part := range strings.Split(nStr, ",") {
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

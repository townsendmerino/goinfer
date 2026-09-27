package decoder

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// TestCPUBatchS0_matmulScaling is an EXPLORATORY probe for MC3c step 2's follow-on (a tuned small-M amd64 kernel for
// the batched CPU decode step; docs/tasks/task-concurrency-2026-09.md). It is not a gate. On a real checkpoint's layer-0
// weights it times the production matmul (decoder.matmul, which picks the kernel and layout) at M = 1, 2, 3, 4 and 8
// rows, and reports each against M separate M = 1 calls — the cost the batched step would pay with no amortisation.
// The ratio t(M) / t(1) is what batching buys: near 1 means the weights are read once for all M rows; near M means
// nothing is shared.
//
// GOINFER_CPUBATCH_S0_MODEL=<checkpoint> (int4). Reps interleave the M values, and the median is reported. Progress
// goes to stderr as it runs (run with -v).
func TestCPUBatchS0_matmulScaling(t *testing.T) {
	path := os.Getenv("GOINFER_CPUBATCH_S0_MODEL")
	if path == "" {
		t.Skip("exploratory: set GOINFER_CPUBATCH_S0_MODEL=<checkpoint>")
	}
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	lw := &m.w.Layers[0]
	lm := &m.w.LMHead
	if lm.Rows() == 0 {
		lm = &m.w.Embed
	}
	shapes := []struct {
		name string
		w    *linalg.WeightMat
	}{
		{"q", &lw.QProj}, {"k", &lw.KProj}, {"v", &lw.VProj}, {"o", &lw.OProj},
		{"gate", &lw.GateProj}, {"up", &lw.UpProj}, {"down", &lw.DownProj}, {"lm_head", lm},
	}
	Ms := []int{1, 2, 3, 4, 8}
	reps := 7
	rng := rand.New(rand.NewSource(1))
	start := time.Now()
	for si, sh := range shapes {
		K, N := sh.w.Cols(), sh.w.Rows()
		a := make([]float32, 8*K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		dst := make([]float32, 8*N)
		times := map[int][]float64{}
		iters := max(3, int(2e9/float64(N*K))) // ~2 G weight-bytes-equivalent per timing
		for r := range reps {
			for i := range Ms {
				M := Ms[(i+r)%len(Ms)]                    // rotate the order each rep
				matmul(m.be, sh.w, a[:M*K], dst[:M*N], M) // warm
				t0 := time.Now()
				for range iters {
					matmul(m.be, sh.w, a[:M*K], dst[:M*N], M)
				}
				times[M] = append(times[M], float64(time.Since(t0).Nanoseconds())/float64(iters))
			}
		}
		med := func(v []float64) float64 { s := append([]float64(nil), v...); sort.Float64s(s); return s[len(s)/2] }
		t1 := med(times[1])
		line := fmt.Sprintf("%-8s N=%-6d K=%-6d layout=%-10s t(1)=%8.0f us", sh.name, N, K, sh.w.Int4Layout(), t1/1e3)
		for _, M := range Ms[1:] {
			line += fmt.Sprintf("  t(%d)/t(1)=%.2f", M, med(times[M])/t1)
		}
		fmt.Fprintf(os.Stderr, "[s0 %6.1fs %d/%d] %s\n", time.Since(start).Seconds(), si+1, len(shapes), line)
		t.Log(line)
	}
}

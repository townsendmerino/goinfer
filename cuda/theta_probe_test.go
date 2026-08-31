//go:build cuda && goinfer_testhooks

package cuda

import (
	"math/rand"
	"os"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestThetaProbe_CUDA measures Theta — the marginal cost of one extra verify node,
// in units of one single-token target step — on the cgo-free CUDA resident path.
//
// WHY THIS EXISTS. decoder/spec_adaptive.go says Theta "is the relative cost of one
// extra verify node on *this backend* — measure it", ships 0.5 as the batched-CPU
// value, and nothing has ever measured it on a GPU backend. The resident path
// therefore runs the adaptive depth controller on a CPU constant. The error is in
// the conservative direction (it under-drafts), so it costs throughput rather than
// correctness — but on a verify that streams the weights ONCE for the whole block,
// the marginal node should be far cheaper than half a step, and the controller is
// plausibly drafting several times shallower than it should.
//
// METHOD, identical to the CPU control in decoder/theta_probe_test.go so the two
// numbers are comparable: seed a context of `depth` positions, then time ForwardN
// over n tokens for a ladder of n, truncating back to `depth` between every call.
// Theta = (least-squares slope of T(n)) / T(1). The CPU control reproduced 0.456 at
// depth 128 against the documented ~0.5, which is what licenses trusting this one.
//
// Timing note: ForwardN is one Submit/Poll, so the call is synchronous at the
// driver boundary and wall clock around it is the kernel time plus the launch glue
// — which is exactly the quantity the controller trades against.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_THETA_PROBE=1 \
//	  go test -tags "cuda goinfer_testhooks" ./ -run TestThetaProbe_CUDA -v
func TestThetaProbe_CUDA(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("GOINFER_THETA_PROBE") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 GOINFER_THETA_PROBE=1")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	for _, mdl := range []string{
		"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
		"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
	} {
		path := modelPath(mdl)
		if _, err := os.Stat(path); err != nil {
			t.Logf("skip %s: %v", mdl, err)
			continue
		}
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load %s: %v", mdl, err)
		}
		if !m.ResidentActive() {
			m.Close()
			t.Logf("skip %s: not resident (%s)", mdl, m.ResidentDecline())
			continue
		}
		rf := m.ResidentForwardForTest()
		if rf == nil {
			m.Close()
			t.Fatalf("%s: ResidentActive but no resident forward", mdl)
		}
		hidden, _, _, _, _, _, _ := m.Dims()
		rng := rand.New(rand.NewSource(7))
		emb := func() []float32 {
			e := make([]float32, hidden)
			for j := range e {
				e[j] = float32(rng.NormFloat64()) * 0.5
			}
			return e
		}
		widths := []int{1, 2, 3, 4, 6, 8, 12, 16}
		for _, depth := range []int{128, 512} {
			rf.Reset()
			for p := 0; p < depth; p++ {
				if _, err := rf.Forward(emb(), p); err != nil {
					t.Fatalf("%s seed at %d: %v", mdl, p, err)
				}
			}
			med := make([]float64, len(widths))
			for i, w := range widths {
				embs := make([][]float32, w)
				for j := range embs {
					embs[j] = emb()
				}
				const reps = 9
				samples := make([]float64, 0, reps)
				for r := 0; r < reps+2; r++ {
					rf.TruncateTo(depth)
					t0 := time.Now()
					if _, err := rf.ForwardN(embs, depth); err != nil {
						t.Fatalf("%s ForwardN(%d): %v", mdl, w, err)
					}
					if r >= 2 { // discard two warm-ups (JIT + caches)
						samples = append(samples, float64(time.Since(t0).Microseconds()))
					}
				}
				med[i] = medianF(samples)
			}
			slope, t1, theta := fitTheta(widths, med)
			t.Logf("CUDA %-38s depth=%4d  T(1)=%7.0f µs  slope=%7.1f µs/node  THETA=%.3f",
				mdl, depth, t1, slope, theta)
			for i, w := range widths {
				t.Logf("     n=%2d  T=%8.0f µs  (T(n)/T(1)=%5.2f)", w, med[i], med[i]/t1)
			}
		}
		m.Close()
	}
}

func fitTheta(widths []int, ys []float64) (slope, t1, theta float64) {
	var sx, sy, sxx, sxy float64
	n := float64(len(widths))
	for i, w := range widths {
		x, y := float64(w), ys[i]
		sx, sy, sxx, sxy = sx+x, sy+y, sxx+x*x, sxy+x*y
	}
	slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
	t1 = ys[0]
	return slope, t1, slope / t1
}

func medianF(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	c := append([]float64(nil), xs...)
	for i := range c {
		for j := i + 1; j < len(c); j++ {
			if c[j] < c[i] {
				c[i], c[j] = c[j], c[i]
			}
		}
	}
	return c[len(c)/2]
}

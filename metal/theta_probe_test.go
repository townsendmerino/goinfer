//go:build darwin && goinfer_testhooks

package metal

import (
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestThetaProbe_Metal measures Theta — the marginal cost of one extra verify node, in units
// of one single-token target step — on the Metal resident path. Method is identical to the
// CUDA probe (cuda/theta_probe_test.go) and the CPU control (decoder/theta_probe_test.go) so
// the three numbers are directly comparable: seed `depth` positions, then time ForwardN over a
// ladder of widths, and take Theta = (least-squares slope of T(n)) / T(1).
//
// decoder/spec_adaptive.go ships Theta = 0.5, calls it the batched-CPU value, and says
// "measure it". CPU measured 0.456, CUDA 0.155-0.251. Metal was unmeasured.
//
// WHAT TO EXPECT HERE, AND WHY IT IS NOT THE CUDA STORY. CUDA's low Theta comes from a verify
// that streams the weights ONCE for the whole block, so the marginal node is far cheaper than a
// step. Metal's ForwardN (metal/backend.go:187) is NOT a batched kernel — it is a plain loop of
// single-token Forward calls. If that is the whole story then T(n) = n*T(1) and Theta ~ 1.0,
// which would mean the controller is running Metal on a constant that is too LOW and therefore
// OVER-drafting — the opposite direction from CUDA, where 0.5 is too high and under-drafts.
// That is a prediction from reading the dispatch, and the point of this test is to measure it
// rather than assert it.
//
// TruncateTo is a NO-OP on Metal (metal/backend.go:213), unlike CUDA where the probe leans on
// it to hold context depth constant between timed calls. It is safe here for the reason its own
// comment gives — KV positions are overwritten on write and attention reads only keys[0..pos],
// so re-running ForwardN at the same startPos re-attends over the same span. It is called
// anyway, so the two probes stay line-for-line comparable.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_THETA_PROBE=1 \
//	  go test -tags "darwin goinfer_testhooks" ./metal/ -run TestThetaProbe_Metal -v
func TestThetaProbe_Metal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("GOINFER_THETA_PROBE") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 GOINFER_THETA_PROBE=1")
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
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
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
		// Shallow AND deep: Theta moved in OPPOSITE directions with depth on the two backends
		// already measured, so one cell would not characterise it.
		for _, depth := range []int{128, 512} {
			rf.Reset()
			for p := range depth {
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
				for r := range reps + 2 {
					rf.TruncateTo(depth)
					t0 := time.Now()
					if _, err := rf.ForwardN(embs, depth); err != nil {
						t.Fatalf("%s ForwardN(%d): %v", mdl, w, err)
					}
					if r >= 2 { // discard two warm-ups (pipeline compile + caches)
						samples = append(samples, float64(time.Since(t0).Microseconds()))
					}
				}
				med[i] = medianTheta(samples)
			}
			slope, t1, theta := fitThetaMetal(widths, med)
			t.Logf("Metal %-38s depth=%4d  T(1)=%7.0f µs  slope=%7.1f µs/node  THETA=%.3f",
				mdl, depth, t1, slope, theta)
			for i, w := range widths {
				t.Logf("     n=%2d  T=%8.0f µs  (T(n)/T(1)=%5.2f)", w, med[i], med[i]/t1)
			}
		}
		m.Close()
	}
}

func fitThetaMetal(widths []int, ys []float64) (slope, t1, theta float64) {
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

func medianTheta(xs []float64) float64 {
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

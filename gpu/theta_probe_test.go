//go:build gpu && goinfer_testhooks

package gpu

import (
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestThetaProbe_WebGPU measures Theta — the marginal cost of one extra verify node, in units of
// one single-token target step — on the cgo WebGPU resident path.
//
// WHY THIS EXISTS. decoder/spec_adaptive.go says Theta "is the relative cost of one extra verify
// node on *this backend* — measure it", ships 0.5 as the batched-CPU value, and CPU/CUDA/Metal
// have each had a real probe since (decoder/theta_probe_test.go, cuda/theta_probe_test.go,
// metal/theta_probe_test.go — docs/measurements/theta-per-backend-2026-09-01.md,
// theta-cuda-ab-2026-09-01.md) — P22 (docs/queue-performance.md) named WebGPU as the one backend
// still falling through to the unmeasured 0.5 default, filed open rather than assumed either way.
// The Metal probe found the domain check itself excluded Metal's real value (1.00–1.05) and
// silently substituted 0.5 — the WORST available choice, since a smaller Theta drafts deeper, not
// shallower (spec_adaptive.go's `Depth()` is monotone-decreasing in Theta). Whether WebGPU has the
// same shape of defect is exactly what this measures.
//
// METHOD, identical to the CPU control and the CUDA/Metal probes so all four numbers are directly
// comparable: seed a context of `depth` positions, then time ForwardN over n tokens for a ladder
// of n, truncating back to `depth` between every call. Theta = (least-squares slope of T(n)) /
// T(1). residentDecoder.TruncateTo is a documented no-op on this backend (gpu/residency.go: the
// cache is positional, Forward sets nKeys=pos+1, so entries past pos are simply never read and get
// overwritten next round) — safe for this probe specifically because every call passes the SAME
// startPos, so nothing past `depth` is ever read regardless of what a prior wider call left there.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_THETA_PROBE=1 go test -tags "gpu goinfer_testhooks" -run TestThetaProbe_WebGPU -v ./gpu/
func TestThetaProbe_WebGPU(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("GOINFER_THETA_PROBE") == "" {
		t.Skip("set GOINFER_THETA_PROBE=1")
	}
	newOrSkipHW(t).Close() // real-HW gate — a software adapter's timings would not answer P22

	for _, mdl := range []string{
		"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
		"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
	} {
		path := os.ExpandEnv("$HOME/models/" + mdl)
		if _, err := os.Stat(path); err != nil {
			t.Logf("skip %s: %v", mdl, err)
			continue
		}
		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4"})
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
					if r >= 2 { // discard two warm-ups (pipeline/shader caches)
						samples = append(samples, float64(time.Since(t0).Microseconds()))
					}
				}
				med[i] = medianF(samples)
			}
			slope, t1, theta := fitTheta(widths, med)
			t.Logf("WebGPU %-38s depth=%4d  T(1)=%7.0f µs  slope=%7.1f µs/node  THETA=%.3f",
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

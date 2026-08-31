package decoder

import (
	"context"
	"math"
	"os"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Theta — the marginal cost of one extra verify node, measured.
//
// spec_adaptive.go's own words: "Theta is the relative cost of one extra verify
// node on *this backend* — measure it." The shipped default is 0.5, described as
// the batched-CPU ForwardN value. It has never been measured on any backend; the
// GPU-resident path uses the CPU constant today (docs/spec/02).
//
// DEFINITION, so the CPU control and the CUDA probe compute the same number:
//
//	T(n)  = wall time of one verify pass over n tokens at a fixed context depth
//	Theta = dT/dn / T(1)
//
// i.e. the slope of T(n) expressed in units of one single-token target step. The
// AdaptiveDepth rule D = floor(ln Theta / ln alpha) is exactly "extend while the
// chance of reaching depth d beats the marginal node cost", so this is the
// quantity the controller wants.
//
// THIS FILE IS THE CONTROL. It measures Theta on the staged CPU path, where the
// design says it should come out ~0.5. If the instrument cannot reproduce the one
// value already believed, its answer on CUDA is not trustworthy either — that is
// the whole reason the control exists and runs first.
// ---------------------------------------------------------------------------

// thetaFit returns the least-squares slope of T(n) over the sampled widths, the
// measured T(1), and Theta = slope / T(1).
func thetaFit(widths []int, ns []float64) (slope, t1, theta float64) {
	var sx, sy, sxx, sxy float64
	n := float64(len(widths))
	for i, w := range widths {
		x, y := float64(w), ns[i]
		sx, sy, sxx, sxy = sx+x, sy+y, sxx+x*x, sxy+x*y
	}
	slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
	t1 = ns[0] // widths[0] must be 1
	return slope, t1, slope / t1
}

// TestThetaProbe_CPU measures Theta on the batched-CPU ForwardN path.
//
// Run: GOINFER_THETA_PROBE=1 go test ./decoder -run TestThetaProbe_CPU -v
func TestThetaProbe_CPU(t *testing.T) {
	if os.Getenv("GOINFER_THETA_PROBE") == "" {
		t.Skip("set GOINFER_THETA_PROBE=1 (loads a real model)")
	}
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v)", err)
	}
	ctx := context.Background()
	widths := []int{1, 2, 3, 4, 6, 8, 12, 16}

	for _, depth := range []int{128, 512} {
		cache := m.NewCache(depth + 64)
		seed := make([]int, depth)
		for i := range seed {
			seed[i] = 1 + i%97 // deterministic filler; timing is content-independent
		}
		if _, err := m.prefillLogits(ctx, seed, cache); err != nil {
			t.Skipf("prefill at depth %d: %v", depth, err)
		}
		med := make([]float64, len(widths))
		for i, w := range widths {
			seq := make([]int, w)
			for j := range seq {
				seq[j] = 5 + j
			}
			const reps = 7
			samples := make([]float64, 0, reps)
			for r := 0; r < reps+1; r++ {
				cache.TruncateTo(depth)
				t0 := time.Now()
				if _, err := m.forwardN(ctx, seq, cache); err != nil {
					t.Fatalf("forwardN(%d): %v", w, err)
				}
				if r > 0 { // discard the first as warm-up
					samples = append(samples, float64(time.Since(t0).Microseconds()))
				}
			}
			med[i] = median(samples)
		}
		slope, t1, theta := thetaFit(widths, med)
		t.Logf("CPU depth=%4d  T(1)=%8.0f µs  slope=%8.0f µs/node  THETA=%.3f", depth, t1, slope, theta)
		for i, w := range widths {
			t.Logf("    n=%2d  T=%9.0f µs  (T(n)/T(1)=%5.2f)", w, med[i], med[i]/t1)
		}
		if math.IsNaN(theta) || theta <= 0 {
			t.Fatalf("depth %d: nonsensical Theta %v — instrument is broken", depth, theta)
		}
	}
}

func median(xs []float64) float64 {
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

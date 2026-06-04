//go:build gpu

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestWebGPUBackend_matchesCPU validates the WebGPU matmul against the CPU
// reference on real hardware (run with `go test -tags gpu ./...`). It skips
// cleanly when no adapter is present (headless CI). The shapes are the
// decoder's actual M=1 projections plus a slice of the LM head.
func TestWebGPUBackend_matchesCPU(t *testing.T) {
	be, err := newWebGPUBackend("decoder")
	if err != nil {
		t.Skipf("no GPU backend: %v", err)
	}
	defer be.Close()
	t.Logf("GPU backend: %s", be.Name())

	rng := rand.New(rand.NewSource(1))
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	for _, s := range []struct {
		name    string
		M, K, N int
	}{
		{"qproj", 1, 640, 1024},
		{"oproj", 1, 1024, 640},
		{"gate", 1, 640, 2048},
		{"down", 1, 2048, 640},
		{"lmhead_slice", 1, 640, 8192},
	} {
		a := randVec(s.M * s.K)
		b := randVec(s.N * s.K)
		want := make([]float32, s.M*s.N)
		linalg.MatmulBT(a, b, want, s.M, s.K, s.N)
		got := make([]float32, s.M*s.N)
		be.MatmulBT(a, b, got, s.M, s.K, s.N)

		var maxRel float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			rel := d / (1 + math.Abs(float64(want[i])))
			if rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > 1e-3 {
			t.Errorf("%s (M=%d K=%d N=%d): max rel diff %.2e vs CPU, want ≤ 1e-3", s.name, s.M, s.K, s.N, maxRel)
		} else {
			t.Logf("%s: max rel diff %.2e", s.name, maxRel)
		}
		// Call again with a fresh activation: exercises the resident-weight
		// reuse path (the weight must NOT be re-uploaded, result still correct).
		a2 := randVec(s.M * s.K)
		linalg.MatmulBT(a2, b, want, s.M, s.K, s.N)
		be.MatmulBT(a2, b, got, s.M, s.K, s.N)
		for i := range want {
			if math.Abs(float64(got[i]-want[i]))/(1+math.Abs(float64(want[i]))) > 1e-3 {
				t.Errorf("%s reuse: result wrong after resident reuse", s.name)
				break
			}
		}
	}
}

package decoder

import (
	"github.com/townsendmerino/aikit/linalg"
	"math/rand"
	"testing"
)

// TestMatmulInt4_MConsistent verifies the int4 weight matmul gives bit-identical
// per-row results at M=1 (decode) and M=K (prefill). This holds only because both
// now run the W4A8 integer kernel (weightmat.go): before prefill was routed onto
// MatmulBTW4A8 it used the f32-activation MatmulBTQ4, so the two paths differed.
// The W4A8 dot product for a given (row, output) is independent of M, so the
// match is exact — this is the self-contained gate for the prefill→W4A8 switch
// and the prefill↔decode numerics-seam removal.
func TestMatmulInt4_MConsistent(t *testing.T) {
	const rows, cols, K = 64, 64, 5 // dims are multiples of int4GroupSize (32)
	rng := rand.New(rand.NewSource(1))
	uni := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}
	w := linalg.WrapF32(uni(rows*cols), rows, cols)
	w = quantizeWM(w, quantInt4)
	if w.Kind() != "int4" {
		t.Fatal("weights did not quantize to int4")
	}

	a := uni(K * cols)
	prefill := make([]float32, K*rows)
	matmul(nil, &w, a, prefill, K) // M=K (int4 path doesn't touch the backend)

	for i := range K {
		decode := make([]float32, rows)
		matmul(nil, &w, a[i*cols:(i+1)*cols], decode, 1) // M=1
		for j := range rows {
			if prefill[i*rows+j] != decode[j] {
				t.Fatalf("row %d out %d: prefill(M=%d)=%v != decode(M=1)=%v — int4 matmul not M-consistent",
					i, j, K, prefill[i*rows+j], decode[j])
			}
		}
	}
}

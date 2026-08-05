//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"
)

// TestMoEExpertW4A8_parity gates the indexed stacked-int4 expert GEMV: for several
// router slots it runs the GPU W4A8 kernel against refMatmulW4A8 (the CPU twin that
// dequantizes the exact f16 group scales the GPU reads). It exercises the stacking
// index (e = idx[slot], row = e*N+n), the nibble unpack, and the group scales.
func TestMoEExpertW4A8_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const nE, N, K = 6, 48, 64 // K is a multiple of the 32-wide W4A8 group
	const group = w4a8GroupSize
	ng := K / group
	rng := rand.New(rand.NewSource(7))

	nib := make([][]uint8, nE)
	sc := make([][]float32, nE)
	for e := 0; e < nE; e++ {
		ne := make([]uint8, N*K)
		for i := range ne {
			ne[i] = uint8(rng.Intn(16)) // nibble 0..15 (value−8)
		}
		se := make([]float32, N*ng)
		for i := range se {
			se[i] = float32(rng.Float64()*0.05 + 0.005)
		}
		nib[e], sc[e] = ne, se
	}
	stack, err := ctx.UploadStackedExpertsInt4(nib, sc, nE, N, K)
	if err != nil {
		t.Fatalf("UploadStackedExpertsInt4: %v", err)
	}
	defer stack.Release()
	if !stack.w4 {
		t.Fatal("stack.w4 not set")
	}

	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 127)
	}
	aScale := float32(0.02)

	idx := []int{5, 0, 3} // chosen experts per slot
	for slot := 0; slot < len(idx); slot++ {
		got, err := ctx.IndexedGEMVForTestInt4(stack, act, aScale, idx, slot)
		if err != nil {
			t.Fatalf("IndexedGEMVForTestInt4 slot %d: %v", slot, err)
		}
		e := idx[slot]
		want := refMatmulW4A8(act, aScale, nib[e], sc[e], N, K, group)
		var dot, na, nb, maxAbs float64
		for n := 0; n < N; n++ {
			dot += float64(got[n]) * float64(want[n])
			na += float64(got[n]) * float64(got[n])
			nb += float64(want[n]) * float64(want[n])
			if d := math.Abs(float64(got[n] - want[n])); d > maxAbs {
				maxAbs = d
			}
		}
		cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
		t.Logf("slot %d → expert %d: cosine %.6f maxAbs %.3e", slot, e, cos, maxAbs)
		if cos < 0.99999 {
			t.Errorf("slot %d expert %d: cosine %.6f < 0.99999", slot, e, cos)
		}
	}
}

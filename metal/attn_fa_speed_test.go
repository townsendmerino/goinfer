//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
	"time"
)

// TestAttentionFA_speedProbe is a throwaway A/B (not a gate, not wired into any suite the
// -run default picks up unless named): the shipped attention kernel vs attention_fa+combine, same
// synthetic K/V, same qwen2.5-1.5b shape, at R2's registered decision depths. Answers the only
// question gate (1) correctness cannot: is this worth measuring end to end at all. Min-of-N wall
// time around a GPU-synchronous readback (Floats()), tight-interleaved per depth.
func TestAttentionFA_speedProbe(t *testing.T) {
	if testing.Short() {
		t.Skip("speed probe, not a correctness gate")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pAttn, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		t.Fatalf("pipeline attention: %v", err)
	}
	pFA, err := d.NewComputePipeline(lib, "attention_fa")
	if err != nil {
		t.Fatalf("pipeline attention_fa: %v", err)
	}
	pCombine, err := d.NewComputePipeline(lib, "attention_fa_combine")
	if err != nil {
		t.Fatalf("pipeline attention_fa_combine: %v", err)
	}

	const nH, nKV, hd = 12, 2, 128
	G := nH / nKV
	kvDim := nKV * hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(1))

	for _, tc := range []struct {
		nKeys, splitFA int
	}{
		{128, 1},
		{256, 1},
		{384, 1},
		{512, 1},
		{512, 4},
		{512, 8},
		{512, 14},
		{768, 1},
		{768, 8},
		{768, 14},
		{1024, 1},
		{1024, 8},
		{1024, 14},
		{1536, 8},
		{1536, 14},
		{2048, 4},
		{2048, 8},
		{2048, 14},
		{2048, 20},
		{3900, 8},
		{3900, 14},
		{3900, 20},
		{3900, 28},
	} {
		nKeys := tc.nKeys
		q := randSlice(rng, nH*hd)
		kh := randU16Slice(rng, nKeys*kvDim)
		vh := randU16Slice(rng, nKeys*kvDim)

		qB := NewBufferFloats(d, q)
		kc, vc := NewBufferU16s(d, kh), NewBufferU16s(d, vh)
		outShipped := d.NewBufferLen(nH * hd)
		outFA := d.NewBufferLen(nH * hd)
		uNH, uNKV := NewBufferU32(d, uint32(nH)), NewBufferU32(d, uint32(nKV))
		uHd, uNKeys := NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(nKeys))
		uScale := NewBufferFloats(d, []float32{scale})
		uWin := NewBufferU32(d, 0)
		uSinks, uHasSink0 := NewBufferFloats(d, []float32{0}), NewBufferU32(d, 0)

		G32 := uint32(G)
		uG := NewBufferU32(d, G32)
		pStride := G * (hd + 2)
		nSplit := tc.splitFA
		uNSplit := NewBufferU32(d, uint32(nSplit))
		partial := d.NewBufferLen(nKV * nSplit * pStride)
		shmBytes := 128 * 6 * G * 4

		cq := d.NewCommandQueue()
		runShipped := func() time.Duration {
			t0 := time.Now()
			enc := cq.Begin()
			enc.Dispatch(pAttn, nH*128, 128, qB, kc, vc, outShipped, uNH, uNKV, uHd, uNKeys, uScale, uWin, uSinks, uHasSink0)
			enc.End()
			_ = outShipped.Floats() // sync
			return time.Since(t0)
		}
		runFA := func() time.Duration {
			t0 := time.Now()
			enc := cq.Begin()
			enc.DispatchTG(pFA, nKV*nSplit*128, 128, shmBytes, qB, kc, vc, partial, uNKV, uG, uNKeys, uScale, uWin, uNSplit)
			enc.Dispatch(pCombine, nH*hd, hd, partial, outFA, uG, uHd, uNSplit)
			enc.End()
			_ = outFA.Floats() // sync
			return time.Since(t0)
		}

		const n = 40
		var bestShipped, bestFA time.Duration
		for i := 0; i < n; i++ {
			if dt := runShipped(); i == 0 || dt < bestShipped {
				bestShipped = dt
			}
			if dt := runFA(); i == 0 || dt < bestFA {
				bestFA = dt
			}
		}
		ratio := float64(bestShipped) / float64(bestFA)
		t.Logf("K=%d splitFA=%d: shipped=%v attention_fa=%v ratio(shipped/fa)=%.3fx",
			nKeys, nSplit, bestShipped, bestFA, ratio)
	}
}

func randSlice(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64()) * 0.5
	}
	return v
}

func randU16Slice(rng *rand.Rand, n int) []uint16 {
	f := randSlice(rng, n)
	v := make([]uint16, n)
	for i := range f {
		v[i] = f32ToF16(f[i])
	}
	return v
}

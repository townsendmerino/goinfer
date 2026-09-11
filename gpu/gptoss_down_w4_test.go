//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"math/rand"
	"testing"
)

// TestGptOssDownW4_parity gates gpt-oss's down-projection combine on an INT4 stacked expert set.
// That is the layout Quant "int4" uploads, and the only one that fits gpt-oss-20b on an 8 GB card
// (audit-2026-09-10 C-06). It runs through gptOssDownPipelineFor, the resident builder's own kernel
// choice, and checks it against the CPU reference: dst + wgt·(W4A8 matmul + down bias).
//
// The matmul term is made to dominate the bias ON PURPOSE. G-07 is the record of a bias-dominated
// gate that could not see this term collapse.
func TestGptOssDownW4_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const nE, N, K = 6, 48, 64 // K is a multiple of the 32-wide W4A8 group
	const group = w4a8GroupSize
	ng := K / group
	rng := rand.New(rand.NewSource(9))
	nib := make([][]uint8, nE)
	sc := make([][]float32, nE)
	for e := range nE {
		ne := make([]uint8, N*K)
		for i := range ne {
			ne[i] = uint8(rng.Intn(16))
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
		t.Fatal("stack.w4 not set — this gate is about the int4 layout")
	}

	act := make([]int8, K)
	for i := range act {
		act[i] = int8(rng.Intn(255) - 127)
	}
	const aScale = float32(0.02)
	dbias := make([]float32, nE*N)
	for i := range dbias {
		dbias[i] = float32(rng.NormFloat64() * 0.05)
	}
	dstInit := make([]float32, N)
	for i := range dstInit {
		dstInit[i] = float32(rng.NormFloat64() * 0.1)
	}
	idx := []int{5, 0, 3}
	wgt := []float32{0.55, 0.3, 0.15}

	for slot := range idx {
		e := idx[slot]
		mm := refMatmulW4A8(act, aScale, nib[e], sc[e], N, K, group)
		want := make([]float32, N)
		var mmSq, bSq float64
		for n := range N {
			want[n] = dstInit[n] + wgt[slot]*(mm[n]+dbias[e*N+n])
			mmSq += float64(mm[n]) * float64(mm[n])
			bSq += float64(dbias[e*N+n]) * float64(dbias[e*N+n])
		}
		if math.Sqrt(mmSq) < 4*math.Sqrt(bSq) {
			t.Fatalf("slot %d: matmul term (%.3g) does not dominate the bias (%.3g) — a bias-dominated "+
				"fixture cannot see the term this gate exists for (G-07)", slot, math.Sqrt(mmSq), math.Sqrt(bSq))
		}
		got, err := ctx.GptOssDownForTest(stack, act, aScale, idx, wgt, dbias, dstInit, slot)
		if err != nil {
			t.Fatalf("GptOssDownForTest slot %d: %v", slot, err)
		}
		var dot, na, nb, maxAbs, peak float64
		for n := range N {
			dot += float64(got[n]) * float64(want[n])
			na += float64(got[n]) * float64(got[n])
			nb += float64(want[n]) * float64(want[n])
			maxAbs = math.Max(maxAbs, math.Abs(float64(got[n]-want[n])))
			peak = math.Max(peak, math.Abs(float64(want[n])))
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		t.Logf("slot %d → expert %d: cosine %.6f maxAbs %.3e (peak %.3g)", slot, e, cos, maxAbs, peak)
		if cos < 0.99999 || maxAbs > 1e-4*peak {
			t.Errorf("slot %d expert %d: cosine %.6f maxAbs %.3e — the gpt-oss down combine does not compute "+
				"the int4 expert's matmul (audit C-06)", slot, e, cos, maxAbs)
		}
	}
}

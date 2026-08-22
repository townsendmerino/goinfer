//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestRope2Kv_matchesRope2ThenKv gates the shipped rope2_kv kernel (kernels.go) — the fused
// RoPE+cache-store dispatch every hot decode call site would use instead of rope2 followed by
// kv_store — against running the ALREADY-PROVEN rope2 then kv_store sequence, the exact
// production pattern rope2_kv replaces. Compiles allKernels (the real shared library).
//
// GQA-realistic geometry (nH != nKV), same discipline as TestRope2_matchesTwoRope: a merge bug
// mixing up Q/K/V ranges or the kc/vc addressing would only show up with nH != nKV.
func TestRope2Kv_matchesRope2ThenKv(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pRope2, err := d.NewComputePipeline(lib, "rope2")
	if err != nil {
		t.Fatalf("pipeline rope2: %v", err)
	}
	pKv, err := d.NewComputePipeline(lib, "kv_store")
	if err != nil {
		t.Fatalf("pipeline kv_store: %v", err)
	}
	pFused, err := d.NewComputePipeline(lib, "rope2_kv")
	if err != nil {
		t.Fatalf("pipeline rope2_kv: %v", err)
	}

	const nH, nKV, hd, pos = 12, 2, 128, 37
	const half = hd / 2
	const scale = float32(0.85) // real YaRN-class value, exercises mscale threading too
	const cacheLen = 64         // KV cache capacity (positions), pos=37 must be within it
	qDim, kvDim := nH*hd, nKV*hd
	kOff, vOff := qDim, qDim+kvDim
	total := qDim + 2*kvDim // Q ‖ K ‖ V, matching r.qkv's production layout

	rng := rand.New(rand.NewSource(103))
	invf := make([]float32, half)
	for i := range invf {
		invf[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(hd)))
	}
	src := make([]float32, total)
	for i := range src {
		src[i] = rng.Float32()*2 - 1
	}

	q_ := d.NewCommandQueue()

	// (a) rope2 THEN kv_store — the production pattern rope2_kv replaces.
	twoDispatch := func() (qkv []float32, kc, vc []uint16) {
		xb := NewBufferFloats(d, src)
		kcb := NewBufferU16s(d, make([]uint16, cacheLen*kvDim))
		vcb := NewBufferU16s(d, make([]uint16, cacheLen*kvDim))
		q_.Run1D(pRope2, nH*half+nKV*half, 64,
			xb, NewBufferFloats(d, invf), NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(pos)),
			NewBufferU32(d, uint32(nH*half)), NewBufferU32(d, uint32(nKV*half)),
			NewBufferU32(d, uint32(half)), NewBufferFloats(d, []float32{scale}), NewBufferU32(d, uint32(kOff)))
		q_.Run1D(pKv, kvDim, 64,
			xb.At(kOff*4), xb.At(vOff*4), kcb, vcb, NewBufferU32(d, uint32(kvDim)), NewBufferU32(d, uint32(pos)))
		return xb.Floats(), kcb.U16s(), vcb.U16s()
	}

	// (b) ONE fused dispatch — rope2_kv on the whole, UNOFFSET buffer.
	oneDispatch := func() (qkv []float32, kc, vc []uint16) {
		xb := NewBufferFloats(d, src)
		kcb := NewBufferU16s(d, make([]uint16, cacheLen*kvDim))
		vcb := NewBufferU16s(d, make([]uint16, cacheLen*kvDim))
		q_.Run1D(pFused, nH*half+2*nKV*half, 64,
			xb, NewBufferFloats(d, invf), NewBufferU32(d, uint32(hd)), NewBufferU32(d, uint32(pos)),
			NewBufferU32(d, uint32(nH*half)), NewBufferU32(d, uint32(nKV*half)),
			NewBufferU32(d, uint32(half)), NewBufferFloats(d, []float32{scale}),
			NewBufferU32(d, uint32(kOff)), NewBufferU32(d, uint32(vOff)),
			kcb, vcb, NewBufferU32(d, uint32(kvDim)))
		return xb.Floats(), kcb.U16s(), vcb.U16s()
	}

	gotQkv, gotKc, gotVc := oneDispatch()
	wantQkv, wantKc, wantVc := twoDispatch()

	var worst float64
	at := -1
	for i := range total {
		if diff := math.Abs(float64(gotQkv[i] - wantQkv[i])); diff > worst {
			worst, at = diff, i
		}
	}
	mustFinite(t, "rope2_kv vs rope2+kv_store qkv max|diff|", worst)
	if worst != 0 {
		t.Errorf("rope2_kv qkv buffer differs from rope2+kv_store: max|diff| %.3e at index %d (got %v want %v)",
			worst, at, gotQkv[at], wantQkv[at])
	}

	if len(gotKc) != len(wantKc) || len(gotVc) != len(wantVc) {
		t.Fatalf("cache buffer length mismatch: kc got=%d want=%d, vc got=%d want=%d", len(gotKc), len(wantKc), len(gotVc), len(wantVc))
	}
	kcMism, vcMism := 0, 0
	for i := range gotKc {
		if gotKc[i] != wantKc[i] {
			kcMism++
		}
	}
	for i := range gotVc {
		if gotVc[i] != wantVc[i] {
			vcMism++
		}
	}
	if kcMism != 0 || vcMism != 0 {
		t.Errorf("rope2_kv cache writes differ from rope2+kv_store: kc mismatches=%d/%d vc mismatches=%d/%d",
			kcMism, len(gotKc), vcMism, len(gotVc))
	}
	t.Logf("rope2_kv (1 dispatch) vs rope2+kv_store (2 dispatches), nH=%d nKV=%d hd=%d: qkv max|diff|=%.3e, kc/vc exact match",
		nH, nKV, hd, worst)
}

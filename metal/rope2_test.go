//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestRope2_matchesTwoRope gates the shipped rope2 kernel (kernels.go) — the merged Q+K RoPE
// dispatch every production call site will use instead of two separate `rope` dispatches per
// layer — against running the ALREADY-PROVEN `rope` kernel twice (once at buffer offset 0 for Q,
// once at a Metal bind-time offset for K), the exact production pattern rope2 replaces. Compiles
// allKernels (the real shared library) so both kernels under test are the ones that ship.
//
// GQA-realistic geometry (nH != nKV), matching production: a merge that accidentally used nH for
// both Q and K's head count, or mixed up which range gets which offset, would only show up with
// nH != nKV — a nH==nKV control could pass by coincidence.
func TestRope2_matchesTwoRope(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pRope, err := d.NewComputePipeline(lib, "rope")
	if err != nil {
		t.Fatalf("pipeline rope: %v", err)
	}
	pRope2, err := d.NewComputePipeline(lib, "rope2")
	if err != nil {
		t.Fatalf("pipeline rope2: %v", err)
	}

	const nH, nKV, hd, pos = 12, 2, 128, 37
	const half = hd / 2
	const scale = float32(0.85) // real YaRN-class value, not the 1.0 no-op — exercises mscale threading too
	qDim, kvDim := nH*hd, nKV*hd
	total := qDim + 2*kvDim // Q ‖ K ‖ V, matching r.qkv's production layout

	rng := rand.New(rand.NewSource(97))
	invf := make([]float32, half)
	for i := range invf {
		invf[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(hd)))
	}
	src := make([]float32, total)
	for i := range src {
		src[i] = rng.Float32()*2 - 1
	}

	q_ := d.NewCommandQueue()

	// (a) TWO dispatches — the pattern every production call site uses today: rope on Q at
	// buffer offset 0, then rope again on K at a Metal bind-time offset (aikit/gpu Buffer.At).
	twoDispatch := func() []float32 {
		xb := d.NewBufferFloats(src)
		invfb := d.NewBufferFloats(invf)
		q_.Run1D(pRope, nH*half, 64,
			xb, invfb, d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(pos)),
			d.NewBufferU32(uint32(nH*half)), d.NewBufferU32(uint32(half)), d.NewBufferFloats([]float32{scale}))
		q_.Run1D(pRope, nKV*half, 64,
			xb.At(qDim*4), invfb, d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(pos)),
			d.NewBufferU32(uint32(nKV*half)), d.NewBufferU32(uint32(half)), d.NewBufferFloats([]float32{scale}))
		return xb.Floats()
	}

	// (b) ONE dispatch — rope2 on the whole, UNOFFSET buffer.
	oneDispatch := func() []float32 {
		xb := d.NewBufferFloats(src)
		q_.Run1D(pRope2, nH*half+nKV*half, 64,
			xb, d.NewBufferFloats(invf), d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(pos)),
			d.NewBufferU32(uint32(nH*half)), d.NewBufferU32(uint32(nKV*half)),
			d.NewBufferU32(uint32(half)), d.NewBufferFloats([]float32{scale}), d.NewBufferU32(uint32(qDim)))
		return xb.Floats()
	}

	got2 := oneDispatch()
	want := twoDispatch()

	// Q and K sections must match bit-for-bit — same math, no reduction, no reassociation freedom.
	var worst float64
	at := -1
	for i := 0; i < qDim+kvDim; i++ {
		if diff := math.Abs(float64(got2[i] - want[i])); diff > worst {
			worst, at = diff, i
		}
	}
	mustFinite(t, "rope2 vs two-rope max|diff|", worst)
	if worst != 0 {
		t.Errorf("rope2 vs two separate rope dispatches: max|diff| %.3e at index %d (got %v want %v) — want exactly 0, same math",
			worst, at, got2[at], want[at])
	}
	t.Logf("rope2 (1 dispatch) vs rope×2 (2 dispatches), nH=%d nKV=%d hd=%d: max|diff|=%.3e over Q+K", nH, nKV, hd, worst)

	// The V section (untouched by either kernel) must be byte-identical to the ORIGINAL input —
	// a kOff/offset bug in rope2 could silently corrupt V instead of just mis-rotating K.
	for i := qDim + kvDim; i < total; i++ {
		if got2[i] != src[i] {
			t.Fatalf("rope2 wrote into the V section at index %d: got %v, want untouched %v (kOff/offset bug)",
				i, got2[i], src[i])
		}
	}

	// Real YaRN-class scale (not 1.0) must actually differ from the unscaled result — same
	// discipline as TestRope_mscale, so a scale parameter silently dropped in the merge would fail.
	unscaledXb := d.NewBufferFloats(src)
	q_.Run1D(pRope2, nH*half+nKV*half, 64,
		unscaledXb, d.NewBufferFloats(invf), d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(pos)),
		d.NewBufferU32(uint32(nH*half)), d.NewBufferU32(uint32(nKV*half)),
		d.NewBufferU32(uint32(half)), d.NewBufferFloats([]float32{1.0}), d.NewBufferU32(uint32(qDim)))
	unscaled := unscaledXb.Floats()
	same := true
	for i := 0; i < qDim+kvDim; i++ {
		if math.Abs(float64(got2[i]-unscaled[i])) > 1e-4 {
			same = false
			break
		}
	}
	if same {
		t.Error("rope2 scale=0.85 output is indistinguishable from scale=1 — the scale parameter is not being applied")
	}
}

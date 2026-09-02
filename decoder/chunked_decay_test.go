package decoder

import (
	"math"
	"testing"
)

// N-01: THE CUMULATIVE DECAY UNDERFLOWED f32 AT REAL CHECKPOINT RATES.
//
// Both chunked scans built the decay as a running PRODUCT and then divided: c[i]/c[m]. Real
// checkpoints initialise A_log = log(U(1,16)) with dt in [0.001, 0.1], so a per-step decay around
// 0.2 gives 0.2^64 ≈ 1e-45 — below f32 min-normal. c[m] rounds to ZERO and the ratio is 0/0 = NaN.
//
// The equivalence tests could not see it: they drew A_log from [-2, 0] with seq <= 40, so the
// product never got small enough. That is the minimal-repro rule again — the benign regime removed
// the only dimension the defect lives in.
//
// This test is that regime, in the small: it demonstrates the failure of the PRODUCT form and the
// survival of the LOG form at the same rates, with no model and no scan — arithmetic is the whole
// mechanism, so arithmetic is what it checks.
func TestChunkedDecay_logFormSurvivesRealCheckpointRates(t *testing.T) {
	// A per-step decay a checkpoint really produces.
	const step = float32(0.2)
	const L = 96 // chunk >= 64 is where it bites; the old tests stopped at seq 40

	// The product form, as both scans had it.
	prod := make([]float32, L)
	acc := float32(1)
	for i := range L {
		acc *= step
		prod[i] = acc
	}
	// The log form, as they have it now.
	logc := make([]float32, L)
	var lacc float32
	for i := range L {
		lacc += float32(math.Log(float64(step)))
		logc[i] = lacc
	}

	// The premise: the product really does flush to zero in this regime.
	zeroAt := -1
	for i := range L {
		if prod[i] == 0 {
			zeroAt = i
			break
		}
	}
	if zeroAt < 0 {
		t.Fatalf("premise broke: the f32 product never underflowed over %d steps at decay %v, so "+
			"this test no longer describes N-01", L, step)
	}
	t.Logf("f32 cumulative product flushes to zero at step %d of %d (decay %v)", zeroAt, L, step)

	// The defect: past that point every ratio is NaN.
	i, m := L-1, zeroAt+10
	if m < i {
		if r := prod[i] / prod[m]; !math.IsNaN(float64(r)) {
			t.Errorf("premise broke: prod[%d]/prod[%d] = %v, expected NaN (0/0)", i, m, r)
		}
	}

	// The fix: the log form gives the true ratio, which for a decay of 0.2 over (i-m) steps is
	// 0.2^(i-m) — a finite number, and zero only when the real value underflows.
	for _, mm := range []int{0, 10, zeroAt, zeroAt + 10, L - 2} {
		if mm >= L-1 {
			continue
		}
		got := expf32(logc[L-1] - logc[mm])
		want := math.Pow(float64(step), float64(L-1-mm))
		if math.IsNaN(float64(got)) {
			t.Errorf("log form produced NaN for i=%d m=%d", L-1, mm)
			continue
		}
		if want > 1e-30 { // only compare where f32 can represent the answer at all
			if rel := math.Abs(float64(got)-want) / want; rel > 1e-3 {
				t.Errorf("i=%d m=%d: log form %v, want %v (rel %.2e)", L-1, mm, got, want, rel)
			}
		}
	}
}

// The log form must agree with the product form wherever the product is still VALID — otherwise
// the fix would have changed the answer in the benign regime the existing equivalence tests cover,
// which is how a numerics "fix" quietly becomes a numerics regression.
func TestChunkedDecay_logFormAgreesWhereTheProductIsValid(t *testing.T) {
	for _, step := range []float32{0.99, 0.9, 0.5} {
		var acc, lacc float32 = 1, 0
		for i := range 32 {
			acc *= step
			lacc += float32(math.Log(float64(step)))
			if acc < 1e-30 {
				break // the product is no longer trustworthy; nothing to compare against
			}
			got := expf32(lacc)
			if rel := math.Abs(float64(got-acc)) / float64(acc); rel > 1e-4 {
				t.Errorf("step %v at %d: log %v vs product %v (rel %.2e)", step, i, got, acc, rel)
			}
		}
	}
}

// THE SCAN ITSELF, AT A DECAY THE PRODUCT FORM CANNOT SURVIVE.
//
// The equivalence tests drive the whole layer, so their per-step decay is exp(A*dt) with dt derived
// from random weights — it never gets near the 0.2 that real checkpoints reach, which is exactly
// why they could not see N-01 and why extending their A_log range alone does NOT reproduce it (I
// tried; a reverted scan still passed). scanChunk takes the decay array directly, so this drives
// the regime rather than hoping to land in it.
//
// A reverted (product-form) scan produces NaN here at L=96 and this fails; the log form does not.
func TestScanChunk_realDecayDoesNotProduceNaN(t *testing.T) {
	const hk, hv, keyDim = 4, 4, 8
	const L = 96
	const decay = float32(0.2) // exp(A*dt) with A ~ -16, dt ~ 0.1 — the audit's own numbers

	conv := make([][]float32, L)
	gt := make([][]float32, L)
	beta := make([][]float32, L)
	for i := range L {
		conv[i] = make([]float32, 2*keyDim+hv)
		for j := range conv[i] {
			conv[i][j] = float32(math.Sin(float64(i*7+j))) * 0.5
		}
		gt[i] = []float32{decay}
		beta[i] = []float32{0.5}
	}
	core := make([][]float32, L)
	for i := range core {
		core[i] = make([]float32, hv)
	}
	S := make([]float32, hk*hv)

	scanChunk(core, S, conv, gt, beta, 0, L, 0, 0, hk, hv, keyDim, 1)

	for i := range core {
		for j, v := range core[i] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("position %d component %d is %v — the cumulative decay underflowed and the "+
					"ratio became 0/0 (audit-2026-09-02 N-01)", i, j, v)
			}
		}
	}
	for i, v := range S {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("carried state component %d is %v", i, v)
		}
	}
	// The premise: at this decay the f32 product really does reach zero inside the chunk, so a
	// product-form scan would have divided by it.
	acc := float32(1)
	zeroed := false
	for range L {
		acc *= decay
		if acc == 0 {
			zeroed = true
			break
		}
	}
	if !zeroed {
		t.Fatal("premise broke: the product no longer underflows at this decay and length, so this " +
			"test would pass for a reverted scan too")
	}
}

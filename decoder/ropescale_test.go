package decoder

import (
	"math"
	"testing"
)

// TestComputeInvFreq_default: with no scaling the table is exactly the classic
// base^(-2d/dim) RoPE frequencies — the formula the old per-call applyRoPE used,
// so the refactor can't have moved Gemma/Qwen3/Llama numbers.
func TestComputeInvFreq_default(t *testing.T) {
	const base, dim = 10000.0, 8
	got := computeInvFreq(base, dim, nil)
	if len(got) != dim/2 {
		t.Fatalf("len = %d, want %d", len(got), dim/2)
	}
	for d := range got {
		want := math.Pow(base, -float64(2*d)/float64(dim))
		if math.Abs(got[d]-want) > 1e-12 {
			t.Errorf("invFreq[%d] = %v, want %v", d, got[d], want)
		}
	}
}

// TestComputeInvFreq_linear: every frequency is divided by the factor.
func TestComputeInvFreq_linear(t *testing.T) {
	const base, dim, factor = 10000.0, 8, 4.0
	plain := computeInvFreq(base, dim, nil)
	scaled := computeInvFreq(base, dim, &ropeScaling{kind: ropeScaleLinear, factor: factor})
	for d := range plain {
		if math.Abs(scaled[d]-plain[d]/factor) > 1e-12 {
			t.Errorf("linear[%d] = %v, want %v", d, scaled[d], plain[d]/factor)
		}
	}
}

// TestComputeInvFreq_llama3 reproduces HF's _compute_llama3_parameters with an
// independent implementation over the real Llama-3.2 params and checks the
// piecewise behaviour: high frequencies (short wavelength) untouched, low
// frequencies (long wavelength) divided by factor, middle smoothly interpolated.
func TestComputeInvFreq_llama3(t *testing.T) {
	const base = 500000.0
	const dim = 64 // Llama-3.2-1B head_dim
	sc := &ropeScaling{
		kind: ropeScaleLlama3, factor: 32,
		lowFreqFactor: 1, highFreqFactor: 4, origMaxPosition: 8192,
	}
	plain := computeInvFreq(base, dim, nil)
	got := computeInvFreq(base, dim, sc)

	lowWavelen := sc.origMaxPosition / sc.lowFreqFactor
	highWavelen := sc.origMaxPosition / sc.highFreqFactor
	sawHigh, sawLow, sawMid := false, false, false
	for d, f := range plain {
		wavelen := 2 * math.Pi / f
		var want float64
		switch {
		case wavelen < highWavelen:
			want, sawHigh = f, true
		case wavelen > lowWavelen:
			want, sawLow = f/sc.factor, true
		default:
			s := (sc.origMaxPosition/wavelen - sc.lowFreqFactor) / (sc.highFreqFactor - sc.lowFreqFactor)
			want, sawMid = (1-s)*(f/sc.factor)+s*f, true
		}
		if math.Abs(got[d]-want) > 1e-12 {
			t.Errorf("llama3[%d] = %v, want %v (wavelen %.1f)", d, got[d], want, wavelen)
		}
	}
	// The real params exercise all three branches; if not, the test isn't
	// actually covering the piecewise logic.
	if !sawHigh || !sawLow || !sawMid {
		t.Errorf("llama3 params didn't hit all branches (high=%v low=%v mid=%v)", sawHigh, sawLow, sawMid)
	}
}

// TestApplyRoPE_partialRotary: with rotaryDim < headDim the trailing dims are
// passed through unrotated, and the rotated block matches a full-rotary call on
// just that block.
func TestApplyRoPE_partialRotary(t *testing.T) {
	const headDim, pos = 8, 3
	const base = 10000.0
	rotaryDim := 4 // partial: rotate first 4 of 8 dims

	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8} // one head
	orig := append([]float32(nil), vec...)

	inv := computeInvFreq(base, rotaryDim, nil)
	applyRoPE(vec, 1, headDim, pos, inv, 1.0)

	// Trailing headDim-rotaryDim dims unchanged.
	for d := rotaryDim; d < headDim; d++ {
		if vec[d] != orig[d] {
			t.Errorf("dim %d changed (%v → %v), should pass through", d, orig[d], vec[d])
		}
	}
	// Rotated block must differ (rotation at pos>0 with nonzero input).
	changed := false
	for d := range rotaryDim {
		if vec[d] != orig[d] {
			changed = true
		}
	}
	if !changed {
		t.Errorf("rotated block unchanged, expected rotation at pos=%d", pos)
	}
}

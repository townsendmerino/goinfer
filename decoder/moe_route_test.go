package decoder

import (
	"math"
	"testing"
)

// TestRouteExperts covers both MoE routing schemes model-free: Mixtral/Qwen2-MoE
// (softmax) unchanged, and the GLM/DeepSeek sigmoid + e_score_correction_bias path
// where the bias steers SELECTION but the weights are the un-biased scores.
func TestRouteExperts(t *testing.T) {
	logits := []float32{1, 2, 3, 0}

	// --- GLM path: sigmoid scoring, bias drops the top-scoring expert from selection ---
	// scores = sigmoid(logits) ≈ [.731, .881, .953, .5]; bias[-5] on expert 2 → its
	// selection score goes negative, so top-2 are {1,0}, NOT {2,1} — yet the weights
	// use expert 1/0's UN-biased sigmoid scores.
	bias := []float32{0, 0, -5, 0}
	idx, wts := routeExperts(logits, bias, 2, true, true, 1.0)
	sel := map[int]float32{}
	for j, e := range idx {
		sel[e] = wts[j]
	}
	if _, picked := sel[2]; picked {
		t.Errorf("expert 2 (highest raw score) should be biased OUT of selection; got idx=%v", idx)
	}
	if _, ok0 := sel[0]; !ok0 {
		t.Errorf("expert 0 should be selected; idx=%v", idx)
	}
	if _, ok1 := sel[1]; !ok1 {
		t.Errorf("expert 1 should be selected; idx=%v", idx)
	}
	// un-biased sigmoid scores renormalized: s0=.7311, s1=.8808 → /sum
	s0 := 1 / (1 + math.Exp(-1))
	s1 := 1 / (1 + math.Exp(-2))
	sum := s0 + s1
	if d := math.Abs(float64(sel[1]) - s1/sum); d > 1e-5 {
		t.Errorf("expert-1 weight = %v, want %v (un-biased, renormalized)", sel[1], s1/sum)
	}
	if d := math.Abs(float64(sel[0]) - s0/sum); d > 1e-5 {
		t.Errorf("expert-0 weight = %v, want %v", sel[0], s0/sum)
	}

	// --- Mixtral path: softmax + top-k + renorm, unchanged from the prior code ---
	mi, mw := routeExperts(logits, nil, 2, false, true, 0)
	// reference: the exact old path.
	probs := softmaxF32(logits)
	oi, ow := topK(probs, 2)
	var s float32
	for _, w := range ow {
		s += w
	}
	for j := range ow {
		ow[j] /= s
	}
	if len(mi) != len(oi) {
		t.Fatalf("mixtral idx len %d vs %d", len(mi), len(oi))
	}
	rm := map[int]float32{}
	for j, e := range mi {
		rm[e] = mw[j]
	}
	for j, e := range oi {
		if d := math.Abs(float64(rm[e] - ow[j])); d > 1e-6 {
			t.Errorf("mixtral expert %d weight %v != reference %v", e, rm[e], ow[j])
		}
	}

	// --- routed_scaling_factor multiplies the weights ---
	_, sw := routeExperts(logits, nil, 2, true, false, 2.5)
	_, sw1 := routeExperts(logits, nil, 2, true, false, 1.0)
	for j := range sw {
		if d := math.Abs(float64(sw[j] - 2.5*sw1[j])); d > 1e-5 {
			t.Errorf("scale: weight[%d]=%v, want 2.5×%v", j, sw[j], sw1[j])
		}
	}
}

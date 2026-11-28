package cpuengine

import (
	"math"
	"testing"
)

func TestSoftmaxF32(t *testing.T) {
	got := SoftmaxF32([]float32{1, 2, 3})
	// Reference softmax.
	var sum float64
	exp := make([]float64, 3)
	for i, v := range []float64{1, 2, 3} {
		exp[i] = math.Exp(v - 3)
		sum += exp[i]
	}
	var total float64
	for i := range got {
		want := exp[i] / sum
		if math.Abs(float64(got[i])-want) > 1e-6 {
			t.Errorf("softmax[%d] = %v, want %v", i, got[i], want)
		}
		total += float64(got[i])
	}
	if math.Abs(total-1) > 1e-5 {
		t.Errorf("softmax sums to %v, want 1", total)
	}
}

func TestTopK(t *testing.T) {
	// Descending by value; indices track the original positions.
	idx, val := TopK([]float32{0.1, 0.5, 0.2, 0.9, 0.3}, 2)
	if len(idx) != 2 || idx[0] != 3 || idx[1] != 1 {
		t.Errorf("TopK idx = %v, want [3 1]", idx)
	}
	if val[0] != 0.9 || val[1] != 0.5 {
		t.Errorf("TopK val = %v, want [0.9 0.5]", val)
	}

	// k == n returns every index, still sorted descending.
	idx2, _ := TopK([]float32{0.3, 0.1, 0.2}, 3)
	if len(idx2) != 3 || idx2[0] != 0 || idx2[1] != 2 || idx2[2] != 1 {
		t.Errorf("TopK(all) idx = %v, want [0 2 1]", idx2)
	}
}

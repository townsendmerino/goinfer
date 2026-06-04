package decoder

import (
	"math"
	"testing"
)

func TestSoftmaxF32(t *testing.T) {
	got := softmaxF32([]float32{1, 2, 3})
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
	idx, val := topK([]float32{0.1, 0.5, 0.2, 0.9, 0.3}, 2)
	if len(idx) != 2 || idx[0] != 3 || idx[1] != 1 {
		t.Errorf("topK idx = %v, want [3 1]", idx)
	}
	if val[0] != 0.9 || val[1] != 0.5 {
		t.Errorf("topK val = %v, want [0.9 0.5]", val)
	}

	// k == n returns every index, still sorted descending.
	idx2, _ := topK([]float32{0.3, 0.1, 0.2}, 3)
	if len(idx2) != 3 || idx2[0] != 0 || idx2[1] != 2 || idx2[2] != 1 {
		t.Errorf("topK(all) idx = %v, want [0 2 1]", idx2)
	}
}

// TestResolveArchitecture_mixtral: Mixtral is the llama descriptor with a MoE
// config (top-k of NumExperts) and the MoE tensor schema.
func TestResolveArchitecture_mixtral(t *testing.T) {
	cfg := validLlamaConfig()
	cfg.ModelType = "mixtral"
	cfg.NumLocalExperts = 8
	cfg.NumExpertsPerTok = 2
	a, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if a.Name != "mixtral" {
		t.Errorf("Name = %q, want mixtral", a.Name)
	}
	if a.MoE == nil || a.MoE.NumExperts != 8 || a.MoE.TopK != 2 {
		t.Fatalf("MoE = %+v, want 8x top-2", a.MoE)
	}
	if !a.MoE.NormTopKProb { // absent norm_topk_prob ⇒ HF default true
		t.Errorf("NormTopKProb = false, want true (default)")
	}
	if schema.Router == "" || schema.ExpertGate == "" {
		t.Errorf("mixtral schema missing MoE tensor names")
	}

	// k > E is rejected.
	cfg.NumExpertsPerTok = 9
	if _, _, err := resolveArchitecture(cfg); err == nil {
		t.Errorf("expected rejection for top-k > num_experts")
	}
}

package decoder

import "testing"

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

package decoder

import (
	"strings"
	"testing"
)

// A config whose feed-forward width is absurd would make newDecodeScratch request terabytes at the first cache, with no tensor in the
// file to give it away (an MoE layer has none of that width); validateConfigBounds refuses it before any allocation.
func TestValidateConfigBounds_refusesAnAbsurdFeedForwardWidth(t *testing.T) {
	real := func() *Config {
		return &Config{NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 8, VocabSize: 128256, IntermediateDim: 53248}
	}
	if err := validateConfigBounds(real()); err != nil {
		t.Fatalf("the widest real FFN (53248) refused: %v", err)
	}
	huge := 1 << 40
	for name, mutate := range map[string]func(*Config){
		"intermediate_size":                   func(c *Config) { c.IntermediateDim = huge },
		"moe_intermediate_size":               func(c *Config) { c.MoeIntermediateSize = huge },
		"shared_expert_intermediate_size":     func(c *Config) { c.SharedExpertIntermediateSize = huge },
		"moe_shared_expert_intermediate_size": func(c *Config) { c.MoeSharedExpertIntermediateSize = huge },
		"intermediate_size_mlp":               func(c *Config) { c.IntermediateSizeMLP = huge },
		"shared_intermediate_size":            func(c *Config) { c.SharedIntermediateSize = huge },
		"per-layer feed_forward_length":       func(c *Config) { c.FFNPerLayer = []int{1024, huge} },
	} {
		c := real()
		mutate(c)
		err := validateConfigBounds(c)
		if err == nil || !strings.Contains(err.Error(), "ceiling") {
			t.Errorf("%s = 2^40: err = %v, want a refusal naming the ceiling", name, err)
		}
	}
}

// The same through the one chokepoint every loader reaches, so the bound is not only a predicate nobody calls.
func TestResolveArchitecture_refusesAnAbsurdFeedForwardWidth(t *testing.T) {
	cfg := &Config{ModelType: "qwen3_moe", NumLayers: 2, HiddenDim: 64, NumHeads: 4, NumKVHeads: 2, VocabSize: 100, IntermediateDim: 1 << 40}
	if _, _, err := resolveArchitecture(cfg); err == nil || !strings.Contains(err.Error(), "ceiling") {
		t.Errorf("resolveArchitecture with intermediate_size 2^40: err = %v, want the ceiling refusal", err)
	}
}

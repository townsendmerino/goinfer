package decoder

import (
	"strings"
	"testing"
)

// M-10: two unbounded per-layer allocations from UNTRUSTED metadata — the M16 fatal-OOM gap,
// reopened on the paths that did not get ggufLayerCount.
//
// The failure mode is worth naming precisely, because it is not a panic: a count like
// 68719476736 asks for an allocation that is under Go's maxAlloc, so the runtime does not
// reject it — it tries, and the process dies with a FATAL "out of memory" that no recover()
// can catch. The .giw loader HAS a recover() and its doc promises a typed error; neither
// helps. So the bound has to come before the allocation, not around it.
//
// Checked at resolveArchitecture, which is the single point every source of config reaches —
// .giw, safetensors, GGUF — rather than at the two JSON call sites the audit names. Putting it
// at the callers would be the "one predicate, N consumers" shape that produced a large share
// of this audit's findings.
func TestValidateConfigBounds_hostileCountsAreRefusedBeforeAllocation(t *testing.T) {
	for name, tc := range map[string]struct {
		mutate func(*Config)
		names  string
	}{
		// The audit's own example: a 300-byte .giw declaring this is a fatal OOM today.
		"num_hidden_layers 2^36": {func(c *Config) { c.NumLayers = 1 << 36 }, "num_hidden_layers"},
		"hidden_size":            {func(c *Config) { c.HiddenDim = 1 << 40 }, "hidden_size"},
		"num_attention_heads":    {func(c *Config) { c.NumHeads = 1 << 30 }, "num_attention_heads"},
		"num_key_value_heads":    {func(c *Config) { c.NumKVHeads = 1 << 30 }, "num_key_value_heads"},
		"vocab_size":             {func(c *Config) { c.VocabSize = 1 << 40 }, "vocab_size"},
		"num_experts":            {func(c *Config) { c.NumExperts = 1 << 30 }, "num_experts"},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := &Config{}
			tc.mutate(cfg)
			err := validateConfigBounds(cfg)
			if err == nil {
				t.Fatalf("accepted; an adapter would allocate from this before anything else " +
					"ran, and the result is a fatal OOM rather than an error (M-10)")
			}
			if !strings.Contains(err.Error(), tc.names) {
				t.Errorf("error %q does not name %q", err, tc.names)
			}
		})
	}

	// A real model's dimensions must pass. Deliberately generous — larger than any open model
	// today — so the ceilings police hostile input and not legitimate growth.
	t.Run("a large real model passes", func(t *testing.T) {
		cfg := &Config{NumLayers: 126, HiddenDim: 18432, NumHeads: 128, NumKVHeads: 8,
			VocabSize: 256000, NumExperts: 256}
		if err := validateConfigBounds(cfg); err != nil {
			t.Errorf("a 405B-class config is rejected: %v", err)
		}
	})

	// Zero and negative are NOT this function's job — they are validateResolved's, after the
	// adapter has filled in what the config omits. Pinned so a later "tidy-up" that moves them
	// here does not start rejecting the MoE checkpoints that legitimately leave fields at zero.
	t.Run("zero is left to validateResolved", func(t *testing.T) {
		if err := validateConfigBounds(&Config{}); err != nil {
			t.Errorf("an all-zero config is rejected here: %v", err)
		}
	})
}

// The end-to-end shape of M-10(a): laguna's block_count reaches make([]string, numLayers).
// granitehybrid, nemotron and llama4 got ggufLayerCount when M16 was fixed; laguna did not.
func TestGGUFLayerCount_boundsTheLagunaPath(t *testing.T) {
	if _, err := ggufLayerCount(1 << 36); err == nil {
		t.Fatal("ggufLayerCount accepted 2^36")
	}
	if n, err := ggufLayerCount(48); err != nil || n != 48 {
		t.Fatalf("ggufLayerCount(48) = %d, %v — a real layer count must pass", n, err)
	}
}

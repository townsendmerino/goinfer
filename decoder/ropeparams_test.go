package decoder

import (
	"encoding/json"
	"testing"
)

// TestRopeParameters_singleBaseArchs is the regression gate for a LOAD-TIME break on modern
// upstream checkpoints.
//
// transformers >= 5.10 moved single-base RoPE out of the flat top-level rope_theta /
// rope_scaling fields into a rope_parameters object:
//
//	"rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
//
// phi3 handled that (parseRopeFlat) and gemma3/mellum handle the per-attention-type nesting
// (full_attention / sliding_attention). llama, mistral, qwen2 and qwen3 read ONLY the flat
// field, so every one of them REJECTED such a checkpoint outright:
//
//	decoder(llama): rope_theta must be >0, got 0
//
// That is not a niche path — it is any Llama/Mistral/Qwen safetensors saved by a current
// transformers. It was found by generating a tiny Mistral fixture with transformers 5.12 and
// watching it fail to load, while the committed phi3-tiny (same transformers, same config
// shape) loaded fine. Each arch has its own architecture func, which is exactly how one got
// the fix and the others did not; the shared backfillFlatRope helper is the answer to that,
// and this table is what stops the next single-base arch from re-opening it.
func TestRopeParameters_singleBaseArchs(t *testing.T) {
	const theta = 12345.0
	ropeParams, err := json.Marshal(map[string]any{"rope_theta": theta, "rope_type": "default"})
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		arch string
		mk   func() *Config
	}{
		{"llama", func() *Config { c := validLlamaConfig(); c.ModelType = "llama"; return c }},
		{"mistral", func() *Config {
			c := validLlamaConfig()
			c.ModelType = "mistral"
			c.SlidingWindow = 4096
			return c
		}},
		{"qwen2", func() *Config { c := validLlamaConfig(); c.ModelType = "qwen2"; return c }},
		{"qwen3", func() *Config { c := validLlamaConfig(); c.ModelType = "qwen3"; c.HeadDim = 4; return c }},
	}

	for _, c := range cases {
		t.Run(c.arch, func(t *testing.T) {
			cfg := c.mk()
			// The modern shape: rope lives ONLY in rope_parameters, the flat field is absent.
			cfg.RoPEGlobalBase = 0
			cfg.RopeParameters = ropeParams

			a, _, err := resolveArchitecture(cfg)
			if err != nil {
				t.Fatalf("%s config with rope_parameters (transformers >=5.10 format) failed to load: %v\n"+
					"Any %s checkpoint saved by a current transformers is unloadable.", c.arch, err, c.arch)
			}
			if got := a.RoPEGlobalBase; got != theta {
				t.Errorf("rope base = %v, want %v — rope_parameters was accepted but its rope_theta "+
					"was not actually used, so RoPE would run at the wrong frequency (silently wrong "+
					"output, which is worse than the load error this replaced)", got, theta)
			}
		})
	}
}

// TestRopeParameters_flatStillWins guards the other direction: an OLD config (flat rope_theta,
// no rope_parameters) must be untouched, and an explicit flat value must never be clobbered by
// the backfill. Every GGUF takes this path, as does every checkpoint saved before 5.10.
func TestRopeParameters_flatStillWins(t *testing.T) {
	cfg := validLlamaConfig()
	cfg.ModelType = "llama"
	cfg.RoPEGlobalBase = 500000 // explicit, flat — the pre-5.10 format
	// A rope_parameters that disagrees: the flat field is already set, so it must win.
	cfg.RopeParameters = json.RawMessage(`{"rope_theta": 10000.0, "rope_type": "default"}`)

	a, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if a.RoPEGlobalBase != 500000 {
		t.Errorf("rope base = %v, want 500000 — the backfill clobbered an explicit flat rope_theta; "+
			"it must only fill in what is ABSENT", a.RoPEGlobalBase)
	}
}

// TestRopeParameters_noneIsNoop: a config with neither form must still fail its own validation,
// not be silently rescued into a bogus base by the backfill.
func TestRopeParameters_noneIsNoop(t *testing.T) {
	cfg := validLlamaConfig()
	cfg.ModelType = "llama"
	cfg.RoPEGlobalBase = 0
	cfg.RopeParameters = nil
	if _, _, err := resolveArchitecture(cfg); err == nil {
		t.Error("a config with NO rope config at all resolved successfully — the backfill has started " +
			"inventing a base rather than filling in a declared one")
	}
}

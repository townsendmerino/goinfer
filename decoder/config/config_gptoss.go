package config

import (
	"fmt"
)

// ValidateGptOss pins the gpt-oss assumptions: a valid sparse MoE on every layer
// (num_local_experts / num_experts_per_tok / expert width) and a sliding window for
// the alternating local layers. The attention (GQA + biases + per-head sinks) and
// YaRN RoPE are validated by the generic head-dim checks + rope.ParseScaling at
// resolve time.
func (c *Config) ValidateGptOss() error {
	nExperts := c.NumLocalExperts
	if nExperts == 0 {
		nExperts = c.NumExperts
	}
	expInter := c.MoeIntermediateSize
	if expInter == 0 {
		expInter = c.IntermediateDim
	}
	switch {
	case c.NumLayers <= 0:
		return fmt.Errorf("decoder(gpt-oss): num_hidden_layers must be >0, got %d", c.NumLayers)
	case nExperts <= 0:
		return fmt.Errorf("decoder(gpt-oss): num_local_experts must be >0, got %d", nExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > nExperts:
		return fmt.Errorf("decoder(gpt-oss): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, nExperts)
	case expInter <= 0:
		return fmt.Errorf("decoder(gpt-oss): expert intermediate size must be >0, got %d", expInter)
	case c.SlidingWindow <= 0:
		return fmt.Errorf("decoder(gpt-oss): sliding_window must be >0, got %d", c.SlidingWindow)
	}
	return nil
}

package config

import (
	"fmt"
)

// ValidateMixtral pins the Mixtral assumptions: the llama dense constraints
// (reused) plus a valid MoE config — top-k experts of num_local_experts, both
// positive and k ≤ E.
func (c *Config) ValidateMixtral() error {
	if err := c.ValidateLlama(); err != nil {
		return err
	}
	switch {
	case c.NumLocalExperts <= 0:
		return fmt.Errorf("decoder(mixtral): num_local_experts must be >0, got %d", c.NumLocalExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumLocalExperts:
		return fmt.Errorf("decoder(mixtral): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumLocalExperts)
	}
	return nil
}

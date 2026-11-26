package config

import (
	"fmt"
	"slices"
)

// NormalizeNemotronBlocks expands NVIDIA's hybrid_override_pattern ("M" mamba, "*" attention, "-" mlp) into the
// layers_block_type spelling the rest of the loader reads, so a released NemotronH config.json (which carries only the
// pattern) and a transformers-instantiated one (which carries only layers_block_type) load identically.
//
// An unrecognized character is an error, not a mamba block: nemotronhArchitecture's kind switch defaults unknown strings to
// mamba, and inheriting that here would turn one typo in a 56-character string into a silently different model that still
// generates text.
func (c *Config) NormalizeNemotronBlocks() error {
	if len(c.LayersBlockType) > 0 || c.HybridOverridePattern == "" {
		return nil
	}
	types := make([]string, 0, len(c.HybridOverridePattern))
	for _, r := range c.HybridOverridePattern {
		switch r {
		case 'M':
			types = append(types, "mamba")
		case '*':
			types = append(types, "attention")
		case '-':
			types = append(types, "mlp")
		case 'E':
			// Nemotron 3 Nano's MoE FFN layer (sparse routed + shared expert), replacing the plain "-" dense-MLP block at this position.
			types = append(types, "moe")
		default:
			return fmt.Errorf("decoder(nemotron_h): hybrid_override_pattern has unknown block %q (want M, *, - or E)", r)
		}
	}
	c.LayersBlockType = types
	return nil
}

// ValidateNemotron pins the Nemotron-H (nemotron_h) assumptions: a layers_block_type
// covering every layer (mamba | attention | mlp | moe), a valid Mamba-2 geometry, a
// usable mlp width + layer-norm eps, and — only when the pattern actually contains an
// "moe" block (Nemotron 3 Nano; plain Nemotron-H never does) — a usable MoE shape.
func (c *Config) ValidateNemotron() error {
	switch {
	case len(c.LayersBlockType) != c.NumLayers:
		return fmt.Errorf("decoder(nemotron_h): layers_block_type has %d entries, want %d", len(c.LayersBlockType), c.NumLayers)
	case c.MambaNumHeads <= 0 || c.MambaHeadDim <= 0 || c.SSMStateSize <= 0 || c.ConvKernel <= 0 || c.NGroups <= 0:
		return fmt.Errorf("decoder(nemotron_h): bad mamba dims (num_heads=%d head_dim=%d state=%d conv=%d groups=%d)", c.MambaNumHeads, c.MambaHeadDim, c.SSMStateSize, c.ConvKernel, c.NGroups)
	case c.MambaNumHeads%c.NGroups != 0:
		return fmt.Errorf("decoder(nemotron_h): mamba_num_heads %d not divisible by n_groups %d", c.MambaNumHeads, c.NGroups)
	case c.IntermediateDim <= 0:
		return fmt.Errorf("decoder(nemotron_h): intermediate_size must be >0")
	case c.LayerNormEpsilon <= 0:
		return fmt.Errorf("decoder(nemotron_h): layer_norm_epsilon must be >0")
	}
	hasMoE := slices.Contains(c.LayersBlockType, "moe")
	if !hasMoE {
		return nil
	}
	switch {
	case c.NRoutedExperts <= 0:
		return fmt.Errorf("decoder(nemotron_h): n_routed_experts must be >0, got %d", c.NRoutedExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NRoutedExperts:
		return fmt.Errorf("decoder(nemotron_h): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NRoutedExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(nemotron_h): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case c.NSharedExperts > 0 && c.MoeSharedExpertIntermediateSize <= 0:
		return fmt.Errorf("decoder(nemotron_h): n_shared_experts=%d but moe_shared_expert_intermediate_size is unset", c.NSharedExperts)
	case c.NGroup > 0 && c.NRoutedExperts%c.NGroup != 0:
		return fmt.Errorf("decoder(nemotron_h): n_routed_experts %d not divisible by n_group %d", c.NRoutedExperts, c.NGroup)
	}
	return nil
}

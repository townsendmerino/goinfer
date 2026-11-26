package config

import (
	"fmt"
)

func (c *Config) ValidateGemma12(arch string) error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0 || c.IntermediateDim == 0:
		return fmt.Errorf("decoder(%s): missing required dim (hidden=%d layers=%d heads=%d head_dim=%d inter=%d)",
			arch, c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim, c.IntermediateDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(%s): num_attention_heads %d not a multiple of num_key_value_heads %d", arch, c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(%s): vocab_size is zero", arch)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(%s): rms_norm_eps must be >0, got %v", arch, c.RMSNormEps)
	case c.HiddenActivation != "" && c.HiddenActivation != "gelu_pytorch_tanh":
		return fmt.Errorf("decoder(%s): hidden_activation %q; the Gemma MLP here is GeGLU with the tanh GELU", arch, c.HiddenActivation)
	case c.FinalLogitSoftcap < 0 || c.AttnLogitSoftcap < 0:
		return fmt.Errorf("decoder(%s): negative softcap (final %v, attention %v)", arch, c.FinalLogitSoftcap, c.AttnLogitSoftcap)
	}
	return nil
}

package config

import (
	"fmt"
)

// ValidateGlmOcr rejects a config the glm_ocr descriptor does not implement, instead of loading it
// and computing something else. hd is the resolved head_dim.
func (c *Config) ValidateGlmOcr(hd int) error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || hd == 0:
		return fmt.Errorf("decoder(glm_ocr): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, hd)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(glm_ocr): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case hd%2 != 0:
		return fmt.Errorf("decoder(glm_ocr): head_dim %d is odd (pairwise rotation needs pairs)", hd)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(glm_ocr): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(glm_ocr): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(glm_ocr): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(glm_ocr): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.AttentionBias:
		return fmt.Errorf("decoder(glm_ocr): attention_bias=true unsupported (GlmOcrTextAttention is bias-free)")
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(glm_ocr): rope_parameters {rope_theta, mrope_section} is required")
	}
	return nil
}

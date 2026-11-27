package config

import "fmt"

// Inkling2Config is the Inkling 2 geometry beyond the flat fields (HiddenDim, NumLayers counts physical layers, heads,
// HeadDim, NumExperts, NumExpertsPerTok, MoeIntermediateSize, SlidingWindow, RMSNormEps, FinalLogitSoftcap,
// MoeRoutedScalingFactor). LoopPasses <= 1 is plain inkling2.
type Inkling2Config struct {
	SWA            []bool  `json:"swa"`
	DRel           int     `json:"d_rel"`
	RelExtent      int     `json:"rel_extent"`
	ConvK          int     `json:"conv_kernel_size"`
	BlockSize      int     `json:"n_layers_per_attn_res_block"`
	QKClip         float64 `json:"qk_l2_clip_threshold"`
	RelClip        float64 `json:"rel_l2_clip_threshold"`
	LogFloor       int     `json:"log_scaling_n_floor"`
	LogAlpha       float64 `json:"log_scaling_alpha"`
	SituBeta       float64 `json:"mlp_gate_softcap"`
	SituLinearBeta float64 `json:"mlp_up_softcap"`
	PaddedVocab    int     `json:"padded_vocab_size"`
	LoopStart      int     `json:"loop_start"`
	LoopEnd        int     `json:"loop_end"`
	LoopPasses     int     `json:"loop_passes"`
	GateRank       int     `json:"loop_gate_rank"`
}

// ValidateInkling2 checks the flat fields the Inkling 2 forward reads; the geometry itself is checked by the adapter.
func (c *Config) ValidateInkling2() error {
	switch {
	case c.Inkling2 == nil:
		return fmt.Errorf("decoder(inkling2): no Inkling 2 geometry (the GGUF reader fills it; a config.json load is not supported)")
	case c.HiddenDim <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 || c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(inkling2): hidden %d, layers %d, heads %d / %d KV heads are invalid", c.HiddenDim, c.NumLayers, c.NumHeads, c.NumKVHeads)
	case c.EffectiveHeadDim() <= 0:
		return fmt.Errorf("decoder(inkling2): head_dim %d must be positive", c.EffectiveHeadDim())
	case c.NumExperts <= 0 || c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts || c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(inkling2): %d experts, top %d, width %d are invalid", c.NumExperts, c.NumExpertsPerTok, c.MoeIntermediateSize)
	case c.SlidingWindow <= 0 || !(c.RMSNormEps > 0):
		return fmt.Errorf("decoder(inkling2): sliding_window %d and rms_norm_eps %v must be positive", c.SlidingWindow, c.RMSNormEps)
	case c.VocabSize <= 0 || c.Inkling2.PaddedVocab < c.VocabSize:
		return fmt.Errorf("decoder(inkling2): vocabulary %d with padded tables of %d rows", c.VocabSize, c.Inkling2.PaddedVocab)
	}
	return nil
}

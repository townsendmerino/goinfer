package config

import (
	"fmt"
)

// ValidateLFM2 checks the LFM2/LFM2.5 shape before anything is loaded. Each case is a real way a checkpoint can differ:
// layer_types drives which mixer every layer runs, conv_L_cache is the kernel width the rolling state is sized from, and
// conv_bias true would need a bias tensor no released checkpoint ships, so accepting it silently would drop a term.
func (c *Config) ValidateLFM2() error {
	switch {
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(lfm2): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	case c.ConvLCache <= 0:
		return fmt.Errorf("decoder(lfm2): conv_L_cache=%d must be >0 (kernel width / rolling-window depth)", c.ConvLCache)
	case c.ConvDim != 0 && c.ConvDim != c.HiddenDim:
		// Not a tolerance: upstream builds the conv on hidden_size and never reads this key,
		// so honouring a different value would compute a different model than the reference.
		return fmt.Errorf("decoder(lfm2): conv_dim=%d != hidden_size=%d (upstream Lfm2ShortConv uses hidden_size; a differing conv_dim cannot be honoured)", c.ConvDim, c.HiddenDim)
	case c.ConvBias:
		// Refused rather than ignored: the loader has no bias tensor to read, so honouring
		// the flag would mean silently computing a different block than the config asks for.
		return fmt.Errorf("decoder(lfm2): conv_bias=true unsupported (no released checkpoint sets it; the loader reads no bias tensor)")
	case c.NumHeads <= 0 || c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(lfm2): bad GQA (heads=%d kv_heads=%d)", c.NumHeads, c.NumKVHeads)
	case c.IntermediateDim <= 0:
		return fmt.Errorf("decoder(lfm2): intermediate_size=%d must be >0", c.IntermediateDim)
	case c.NormEps <= 0:
		// Every RMSNorm family validates eps > 0. LFM2 spells the key "norm_eps", so a checkpoint that omits it (or a parse that looks
		// for rms_norm_eps) must fail loudly rather than normalise by rsqrt(variance).
		return fmt.Errorf("decoder(lfm2): norm_eps=%v must be >0 (LFM2 spells it norm_eps, not rms_norm_eps)", c.NormEps)
	}
	for i, t := range c.LayerTypes {
		if t != "conv" && t != "full_attention" {
			return fmt.Errorf("decoder(lfm2): layer_types[%d]=%q (have: conv, full_attention)", i, t)
		}
	}
	return nil
}

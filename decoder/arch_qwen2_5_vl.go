package decoder

import (
	"encoding/json"
	"fmt"
)

// qwen2_5_vlArchitecture expresses the Qwen2.5-VL TEXT decoder: a Qwen2 dense decoder (q/k/v bias, GQA, RMSNorm, SwiGLU, derived head_dim)
// whose RoPE config lives under transformers-5.x's nested rope_parameters {mrope_section, rope_theta} instead of a top-level rope_theta.
// For text, m-RoPE degenerates to standard scalar RoPE (the 3 position components are equal), so the text path is exactly Qwen2; the 3D
// position machinery only engages on the image path. The vision_config is ignored here, like the Gemma 3 VL text side.
func qwen2_5_vlArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	// mrope_section + rope_theta live under EITHER the transformers-5.x nested
	// rope_parameters {mrope_section, rope_theta} (the tiny synthetic) OR the older
	// top-level rope_scaling {type:mrope, mrope_section} + rope_theta (the real 3B).
	// Extract from whichever is present.
	var section []int
	if len(cfg.RopeParameters) > 0 {
		var rp struct {
			RopeTheta    float64 `json:"rope_theta"`
			MRopeSection []int   `json:"mrope_section"`
		}
		if err := json.Unmarshal(cfg.RopeParameters, &rp); err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen2_5_vl): parse rope_parameters: %w", err)
		}
		section = rp.MRopeSection
		if cfg.RoPEGlobalBase == 0 {
			cfg.RoPEGlobalBase = rp.RopeTheta
		}
	}
	if len(cfg.RopeScaling) > 0 {
		var rs struct {
			Type         string `json:"type"`
			RopeType     string `json:"rope_type"`
			MRopeSection []int  `json:"mrope_section"`
		}
		if err := json.Unmarshal(cfg.RopeScaling, &rs); err != nil {
			return nil, nil, fmt.Errorf("decoder(qwen2_5_vl): parse rope_scaling: %w", err)
		}
		if rs.Type == "mrope" || rs.RopeType == "mrope" {
			if section == nil {
				section = rs.MRopeSection
			}
			// m-RoPE is not a parseRopeScaling kind (linear/llama3/yarn) — the
			// per-component split IS the "scaling". Clear it so qwen2Architecture's
			// parseRopeScaling doesn't reject rope_type "mrope".
			cfg.RopeScaling = nil
		}
	}
	if section == nil && len(cfg.MRopeSection) == 3 {
		section = cfg.MRopeSection // a .giw's config carries it under its own key (rope_scaling was cleared when it was written)
	}
	if len(section) != 3 {
		// Refuse: a Qwen-VL without its m-RoPE section would run plain RoPE on image positions and still look fine on text. A stale .giw sidecar that
		// dropped it then fails its self-check and rebuilds.
		return nil, nil, fmt.Errorf("decoder(qwen2_5_vl): no m-RoPE section (mrope_section in rope_scaling or rope_parameters); a .giw written before 2026-10-09 dropped it — rebuild it")
	}
	cfg.MRopeSection = section
	arch, schema, err := qwen2Architecture(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.Name = "qwen2_5_vl"
	arch.MRopeSection = section
	return arch, schema, nil
}

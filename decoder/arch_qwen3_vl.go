package decoder

import (
	"encoding/json"
	"fmt"
)

// qwen3_vlArchitecture expresses Qwen3-VL's TEXT decoder only: a Qwen3 dense decoder (per-head q/k RMSNorm, GQA, no q/k/v bias, derived
// head_dim: Qwen3VLTextAttention, not Qwen2's shape that qwen2_5_vlArchitecture aliases) whose RoPE config lives under the same nested
// rope_parameters {mrope_section, rope_theta} convention qwen2_5_vlArchitecture parses. For text, m-RoPE degenerates to standard scalar RoPE
// as for Qwen2.5-VL, so the text path is exactly Qwen3. Qwen3-VL's mrope_section maps frequency index to component through an INTERLEAVED
// formula, unlike Qwen2.5-VL's contiguous blocks (mropeComponentInterleaved vs mropeComponent, decoder/rope.go): irrelevant for a degenerate
// text position, load-bearing once an image path runs, so it is wired here. vision_config/deepstack_visual_indexes are ignored: no vision
// tower and no DeepStack injection in this adapter.
func qwen3_vlArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	section, err := qwen3VLMRopeSection(cfg, "qwen3_vl")
	if err != nil {
		return nil, nil, err
	}
	arch, schema, err := qwen3Architecture(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.Name = "qwen3_vl"
	arch.MRopeSection = section
	arch.MRopeInterleaved = true
	return arch, schema, nil
}

// qwen3VLMRopeSection reads a Qwen3-VL text_config's m-RoPE section (and rope_theta, when nested) for the family name, and
// clears rope_scaling of it (m-RoPE is not a parseRopeScaling kind). Shared by qwen3_vl and qwen3_vl_moe.
func qwen3VLMRopeSection(cfg *Config, name string) ([]int, error) {
	// Same extraction qwen2_5_vlArchitecture uses: mrope_section + rope_theta live under
	// EITHER the nested rope_parameters (the common transformers-5.x shape) OR the older
	// top-level rope_scaling {type:mrope, mrope_section} + rope_theta.
	var section []int
	if len(cfg.RopeParameters) > 0 {
		var rp struct {
			RopeTheta    float64 `json:"rope_theta"`
			MRopeSection []int   `json:"mrope_section"`
		}
		if err := json.Unmarshal(cfg.RopeParameters, &rp); err != nil {
			return nil, fmt.Errorf("decoder(%s): parse rope_parameters: %w", name, err)
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
			return nil, fmt.Errorf("decoder(%s): parse rope_scaling: %w", name, err)
		}
		// The released Qwen3-VL checkpoints write rope_scaling {mrope_interleaved: true, mrope_section: [...], rope_type: "default"}: m-RoPE is
		// marked by the section being there, not by the type, so a section present is taken whatever the type says.
		if rs.Type == "mrope" || rs.RopeType == "mrope" || len(rs.MRopeSection) > 0 {
			if section == nil {
				section = rs.MRopeSection
			}
			// m-RoPE is not a parseRopeScaling kind — clear it defensively, mirroring
			// qwen2_5_vlArchitecture, even though qwen3Architecture doesn't currently
			// call parseRopeScaling at all.
			cfg.RopeScaling = nil
		}
	}
	if section == nil && len(cfg.MRopeSection) == 3 {
		section = cfg.MRopeSection // a .giw's config carries it under its own key (rope_scaling was cleared when it was written)
	}
	if len(section) != 3 {
		// Refuse: a Qwen-VL without its m-RoPE section would run plain RoPE on image positions and still look fine on text. A stale .giw sidecar that
		// dropped it then fails its self-check and rebuilds.
		return nil, fmt.Errorf("decoder(%s): no m-RoPE section (mrope_section in rope_scaling or rope_parameters); a .giw written before 2026-10-09 dropped it — rebuild it", name)
	}
	cfg.MRopeSection = section
	return section, nil
}

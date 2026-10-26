package decoder

// qwen3VLMoeArchitecture expresses Qwen3-VL MoE's text decoder (Qwen3-VL-30B-A3B): qwen3_moe's attention, experts and
// router (128 experts, top 8, renormalised; no shared expert) with qwen3_vl's interleaved m-RoPE. The released checkpoint
// stores its experts in transformers 4.57's fused layout, which loadFusedExperts reads by shape. The vision tower and
// DeepStack are the dense Qwen3-VL's (aikit's tower, GenerateQwenVLDeepstack).
func qwen3VLMoeArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	section, err := qwen3VLMRopeSection(cfg, "qwen3_vl_moe")
	if err != nil {
		return nil, nil, err
	}
	if cfg.NumExperts == 0 { // the released config (transformers 4.57) says num_experts; a 5.x save writes num_local_experts
		cfg.NumExperts = cfg.NumLocalExperts
	}
	arch, schema, err := qwen3MoeArchitecture(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.Name = "qwen3_vl_moe"
	arch.MRopeSection = section
	arch.MRopeInterleaved = true
	return arch, schema, nil
}

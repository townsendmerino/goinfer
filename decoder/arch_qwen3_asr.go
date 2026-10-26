package decoder

// qwen3_asrArchitecture expresses Qwen3-ASR's text decoder: Qwen3 dense (per-head q/k RMSNorm, GQA, no q/k/v bias, head_dim 128) whose config carries Qwen3-VL's
// interleaved m-RoPE section ({mrope_section [24, 20, 20], interleaved}). For ASR the three position components are always equal (audio is sequential like text), and
// interleaved m-RoPE with equal components IS plain RoPE, so the text path is exactly qwen3 and the rope_scaling block is dropped here, as transformers' own Qwen3ASR text
// model (a plain Qwen3Model) never reads it. The audio encoder and projector are aikit's audio.QwenASREncoder; the soft-token splice is Model.GenerateAudio.
func qwen3_asrArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	cfg.RopeScaling = nil
	arch, schema, err := qwen3Architecture(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.Name = "qwen3_asr"
	return arch, schema, nil
}

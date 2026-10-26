package decoder

// voxtralArchitecture expresses Voxtral Mini's text decoder: Llama dense (GQA 32/8, an explicit head_dim of 128 that is NOT hidden/heads = 96, rope_theta 1e8, untied head), whose config
// nests it under text_config and whose tensors sit under language_model.* (the Gemma 3 VL layout the weights loader already strips). The audio tower and projector are
// aikit/audio.VoxtralAudio; the soft-token splice is Model.GenerateAudio.
func voxtralArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	arch, schema, err := llamaArchitecture(cfg)
	if err != nil {
		return nil, nil, err
	}
	arch.Name = "voxtral"
	return arch, schema, nil
}

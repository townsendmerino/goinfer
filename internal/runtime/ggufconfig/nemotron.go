package ggufconfig

import (
	"github.com/townsendmerino/aikit/embed"
)

// ggufNemotronConfig builds a Nemotron-H Config from the nemotron_h.* (or, for Nemotron 3 Nano's MoE variant,
// nemotron_h_moe.*) metadata. Per-layer kind comes from two parallel arrays: attention.head_count_kv (>0 means attention)
// and feed_forward_length (>0 means mlp for plain nemotron_h, moe for nemotron_h_moe, which never carries a plain
// dense-mlp layer); the rest are mamba. head_dim is attention.key_length (NOT embedding/heads). Attention is NoPE (the
// rope.* keys are vestigial). ModelType is normalized to "nemotron_h" either way: llama.cpp's GGUF arch string differs
// from HF's model_type (which stays "nemotron_h" for both the dense and MoE checkpoints), and goinfer's registry
// dispatches on the latter.
func ggufNemotronConfig(g *embed.GGUFFile) (*Config, error) {
	archStr, _ := g.Str("general.architecture") // "nemotron_h" or "nemotron_h_moe"
	isMoE := archStr == "nemotron_h_moe"
	u := func(k string) int {
		v, _ := g.Uint(archStr + "." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float(archStr + "." + k); ok {
			return v
		}
		v, _ := g.Uint(archStr + "." + k)
		return float64(v)
	}
	intAt := func(arr []any, i int) int { // tolerant int read from a metadata array
		if i >= len(arr) {
			return 0
		}
		switch v := arr[i].(type) {
		case uint32:
			return int(v)
		case int32:
			return int(v)
		case uint64:
			return int(v)
		case int64:
			return int(v)
		}
		return 0
	}
	nLayers, err := ggufLayerCount(u("block_count"))
	if err != nil {
		return nil, err
	}
	kvArr, _ := g.Metadata[archStr+".attention.head_count_kv"].([]any)
	ffArr, _ := g.Metadata[archStr+".feed_forward_length"].([]any)
	types := make([]string, nLayers)
	kvHeads, ffLen := 0, 0
	ffnKind := "mlp"
	if isMoE {
		ffnKind = "moe"
	}
	for i := range nLayers {
		switch {
		case intAt(kvArr, i) > 0:
			types[i] = "attention"
			kvHeads = intAt(kvArr, i)
		case intAt(ffArr, i) > 0:
			types[i] = ffnKind
			ffLen = intAt(ffArr, i)
		default:
			types[i] = "mamba"
		}
	}
	nHeadsMamba := u("ssm.time_step_rank")
	inner := u("ssm.inner_size")
	eps := gf("attention.layer_norm_rms_epsilon")
	if eps == 0 {
		eps = gf("attention.layer_norm_epsilon")
	}
	cfg := &Config{
		ModelType:        "nemotron_h",
		MaxPositions:     u("context_length"),
		HiddenDim:        u("embedding_length"),
		NumLayers:        nLayers,
		NumHeads:         u("attention.head_count"),
		NumKVHeads:       kvHeads,
		HeadDim:          u("attention.key_length"), // explicit — heads*head_dim != hidden
		IntermediateDim:  ffLen,
		MambaNumHeads:    nHeadsMamba,
		MambaHeadDim:     inner / nHeadsMamba,
		SSMStateSize:     u("ssm.state_size"),
		NGroups:          u("ssm.group_count"),
		ConvKernel:       u("ssm.conv_kernel"),
		LayersBlockType:  types,
		LayerNormEpsilon: eps,
		HiddenAct:        "silu",
		VocabSize:        ggufVocabSize(g),
	}
	if isMoE {
		// Nemotron 3 Nano's MoE fields: expert_count, expert_used_count, expert_feed_forward_length,
		// expert_shared_feed_forward_length, expert_shared_count, expert_weights_norm, expert_weights_scale, expert_group_count
		// and expert_group_used_count (the topk_group equivalent) are all present in the GGUF; the safetensors config's
		// topk_group has no GGUF key of the same name, which makes the last easy to assume absent.
		normTopK := true
		if b, ok := g.Metadata[archStr+".expert_weights_norm"].(bool); ok {
			normTopK = b
		}
		cfg.NRoutedExperts = u("expert_count")
		cfg.NumExpertsPerTok = u("expert_used_count")
		cfg.MoeIntermediateSize = u("expert_feed_forward_length")
		cfg.MoeSharedExpertIntermediateSize = u("expert_shared_feed_forward_length")
		cfg.NSharedExperts = u("expert_shared_count")
		cfg.NormTopKProb = &normTopK
		cfg.RoutedScalingFactor = gf("expert_weights_scale")
		cfg.NGroup = u("expert_group_count")
		cfg.TopkGroup = u("expert_group_used_count")
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

package decoder

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/embed"
)

// Gemma 1 (model_type "gemma"; Gemma, CodeGemma) and Gemma 2 (model_type "gemma2") — the two Gemma generations before
// gemma3. They ride the generic forward (runLayersFromEmbed / runLayersFromEmbedN): every piece is an existing
// Architecture switch except Gemma 2's attention-score softcap, which attendQuery / attendBatchedHeads apply
// (softcapScore).
//
// What both share with gemma3, read from transformers' modeling_gemma.py / modeling_gemma2.py: RMSNorm with a (1+w)
// weight (RMSAddOne), the sqrt(hidden) embedding scale, GeGLU with the tanh GELU (HF runs gelu_pytorch_tanh for
// Gemma 1 whatever its config's hidden_act says), a tied LM head, head_dim set explicitly (heads*head_dim != hidden on
// the 7B/9B), no q/k/v bias.
//
// Gemma 1: pre-norm only (input_layernorm, then post_attention_layernorm before the MLP), attention scaled by
// head_dim^-0.5, one RoPE base, no sliding window, no softcap. The 2B is MQA (one KV head).
//
// Gemma 2: gemma3's four sandwich norms (Sandwich4) without its QK-norm, attention scaled by query_pre_attn_scalar^-0.5,
// sliding and full layers alternating (layer_types, or every other layer from 0 when a config predates it) at ONE RoPE
// base, the attention-score softcap (attn_logit_softcapping, 50 on the releases) and the final-logit softcap
// (final_logit_softcapping, 30).

func (c *Config) validateGemma12(arch string) error {
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

func gemma12Rope(cfg *Config, arch string) (float64, error) {
	base, scaling, err := ropeBaseFlatOrNested(cfg, arch)
	if err != nil {
		return 0, err
	}
	if scaling != nil {
		return 0, fmt.Errorf("decoder(%s): rope scaling is set; no %s release uses it and this path does not implement it", arch, arch)
	}
	if base == 0 {
		base = 10_000 // the HF default for both generations
	}
	if base < 0 {
		return 0, fmt.Errorf("decoder(%s): rope_theta must be >0, got %v", arch, base)
	}
	return base, nil
}

// gemmaTensorSchema: Gemma 1's names are Llama's, with the head tied to the embedding.
var gemmaTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "", // tied
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	GateProj:    "mlp.gate_proj.weight",
	UpProj:      "mlp.up_proj.weight",
	DownProj:    "mlp.down_proj.weight",
}

// gemma2TensorSchema: gemma3's sandwich names without q_norm / k_norm.
var gemma2TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "", // tied
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "post_attention_layernorm.weight",
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "pre_feedforward_layernorm.weight",
	PostMLPNorm:  "post_feedforward_layernorm.weight",
}

func gemmaArchitecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256 // GemmaConfig's default; both releases set it
	}
	if err := cfg.validateGemma12("gemma"); err != nil {
		return nil, nil, err
	}
	if cfg.FinalLogitSoftcap != 0 || cfg.AttnLogitSoftcap != 0 || cfg.SlidingWindow != 0 {
		return nil, nil, fmt.Errorf("decoder(gemma): a softcap or sliding window is set; that is a Gemma 2 config under model_type gemma")
	}
	base, err := gemma12Rope(cfg, "gemma")
	if err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name: "gemma", HiddenDim: cfg.HiddenDim, NumLayers: cfg.NumLayers, NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads,
		HeadDim: cfg.HeadDim, IntermediateDim: cfg.IntermediateDim, VocabSize: cfg.VocabSize,
		Norm: NormRMS, RMSAddOne: true, NormEps: cfg.RMSNormEps, NormPlacement: NormPre2, Act: ActGeluTanh,
		AttnScale:     1 / math.Sqrt(float64(cfg.HeadDim)),
		RoPELocalBase: base, RoPEGlobalBase: base,
		EmbedScale: math.Sqrt(float64(cfg.HiddenDim)), TiedLMHead: true,
	}, &gemmaTensorSchema, nil
}

func gemma2Architecture(cfg *Config) (*Architecture, *tensorSchema, error) {
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256
	}
	if cfg.QueryPreAttnScalar == 0 {
		cfg.QueryPreAttnScalar = 256 // Gemma2Config's default
	}
	if len(cfg.LayerTypes) == 0 && cfg.SlidingWindowPattern == 0 {
		cfg.SlidingWindowPattern = 2 // before layer_types: every other layer from 0 is sliding (HF's layer_idx % 2 == 0)
	}
	if err := cfg.validateGemma12("gemma2"); err != nil {
		return nil, nil, err
	}
	if cfg.SlidingWindow <= 0 {
		return nil, nil, fmt.Errorf("decoder(gemma2): sliding_window must be >0, got %d", cfg.SlidingWindow)
	}
	for i, t := range cfg.LayerTypes {
		if t != "sliding_attention" && t != "full_attention" {
			return nil, nil, fmt.Errorf("decoder(gemma2): layer_types[%d] = %q", i, t)
		}
	}
	if len(cfg.LayerTypes) > 0 && len(cfg.LayerTypes) != cfg.NumLayers {
		return nil, nil, fmt.Errorf("decoder(gemma2): %d layer_types for %d layers", len(cfg.LayerTypes), cfg.NumLayers)
	}
	base, err := gemma12Rope(cfg, "gemma2")
	if err != nil {
		return nil, nil, err
	}
	return &Architecture{
		Name: "gemma2", HiddenDim: cfg.HiddenDim, NumLayers: cfg.NumLayers, NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads,
		HeadDim: cfg.HeadDim, IntermediateDim: cfg.IntermediateDim, VocabSize: cfg.VocabSize,
		Norm: NormRMS, RMSAddOne: true, NormEps: cfg.RMSNormEps, NormPlacement: NormSandwich4, Act: ActGeluTanh,
		AttnScale:     math.Pow(cfg.QueryPreAttnScalar, -0.5),
		SlidingWindow: cfg.SlidingWindow, layerIsGlobal: cfg.IsGlobalLayer,
		RoPELocalBase: base, RoPEGlobalBase: base,
		EmbedScale: math.Sqrt(float64(cfg.HiddenDim)), TiedLMHead: true,
		FinalLogitSoftcap: cfg.FinalLogitSoftcap, AttnLogitSoftcap: cfg.AttnLogitSoftcap,
	}, &gemma2TensorSchema, nil
}

// ggufGemma12Config builds a Gemma 1 (GGUF arch "gemma") or Gemma 2 ("gemma2") Config from llama.cpp's metadata. The
// GGUF stores the dims, the RMS epsilon, the Gemma 2 sliding window and both softcaps, and llama.cpp's converter adds
// 1 to every norm weight (the loader's vnorm subtracts it back, as for gemma3). What it does not store, mirrored from
// llama.cpp's own loader (llama-model.cpp) rather than guessed:
//   - Gemma 2's query_pre_attn_scalar: llama.cpp sets the attention scale to 1/sqrt(hidden/heads) on the 27B (46
//     layers) and 1/sqrt(head_dim) otherwise, which is the HF configs' 144 and 256;
//   - the sliding pattern: every other layer from 0 (set_swa_pattern(2));
//   - the RoPE base when rope.freq_base is absent: 10000.
func ggufGemma12Config(g *embed.GGUFFile, arch string) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint(arch + "." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:        arch,
		MaxPositions:     u("context_length"),
		HiddenDim:        u("embedding_length"),
		NumLayers:        u("block_count"),
		NumHeads:         u("attention.head_count"),
		NumKVHeads:       u("attention.head_count_kv"),
		HeadDim:          u("attention.key_length"),
		IntermediateDim:  u("feed_forward_length"),
		HiddenActivation: "gelu_pytorch_tanh",
		VocabSize:        ggufVocabSize(g),
	}
	if cfg.NumKVHeads == 0 {
		cfg.NumKVHeads = cfg.NumHeads
	}
	if eps, ok := g.Float(arch + ".attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float(arch + ".rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	}
	if arch == "gemma2" {
		cfg.SlidingWindow = u("attention.sliding_window")
		cfg.SlidingWindowPattern = 2
		if v, ok := g.Float("gemma2.attn_logit_softcapping"); ok {
			cfg.AttnLogitSoftcap = v
		}
		if v, ok := g.Float("gemma2.final_logit_softcapping"); ok {
			cfg.FinalLogitSoftcap = v
		}
		if cfg.NumLayers == 46 && cfg.NumHeads > 0 {
			cfg.QueryPreAttnScalar = float64(cfg.HiddenDim / cfg.NumHeads)
		} else {
			cfg.QueryPreAttnScalar = float64(cfg.HeadDim)
		}
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

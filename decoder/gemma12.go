package decoder

import (
	"fmt"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"math"
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
var gemmaTensorSchema = archparts.TensorSchema{
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
var gemma2TensorSchema = archparts.TensorSchema{
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

func gemmaArchitecture(cfg *Config) (*darch.Architecture, *archparts.TensorSchema, error) {
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256 // GemmaConfig's default; both releases set it
	}
	if err := cfg.ValidateGemma12("gemma"); err != nil {
		return nil, nil, err
	}
	if cfg.FinalLogitSoftcap != 0 || cfg.AttnLogitSoftcap != 0 || cfg.SlidingWindow != 0 {
		return nil, nil, fmt.Errorf("decoder(gemma): a softcap or sliding window is set; that is a Gemma 2 config under model_type gemma")
	}
	base, err := gemma12Rope(cfg, "gemma")
	if err != nil {
		return nil, nil, err
	}
	return &darch.Architecture{
		Name: "gemma", HiddenDim: cfg.HiddenDim, NumLayers: cfg.NumLayers, NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads,
		HeadDim: cfg.HeadDim, IntermediateDim: cfg.IntermediateDim, VocabSize: cfg.VocabSize,
		Norm: darch.NormRMS, RMSAddOne: true, NormEps: cfg.RMSNormEps, NormPlacement: darch.NormPre2, Act: darch.ActGeluTanh,
		AttnScale:     1 / math.Sqrt(float64(cfg.HeadDim)),
		RoPELocalBase: base, RoPEGlobalBase: base,
		EmbedScale: math.Sqrt(float64(cfg.HiddenDim)), TiedLMHead: true,
	}, &gemmaTensorSchema, nil
}

func gemma2Architecture(cfg *Config) (*darch.Architecture, *archparts.TensorSchema, error) {
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 256
	}
	if cfg.QueryPreAttnScalar == 0 {
		cfg.QueryPreAttnScalar = 256 // Gemma2Config's default
	}
	if len(cfg.LayerTypes) == 0 && cfg.SlidingWindowPattern == 0 {
		cfg.SlidingWindowPattern = 2 // before layer_types: every other layer from 0 is sliding (HF's layer_idx % 2 == 0)
	}
	if err := cfg.ValidateGemma12("gemma2"); err != nil {
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
	return &darch.Architecture{
		Name: "gemma2", HiddenDim: cfg.HiddenDim, NumLayers: cfg.NumLayers, NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads,
		HeadDim: cfg.HeadDim, IntermediateDim: cfg.IntermediateDim, VocabSize: cfg.VocabSize,
		Norm: darch.NormRMS, RMSAddOne: true, NormEps: cfg.RMSNormEps, NormPlacement: darch.NormSandwich4, Act: darch.ActGeluTanh,
		AttnScale:     math.Pow(cfg.QueryPreAttnScalar, -0.5),
		SlidingWindow: cfg.SlidingWindow, LayerIsGlobal: cfg.IsGlobalLayer,
		RoPELocalBase: base, RoPEGlobalBase: base,
		EmbedScale: math.Sqrt(float64(cfg.HiddenDim)), TiedLMHead: true,
		FinalLogitSoftcap: cfg.FinalLogitSoftcap, AttnLogitSoftcap: cfg.AttnLogitSoftcap,
	}, &gemma2TensorSchema, nil
}

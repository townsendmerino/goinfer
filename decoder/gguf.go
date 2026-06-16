package decoder

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// GGUF loading — read a quantized llama.cpp checkpoint and
// run it through the generic forward. The GGUF file carries both the
// architecture config (metadata) and the weights (dequantized from the mmap,
// then optionally re-quantized to resident int8/int4 per the quant mode), so no
// separate config.json/safetensors is needed. Layout quirks vs the HF safetensors
// path are normalized at load: tensors use llama.cpp's blk.N.* names; the q/k
// projections of NORM-rope archs (llama/mellum) are in llama.cpp's interleaved-
// RoPE permutation, inverted here to this package's rotate_half order (NEOX-rope
// archs — qwen/gemma — are left as-is, see ggufQKPermuted); and Gemma's (1+w)
// norm offset, baked into the stored weights by llama.cpp, is subtracted back out.
//
// Architectures: llama, qwen2, qwen3, gemma3, mellum. Quant types: F32/F16,
// Q8_0/Q4_0/Q5_0, the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, and IQ4_NL/IQ4_XS.

// ggufConfig synthesizes a Config from GGUF metadata, dispatching on
// general.architecture. The resolved Config feeds resolveArchitecture, so a
// GGUF reuses the same per-family adapter as the safetensors path.
func ggufConfig(g *embed.GGUFFile) (*Config, error) {
	arch, _ := g.Str("general.architecture")
	switch arch {
	case "llama":
		return ggufLlamaConfig(g)
	case "qwen2":
		return ggufQwen2Config(g)
	case "qwen3":
		return ggufQwen3Config(g)
	case "gemma3", "gemma3_text":
		return ggufGemmaConfig(g)
	case "gemma4":
		return ggufGemma4Config(g)
	case "mellum":
		return ggufMellumConfig(g)
	case "qwen35moe":
		return ggufQwen35Config(g)
	case "glm4moe":
		return ggufGlm4MoeConfig(g)
	default:
		return nil, fmt.Errorf("decoder(gguf): architecture %q unsupported (have: llama, qwen2, qwen3, gemma3, gemma4 [wip], mellum, qwen35moe, glm4moe)", arch)
	}
}

// ggufVocabSize reads the vocab size from the embedding tensor's dims (no
// dequant) or, failing that, the tokens metadata array length.
func ggufVocabSize(g *embed.GGUFFile) int {
	if dims, ok := g.Dims("token_embd.weight"); ok && len(dims) == 2 {
		return dims[1] // [in=hidden, out=vocab]
	}
	if toks, ok := g.Metadata["tokenizer.ggml.tokens"].([]any); ok {
		return len(toks)
	}
	return 0
}

// ggufEOS sets cfg.EOSTokenID from the GGUF tokenizer metadata (resolveEOSIDs
// falls back to this when there's no generation_config.json next to the .gguf).
func ggufEOS(g *embed.GGUFFile, cfg *Config) {
	if eos, ok := g.Uint("tokenizer.ggml.eos_token_id"); ok {
		cfg.EOSTokenID = fmt.Appendf(nil, "%d", eos)
	}
}

func ggufLlamaConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint(k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "llama",
		HiddenDim:       u("llama.embedding_length"),
		NumLayers:       u("llama.block_count"),
		NumHeads:        u("llama.attention.head_count"),
		NumKVHeads:      u("llama.attention.head_count_kv"),
		IntermediateDim: u("llama.feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("llama.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("llama.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 10000 // llama.cpp default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufQwen2Config builds a Qwen2/Qwen2.5 Config from the qwen2.* metadata. It is
// the llama config plus the qwen2 model type — the only architectural difference
// is the additive q/k/v projection bias, carried by the qwen2 descriptor
// (QKVBias) and loaded by buildWeightsFromGGUF. RoPE base defaults to llama.cpp's
// qwen2 default (1e6) when absent.
func ggufQwen2Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("qwen2." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "qwen2",
		HiddenDim:       u("embedding_length"),
		NumLayers:       u("block_count"),
		NumHeads:        u("attention.head_count"),
		NumKVHeads:      u("attention.head_count_kv"),
		IntermediateDim: u("feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen2.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("qwen2.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // llama.cpp qwen2 default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufQwen3Config builds a Qwen3 dense Config from the qwen3.* metadata. Versus
// qwen2 it drops the q/k/v bias and adds QK-norm (per-head RMSNorm over head_dim
// before RoPE) — both carried by the qwen3 descriptor, so the GGUF loader only
// needs the metadata mapping (head_dim is explicit via attention.key_length, not
// derived). NEOX rope (no q/k permute, via ggufQKPermuted). RoPE base defaults to
// llama.cpp's qwen3 default (1e6) when absent.
func ggufQwen3Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("qwen3." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:       "qwen3",
		HiddenDim:       u("embedding_length"),
		NumLayers:       u("block_count"),
		NumHeads:        u("attention.head_count"),
		NumKVHeads:      u("attention.head_count_kv"),
		HeadDim:         u("attention.key_length"),
		IntermediateDim: u("feed_forward_length"),
		HiddenAct:       "silu",
		VocabSize:       ggufVocabSize(g),
	}
	if eps, ok := g.Float("qwen3.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("qwen3.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // llama.cpp qwen3 default
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufGemmaConfig builds a Gemma 3 (text) Config from the gemma3.* metadata. The
// gemma3 descriptor already supplies the architecture knobs (sandwich norms,
// GeGLU, QK-norm, embed scale √hidden, dual RoPE bases, tied head); this maps the
// dims + the few values the GGUF doesn't carry explicitly. NEOX rope (no q/k
// permute). The GGUF omits the local RoPE base, the sliding/global pattern, and
// query_pre_attn_scalar, so they default to gemma3's fixed values (10000, a 5:1
// pattern, and head_dim). Gemma 3 has no logit/attn soft-capping (that's Gemma 2),
// so those stay zero — ValidateAssumptions rejects a Gemma 2 GGUF here.
func ggufGemmaConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("gemma3." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:        "gemma3",
		HiddenDim:        u("embedding_length"),
		NumLayers:        u("block_count"),
		NumHeads:         u("attention.head_count"),
		NumKVHeads:       u("attention.head_count_kv"),
		HeadDim:          u("attention.key_length"),
		IntermediateDim:  u("feed_forward_length"),
		SlidingWindow:    u("attention.sliding_window"),
		HiddenActivation: "gelu_pytorch_tanh",
		VocabSize:        ggufVocabSize(g),
	}
	if eps, ok := g.Float("gemma3.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	if base, ok := g.Float("gemma3.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	} else {
		cfg.RoPEGlobalBase = 1000000 // gemma3 rope_theta (global layers)
	}
	cfg.RoPELocalBase = 10000 // gemma3 rope_local_base_freq (sliding layers); not in GGUF metadata
	// Sliding/global pattern: gemma3 is 5 local : 1 global. The GGUF carries no
	// layer_types, so IsGlobalLayer falls back to this period ((i+1)%6==0 global).
	if p := u("attention.sliding_window_pattern"); p > 0 {
		cfg.SlidingWindowPattern = p
	} else {
		cfg.SlidingWindowPattern = 6
	}
	// query_pre_attn_scalar == head_dim for gemma3 (the attention scale is its
	// inverse sqrt); the GGUF doesn't store it.
	cfg.QueryPreAttnScalar = float64(cfg.HeadDim)
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufIntArray / ggufBoolArray coerce a GGUF array metadatum (parsed as []any)
// into a typed slice — for gemma4's per-layer feed_forward_length and the
// sliding/global bool pattern.
func ggufIntArray(v any) []int {
	a, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]int, len(a))
	for i, e := range a {
		switch n := e.(type) {
		case int:
			out[i] = n
		case int32:
			out[i] = int(n)
		case int64:
			out[i] = int(n)
		case uint32:
			out[i] = int(n)
		case uint64:
			out[i] = int(n)
		case float64:
			out[i] = int(n)
		}
	}
	return out
}

func ggufBoolArray(v any) []bool {
	a, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]bool, len(a))
	for i, e := range a {
		if b, ok := e.(bool); ok {
			out[i] = b
		}
	}
	return out
}

// ggufGemma4Config builds a Gemma 4 (E-model) Config from the gemma4.* metadata:
// the variable per-layer FFN width + head_dim, the cross-layer KV-sharing count,
// the PLE (per-layer-embedding) dims, the explicit sliding/global bool pattern,
// dual-base RoPE, and the re-added final-logit softcap (30). The proportional
// (partial-rotary) factor for the global layers isn't in the GGUF metadata, so
// it's set from the known Gemma 4 value (0.25). Defaults mirror ggufGemmaConfig.
func ggufGemma4Config(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("gemma4." + k)
		return int(v)
	}
	cfg := &Config{
		ModelType:               "gemma4",
		HiddenDim:               u("embedding_length"),
		NumLayers:               u("block_count"),
		NumHeads:                u("attention.head_count"),
		NumKVHeads:              u("attention.head_count_kv"),
		HeadDim:                 u("attention.key_length_swa"), // sliding/local head_dim (256)
		GlobalHeadDim:           u("attention.key_length"),     // global/full head_dim (512)
		SlidingWindow:           u("attention.sliding_window"),
		SharedKVLayers:          u("attention.shared_kv_layers"),
		HiddenSizePerLayerInput: u("embedding_length_per_layer_input"),
		HiddenActivation:        "gelu_pytorch_tanh",
		VocabSize:               ggufVocabSize(g),
		PartialRotaryFactor:     0.25, // proportional RoPE on the global layers; not in GGUF metadata
	}
	cfg.VocabSizePerLayerInput = cfg.VocabSize // PLE table shares the main vocab
	// Variable per-layer FFN width (e.g. 6144 then 12288); fall back to a scalar.
	if a, ok := g.Metadata["gemma4.feed_forward_length"]; ok {
		cfg.FFNPerLayer = ggufIntArray(a)
	}
	if len(cfg.FFNPerLayer) > 0 {
		cfg.IntermediateDim = cfg.FFNPerLayer[0]
	} else {
		cfg.IntermediateDim = u("feed_forward_length")
	}
	// Explicit per-layer attention type (true = sliding, false = full/global).
	if a, ok := g.Metadata["gemma4.attention.sliding_window_pattern"]; ok {
		for _, sliding := range ggufBoolArray(a) {
			if sliding {
				cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
			} else {
				cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
			}
		}
	}
	// head_count_kv may be a per-layer array (12B: 8 on sliding, 1 on global) or a
	// scalar (E2B/E4B: 1). For the array, pull the sliding + global values; the
	// scalar literal above already covered the uniform case.
	if arr := ggufIntArray(g.Metadata["gemma4.attention.head_count_kv"]); len(arr) > 0 {
		for i, lt := range cfg.LayerTypes {
			if i >= len(arr) {
				break
			}
			if lt == "full_attention" {
				cfg.NumGlobalKVHeads = arr[i]
			} else {
				cfg.NumKVHeads = arr[i]
			}
		}
	}
	if eps, ok := g.Float("gemma4.attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = eps
	}
	cfg.RoPEGlobalBase = 1000000
	if base, ok := g.Float("gemma4.rope.freq_base"); ok {
		cfg.RoPEGlobalBase = base
	}
	cfg.RoPELocalBase = 10000
	if base, ok := g.Float("gemma4.rope.freq_base_swa"); ok {
		cfg.RoPELocalBase = base
	}
	if sc, ok := g.Float("gemma4.final_logit_softcapping"); ok {
		cfg.FinalLogitSoftcap = sc
	}
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufMellumConfig builds a Mellum2 Config from the mellum.* metadata: dims,
// the MoE counts (incl. the narrower expert_feed_forward_length), the sliding/
// full layer pattern, and the YaRN rope scaling — synthesized into the same
// shapes the mellum adapter consumes (LayerTypes + a rope_parameters JSON), so
// resolveArchitecture runs the identical descriptor build as the safetensors path.
func ggufMellumConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("mellum." + k)
		return int(v)
	}
	gf := func(k string) float64 { // float-or-int metadata
		if v, ok := g.Float("mellum." + k); ok {
			return v
		}
		v, _ := g.Uint("mellum." + k)
		return float64(v)
	}
	normTopK := true
	cfg := &Config{
		ModelType:           "mellum",
		HiddenDim:           u("embedding_length"),
		NumLayers:           u("block_count"),
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             u("attention.key_length"),
		IntermediateDim:     u("feed_forward_length"),        // dense (vestigial)
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NumExperts:          u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		SlidingWindow:       u("attention.sliding_window"),
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
	}
	// Per-layer attention type: sliding_window_pattern[i] true ⇒ sliding (local),
	// false ⇒ full (global).
	pat, ok := g.Metadata["mellum.attention.sliding_window_pattern"].([]any)
	if !ok || len(pat) != cfg.NumLayers {
		return nil, fmt.Errorf("decoder(gguf-mellum): sliding_window_pattern missing or wrong length")
	}
	for _, p := range pat {
		if b, _ := p.(bool); b {
			cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
		} else {
			cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
		}
	}
	// Synthesize rope_parameters: YaRN on the full layers (freq_base + scaling),
	// plain RoPE on the sliding layers (freq_base_swa). llama.cpp truncates the
	// attention_factor to f32; the mscale tolerance absorbs that.
	base := gf("rope.freq_base")
	baseSwa := gf("rope.freq_base_swa")
	if baseSwa == 0 {
		baseSwa = base
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"full_attention":{"rope_type":"yarn","rope_theta":%g,"factor":%g,"original_max_position_embeddings":%g,"beta_fast":%g,"beta_slow":%g,"attention_factor":%g},`+
			`"sliding_attention":{"rope_type":"default","rope_theta":%g}}`,
		base, gf("rope.scaling.factor"), gf("rope.scaling.original_context_length"),
		gf("rope.scaling.yarn_beta_fast"), gf("rope.scaling.yarn_beta_slow"),
		gf("rope.scaling.yarn_attn_factor"), baseSwa))
	ggufEOS(g, cfg)
	return cfg, nil
}

// ggufGlm4MoeConfig builds a GLM-4.5/4.6 Config from the glm4moe.* metadata: dims,
// the DeepSeek-style MoE counts (expert_count / expert_used_count /
// expert_shared_count / expert_feed_forward_length), the leading_dense_block_count
// dense prefix, routed_scaling_factor (expert_weights_scale), and a synthesized
// rope_parameters JSON (partial_rotary_factor = rope.dimension_count / head_dim) so
// glm4moeArchitecture runs the identical descriptor build as the safetensors path.
func ggufGlm4MoeConfig(g *embed.GGUFFile) (*Config, error) {
	u := func(k string) int {
		v, _ := g.Uint("glm4moe." + k)
		return int(v)
	}
	gf := func(k string) float64 {
		if v, ok := g.Float("glm4moe." + k); ok {
			return v
		}
		v, _ := g.Uint("glm4moe." + k)
		return float64(v)
	}
	normTopK := true
	if b, ok := g.Metadata["glm4moe.expert_weights_norm"].(bool); ok {
		normTopK = b
	}
	headDim := u("attention.key_length")
	scale := gf("expert_weights_scale")
	if scale == 0 {
		scale = 1
	}
	// block_count includes the trailing NextN/MTP block(s) goinfer drops.
	numLayers := u("block_count") - u("nextn_predict_layers")
	// GLM-4.5/4.6 carry q/k/v bias but NO QK-norm; the tiny synthetic is the
	// reverse. Neither has a metadata flag, so detect both from blk.0's tensors.
	_, attnBias := g.Dims("blk.0.attn_q.bias")
	_, qkNorm := g.Dims("blk.0.attn_q_norm.weight")
	cfg := &Config{
		ModelType:           "glm4_moe",
		HiddenDim:           u("embedding_length"),
		NumLayers:           numLayers,
		AttentionBias:       attnBias,
		UseQKNorm:           qkNorm,
		NumHeads:            u("attention.head_count"),
		NumKVHeads:          u("attention.head_count_kv"),
		HeadDim:             headDim,
		IntermediateDim:     u("feed_forward_length"),        // dense prefix width
		MoeIntermediateSize: u("expert_feed_forward_length"), // expert FFN width
		NRoutedExperts:      u("expert_count"),
		NumExpertsPerTok:    u("expert_used_count"),
		NSharedExperts:      u("expert_shared_count"),
		FirstKDenseReplace:  u("leading_dense_block_count"),
		RoutedScalingFactor: scale,
		NormTopKProb:        &normTopK,
		HiddenAct:           "silu",
		VocabSize:           ggufVocabSize(g),
		RMSNormEps:          gf("attention.layer_norm_rms_epsilon"),
	}
	rot := u("rope.dimension_count")
	partial := 0.0
	if headDim > 0 {
		partial = float64(rot) / float64(headDim)
	}
	cfg.RopeParameters = json.RawMessage(fmt.Sprintf(
		`{"rope_type":"default","rope_theta":%g,"partial_rotary_factor":%g}`,
		gf("rope.freq_base"), partial))
	ggufEOS(g, cfg)
	return cfg, nil
}

// loadGGUFWeights parses a .gguf file and builds the weight bundle, mapping
// llama.cpp tensor names to the descriptor and un-permuting q/k.
func loadGGUFWeights(path string, quant quantMode, embedInt4 bool) (*Weights, error) {
	// mmap, not heap-read: the raw quantized bytes stay in reclaimable page
	// cache while we dequantize tensor-by-tensor. The weights end up as fresh
	// (f32 or int8) copies, so the mapping is unneeded once the build returns.
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return nil, err
	}
	defer g.Close()
	return buildGGUFWeights(g, quant, embedInt4)
}

// buildGGUFWeights builds the weight bundle from an already-open GGUF — whether
// memory-mapped (loadGGUFWeights) or backed by an in-memory slice
// (LoadGGUFBytes). It resolves the config + architecture from the GGUF's own
// metadata, so both entry points produce an identical model.
func buildGGUFWeights(g *embed.GGUFFile, quant quantMode, embedInt4 bool) (*Weights, error) {
	cfg, err := ggufConfig(g)
	if err != nil {
		return nil, err
	}
	if err := validateGGUFDims(cfg); err != nil {
		return nil, err
	}
	arch, _, err := resolveArchitecture(cfg) // llama descriptor + finalizeRoPE
	if err != nil {
		return nil, err
	}
	return buildWeightsFromGGUF(cfg, arch, g, quant, embedInt4, nil, "")
}

// StreamTranscodeGGUF transcodes the GGUF at path into a .giw weights body written
// to out (the GINFW serialization + trailing CRC), loading ONE layer at a time so
// peak RAM is ~one layer rather than the whole model — the streaming analogue of
// Load + SerializeWeightsTo, for models too large to hold resident (the build-time
// hump that otherwise blocks running a >RAM MoE). Generic-loader families stream;
// the qwen35/gemma4 dedicated loaders fall back to a resident build + serialize
// (those models fit). Returns the bytes written. Typically invoked inside
// giw.WriteStream as the weights half of a .giw bundle.
func StreamTranscodeGGUF(path string, out io.Writer, quant string, embedInt4 bool, id string) (int64, error) {
	q, err := parseQuant(quant)
	if err != nil {
		return 0, err
	}
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return 0, err
	}
	defer g.Close()
	cfg, err := ggufConfig(g)
	if err != nil {
		return 0, err
	}
	if err := validateGGUFDims(cfg); err != nil {
		return 0, err
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		return 0, err
	}
	// Dedicated-loader families (qwen35/gemma4) can't stream; they fit resident.
	if arch.qwen35 != nil || arch.gemma4 != nil {
		w, berr := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, nil, "")
		if berr != nil {
			return 0, berr
		}
		return SerializeWeightsTo(out, w, id)
	}
	wr := &giwWriter{sink: out}
	if _, err := buildWeightsFromGGUF(cfg, arch, g, q, embedInt4, wr, id); err != nil {
		return wr.n, err
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], wr.crc) // running CRC over the streamed body
	if _, err := out.Write(crc[:]); err != nil {
		return wr.n, err
	}
	return wr.n + 4, nil
}

// Generous sanity ceilings for the metadata-derived core dims — orders of
// magnitude above any real checkpoint (largest open models: ~120 layers, hidden
// ~16K, vocab ~256K), but low enough that an untrusted GGUF can't drive a
// pathological allocation. They exist to turn a hostile header into a typed
// error instead of a panic/OOM, not to police legitimate models.
const (
	maxGGUFLayers    = 4096
	maxGGUFHidden    = 1 << 20
	maxGGUFHeads     = 1 << 16
	maxGGUFVocabSize = 1 << 24
)

// validateGGUFDims rejects core dimensions that are out of range before they
// reach an allocation. GGUF reads counts as uint64 and narrows to int, so a
// hostile metadata value (e.g. block_count = 1<<63) silently becomes negative
// and would panic make([]LayerWeights, n) ("makeslice: len out of range"); a
// merely-huge positive value would OOM. Both are caught here. Only the dims
// every family sets from metadata are checked (HeadDim is derived per family;
// IntermediateDim is vestigial/zero for some MoE checkpoints).
func validateGGUFDims(cfg *Config) error {
	switch {
	case cfg.NumLayers <= 0 || cfg.NumLayers > maxGGUFLayers:
		return fmt.Errorf("decoder(gguf): block_count %d out of range (1..%d)", cfg.NumLayers, maxGGUFLayers)
	case cfg.HiddenDim <= 0 || cfg.HiddenDim > maxGGUFHidden:
		return fmt.Errorf("decoder(gguf): embedding_length %d out of range (1..%d)", cfg.HiddenDim, maxGGUFHidden)
	case cfg.NumHeads <= 0 || cfg.NumHeads > maxGGUFHeads:
		return fmt.Errorf("decoder(gguf): head_count %d out of range (1..%d)", cfg.NumHeads, maxGGUFHeads)
	case cfg.NumKVHeads <= 0 || cfg.NumKVHeads > maxGGUFHeads:
		return fmt.Errorf("decoder(gguf): head_count_kv %d out of range (1..%d)", cfg.NumKVHeads, maxGGUFHeads)
	case cfg.VocabSize <= 0 || cfg.VocabSize > maxGGUFVocabSize:
		return fmt.Errorf("decoder(gguf): vocab_size %d out of range (1..%d)", cfg.VocabSize, maxGGUFVocabSize)
	}
	return nil
}

// LoadGGUFBytes loads a GGUF model from an in-memory slice (e.g. an inflated
// //go:embed asset) — same result as Load on the equivalent .gguf file, but
// nothing touches the filesystem (no temp file, no writable disk needed). raw
// is parsed in place and only needs to live until this returns; the built
// weights are fresh copies.
//
// EOS ids come from the GGUF's own tokenizer metadata (there is no directory to
// read a generation_config.json from), matching the file path's resolution for
// a bare .gguf.
func LoadGGUFBytes(raw []byte, opts Options) (*Model, error) {
	be, beErr := NewBackend(opts.Backend)
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	g, err := embed.OpenGGUFBytes(raw)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	w, err := buildGGUFWeights(g, quant, opts.EmbedInt4)
	g.Close()
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if beErr != nil {
		fmt.Println(beErr) // webgpu requested but fell back — not fatal.
	}
	return (&Model{w: w, be: be, quant: opts.Quant, eosIDs: w.Cfg.EOSIDs(), kvF16: opts.KVPrecision == "f16", kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8"}).withResidency(), nil
}

// buildWeightsFromGGUF dequantizes the GGUF tensors into the weight bundle.
// When quant is set, each matmul tensor is re-quantized (per-row int8 or
// group-wise int4) right after it is dequantized (and un-permuted) and its f32
// is freed — so a Q4/Q8 GGUF lands resident as int8/int4 (~¼ / ~⅛ f32) without
// ever materializing the whole model in f32 (see loadWeights). The GGUF's own
// quant is lossy and so is the re-quant, but it captures nearly all of what a
// Q4_K_M file carries.
// When sink is non-nil, the weights are STREAMED to it (a .giw body) instead of
// retained: the header + globals are written, then each layer is loaded, written,
// and freed in turn, so peak RAM is ~one layer rather than the whole model — this is
// what lets a model larger than RAM be transcoded. The returned *Weights then holds
// no layer tensors (they were freed); the caller writes the trailing CRC. Streaming
// is supported for the generic per-layer loader (llama/qwen2/qwen3/mellum/glm4_moe);
// the qwen35/gemma4 dedicated paths reject it (those models fit resident).
func buildWeightsFromGGUF(cfg *Config, arch *Architecture, g *embed.GGUFFile, quant quantMode, embedInt4 bool, sink *giwWriter, id string) (*Weights, error) {
	hidden, hd := arch.HiddenDim, arch.HeadDim
	w := &Weights{Cfg: *cfg, arch: arch, Layers: make([]LayerWeights, arch.NumLayers)}

	// streamMat builds a [out, in] linalg.WeightMat by streaming the tensor's rows through
	// the GGUF dequantizer and quantizing each row directly into the resident
	// arrays — no whole-tensor f32 intermediate (see streamQuantized). rowSrc maps
	// a destination row index to its source element offset in the tensor (identity
	// for a plain load; a permutation for the RoPE-permuted q/k projections).
	streamMat := func(name string, out, in int, mode quantMode, rowSrc func(r int) int) (linalg.WeightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return linalg.WeightMat{}, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return streamQuantized(out, in, mode, func(r int, dst []float32) error {
			return into(rowSrc(r), dst)
		})
	}
	// mat loads a tensor as a [out, in] linalg.WeightMat, shape-checked, quantized when
	// requested.
	mat := func(name string, out, in int) (linalg.WeightMat, error) {
		return streamMat(name, out, in, matmulQuant(quant, name), func(r int) int { return r * in })
	}
	// embMat loads the embedding / LM head, which is logit-critical — quantize
	// it with the embedding policy (int8 even in int4 mode), not the projection
	// mode.
	embMat := func(name string, out, in int) (linalg.WeightMat, error) {
		return streamMat(name, out, in, quant.embeddingWith(embedInt4), func(r int) int { return r * in })
	}
	// vec loads a 1-D tensor (norm or bias).
	vec := func(name string, n int) ([]float32, error) {
		dims, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 1 || dims[0] != n {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [%d]", name, dims, n)
		}
		return data, nil
	}
	// vnorm loads a norm weight. For the Gemma family (RMSAddOne), llama.cpp's GGUF
	// conversion bakes the (1+w) offset into every stored *norm.weight, so this
	// package's RMSAddOne forward would apply it twice — subtract the 1 back out at
	// load to restore the HF convention the shared descriptor expects. No-op for the
	// other architectures (plain RMSNorm), so all norm loads route through here.
	subOneNorm := arch.RMSAddOne
	vnorm := func(name string, n int) ([]float32, error) {
		v, err := vec(name, n)
		if err != nil {
			return nil, err
		}
		if subOneNorm {
			for i := range v {
				v[i] -= 1
			}
		}
		return v, nil
	}
	// permMat loads a q/k projection and inverts llama.cpp's RoPE row permutation.
	// Because the permutation is a pure reorder of whole rows (output features), it
	// commutes with per-row quantization: rather than permute an f32 buffer then
	// quantize, we dequant the rows straight into HF order — destination row hfRow
	// pulls from GGUF source row (h*hd + 2*j + s) — and quantize in place. See
	// ggufInvPermute for the index derivation.
	permMat := func(name string, out, in, nHead int) (linalg.WeightMat, error) {
		hd := out / nHead
		half := hd / 2
		return streamMat(name, out, in, matmulQuant(quant, name), func(hfRow int) int {
			h, rem := hfRow/hd, hfRow%hd
			ggufRow := h*hd + 2*(rem%half) + rem/half
			return ggufRow * in
		})
	}
	// stackedExperts loads a GGUF MoE expert tensor — a 3-D [in, out, nExpert]
	// (fastest-first) blob where each expert occupies a contiguous [out, in]
	// row-major slice — and returns one quantized linalg.WeightMat per expert.
	stackedExperts := func(name string, out, in, nExpert int) ([]linalg.WeightMat, error) {
		dims, into, err := g.RowDequantizer(name)
		if err != nil {
			return nil, err
		}
		if len(dims) != 3 || dims[0] != in || dims[1] != out || dims[2] != nExpert {
			return nil, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d, experts=%d]", name, dims, in, out, nExpert)
		}
		// Each expert occupies a contiguous [out, in] row-major slice; stream its
		// rows directly into a per-expert quantized linalg.WeightMat (no whole-tensor f32).
		res := make([]linalg.WeightMat, nExpert)
		for e := range nExpert {
			m, err := streamQuantized(out, in, matmulQuant(quant, name), func(r int, dst []float32) error {
				return into((e*out+r)*in, dst)
			})
			if err != nil {
				return nil, err
			}
			res[e] = m
		}
		return res, nil
	}

	var err error
	if w.Embed, err = embMat("token_embd.weight", cfg.VocabSize, hidden); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = vnorm("output_norm.weight", hidden); err != nil {
		return nil, err
	}
	// Separate output head when present; else tied to the embedding.
	arch.TiedLMHead = true
	if g.Has("output.weight") {
		if w.LMHead, err = embMat("output.weight", cfg.VocabSize, hidden); err != nil {
			return nil, err
		}
		arch.TiedLMHead = false
	}

	// Streaming transcode: emit the header + globals now; each layer is written and
	// freed as it loads (below), so the whole model never sits resident.
	if sink != nil {
		if arch.qwen35 != nil || arch.gemma4 != nil {
			return nil, fmt.Errorf("decoder(gguf): streaming transcode unsupported for %s (load resident + prequant instead)", arch.Name)
		}
		if err := sink.writeHeadGlobals(w, id); err != nil {
			return nil, err
		}
	}

	// qwen3_5_moe (qwen35moe): the Gated DeltaNet / gated-softmax hybrid. The GGUF
	// converter bakes layout transforms the HF safetensors path never sees — V
	// heads tiled (num_v>num_k), A_log stored as −exp(A_log), the standard norms
	// (1+w)'d (ssm_norm exempt) — all reversed here so the shared forward runs
	// unchanged. Attention stays f32 (parity-first, like the safetensors path);
	// experts/embeddings quantize. Only blk.{0..NumLayers-1} load — the trailing
	// NextN/MTP block is excluded by ggufQwen35Config's NumLayers.
	if q := arch.qwen35; q != nil {
		numK, numV := q.NumKeyHeads, q.NumValueHeads
		vpk := numV / numK
		hkd, hvd := q.KeyHeadDim, q.ValueHeadDim
		keyDim, valueDim := hkd*numK, hvd*numV
		convDim := 2*keyDim + valueDim
		// f32mat dequantizes a [out, in] tensor to a row-major f32 slice — the
		// DeltaNet/softmax projections stay f32 (parity-first), matching the
		// safetensors qwen35 path (loadQwen35Attn), so only experts/embeddings quantize.
		f32mat := func(name string, out, in int) ([]float32, error) {
			dims, data, derr := g.Tensor(name)
			if derr != nil {
				return nil, derr
			}
			if len(dims) != 2 || dims[0] != in || dims[1] != out {
				return nil, fmt.Errorf("decoder(gguf-qwen35): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
			}
			return data, nil
		}
		loadQ35 := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			var e error
			// attn_norm = input_layernorm (pre-attn); post_attention_norm =
			// post_attention_layernorm (pre-MLP, Pre2). Both (1+w)'d → vnorm subtracts.
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
				return e
			}
			if arch.isLinearLayer(i) {
				d := &deltaNetWeights{}
				// in_proj_qkv (attn_qkv): only the V output rows are tiled; q/k untouched.
				qkv, err := f32mat(p+"attn_qkv.weight", convDim, hidden)
				if err != nil {
					return err
				}
				vOff := 2 * keyDim * hidden
				copy(qkv[vOff:], untileVHeads(qkv[vOff:], 1, numK, vpk, hvd, hidden))
				d.inProjQKV = qkv
				// in_proj_z (attn_gate): all rows tiled.
				z, err := f32mat(p+"attn_gate.weight", valueDim, hidden)
				if err != nil {
					return err
				}
				d.inProjZ = untileVHeads(z, 1, numK, vpk, hvd, hidden)
				// in_proj_a / in_proj_b (ssm_alpha/ssm_beta): one row per value head (hd=1).
				a, err := f32mat(p+"ssm_alpha.weight", numV, hidden)
				if err != nil {
					return err
				}
				d.inProjA = untileVHeads(a, 1, numK, vpk, 1, hidden)
				b, err := f32mat(p+"ssm_beta.weight", numV, hidden)
				if err != nil {
					return err
				}
				d.inProjB = untileVHeads(b, 1, numK, vpk, 1, hidden)
				// conv1d (ssm_conv1d): only the V channel block tiled; trail = kernel width.
				cw, err := f32mat(p+"ssm_conv1d.weight", convDim, q.ConvKernel)
				if err != nil {
					return err
				}
				vOffC := 2 * keyDim * q.ConvKernel
				copy(cw[vOffC:], untileVHeads(cw[vOffC:], 1, numK, vpk, hvd, q.ConvKernel))
				d.convW = cw
				// ssm_a is already −exp(A_log); reorder by value head (1-D).
				av, err := vec(p+"ssm_a", numV)
				if err != nil {
					return err
				}
				d.negExpA = untileVHeads(av, 1, numK, vpk, 1, 1)
				// dt_bias (ssm_dt.bias): reorder by value head (1-D).
				dt, err := vec(p+"ssm_dt.bias", numV)
				if err != nil {
					return err
				}
				d.dtBias = untileVHeads(dt, 1, numK, vpk, 1, 1)
				// ssm_norm: per-head_v_dim gated-RMSNorm weight — NOT tiled and NOT
				// (1+w)'d by the converter, so load raw (vec, not vnorm).
				if d.normW, err = vec(p+"ssm_norm.weight", hvd); err != nil {
					return err
				}
				// out_proj (ssm_out): the value/input columns are tiled.
				op, err := f32mat(p+"ssm_out.weight", hidden, valueDim)
				if err != nil {
					return err
				}
				d.outProj = untileVHeads(op, hidden, numK, vpk, hvd, 1)
				l.delta = d
			} else {
				// Gated softmax layer: double-width q_proj (query ‖ gate per head, kept
				// fused in attn_q), per-head QK-norm. NEOX rope ⇒ no q/k permute.
				a := &qwenAttnWeights{}
				shd := arch.HeadDim
				kvDim := arch.NumKVHeads * shd
				if a.qProj, e = f32mat(p+"attn_q.weight", arch.NumHeads*shd*2, hidden); e != nil {
					return e
				}
				if a.kProj, e = f32mat(p+"attn_k.weight", kvDim, hidden); e != nil {
					return e
				}
				if a.vProj, e = f32mat(p+"attn_v.weight", kvDim, hidden); e != nil {
					return e
				}
				if a.oProj, e = f32mat(p+"attn_output.weight", hidden, arch.NumHeads*shd); e != nil {
					return e
				}
				if a.qNorm, e = vnorm(p+"attn_q_norm.weight", shd); e != nil {
					return e
				}
				if a.kNorm, e = vnorm(p+"attn_k_norm.weight", shd); e != nil {
					return e
				}
				l.qattn = a
			}
			// MoE FFN (every layer): router f32 (matching safetensors), 256 stacked
			// experts quantized, shared expert quantized + f32 sigmoid gate.
			if l.Router, e = streamMat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
				return e
			}
			expInter := arch.MoE.IntermediateDim
			gate, gerr := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			up, uerr := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			down, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
			if gerr != nil || uerr != nil || derr != nil {
				return fmt.Errorf("decoder(gguf-qwen35): experts layer %d: %v / %v / %v", i, gerr, uerr, derr)
			}
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for ei := range arch.MoE.NumExperts {
				l.Experts[ei] = expertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
			}
			sInter := arch.MoE.SharedIntermediateDim
			if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
				return e
			}
			// shared_expert_gate (ffn_gate_inp_shexp): 1-D [hidden] → [1, hidden] f32.
			sg, err := vec(p+"ffn_gate_inp_shexp.weight", hidden)
			if err != nil {
				return err
			}
			l.SharedGate = linalg.WrapF32(sg, 1, hidden)
			return nil
		}
		if err := parallelLayers(arch.NumLayers, loadQ35); err != nil {
			return nil, err
		}
		return w, nil
	}

	// Gemma 4 (E-models): a dedicated layer loader. Per-layer head_dim/FFN, the
	// global/sliding split, the cross-layer KV-shared tail (layers ≥ first-shared
	// carry no k/v), and the PLE branch (inp_gate/proj/post_norm + layer scalar).
	// The model-level PLE inputs (per_layer_*) load here. NEOX rope ⇒ no q/k
	// permute, so plain mat() throughout.
	if g4 := arch.gemma4; g4 != nil {
		if g4.HiddenSizePerLayerInput > 0 {
			pleTotal := arch.NumLayers * g4.HiddenSizePerLayerInput
			if w.PerLayerTokenEmbed, err = embMat("per_layer_token_embd.weight", cfg.VocabSize, pleTotal); err != nil {
				return nil, err
			}
			if w.PerLayerModelProj, err = mat("per_layer_model_proj.weight", pleTotal, hidden); err != nil {
				return nil, err
			}
			if w.PerLayerProjNorm, err = vnorm("per_layer_proj_norm.weight", g4.HiddenSizePerLayerInput); err != nil {
				return nil, err
			}
		}
		firstShared := arch.NumLayers - g4.SharedKVLayers
		loadG4 := func(i int) error {
			l := &w.Layers[i]
			p := fmt.Sprintf("blk.%d.", i)
			hdL := arch.headDimAt(i)
			qDimL, kvDimL, ffn := arch.NumHeads*hdL, arch.kvHeadsAt(i)*hdL, arch.ffnAt(i)
			var e error
			if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PostAttnNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
				return e
			}
			if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
				return e
			}
			if l.PostMLPNorm, e = vnorm(p+"post_ffw_norm.weight", hidden); e != nil {
				return e
			}
			if l.QProj, e = mat(p+"attn_q.weight", qDimL, hidden); e != nil {
				return e
			}
			if l.QNorm, e = vnorm(p+"attn_q_norm.weight", hdL); e != nil {
				return e
			}
			l.KVShared = i >= firstShared
			if !l.KVShared {
				if l.KProj, e = mat(p+"attn_k.weight", kvDimL, hidden); e != nil {
					return e
				}
				if l.KNorm, e = vnorm(p+"attn_k_norm.weight", hdL); e != nil {
					return e
				}
				// attention_k_eq_v (12B global layers): no v_proj — V reuses K's
				// projection. Detect by the tensor's absence (robust to whether the
				// converter omits or duplicates it).
				if g.Has(p + "attn_v.weight") {
					if l.VProj, e = mat(p+"attn_v.weight", kvDimL, hidden); e != nil {
						return e
					}
				} else {
					l.VFromK = true
				}
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, qDimL); e != nil {
				return e
			}
			if l.GateProj, e = mat(p+"ffn_gate.weight", ffn, hidden); e != nil {
				return e
			}
			if l.UpProj, e = mat(p+"ffn_up.weight", ffn, hidden); e != nil {
				return e
			}
			if l.DownProj, e = mat(p+"ffn_down.weight", hidden, ffn); e != nil {
				return e
			}
			if g4.HiddenSizePerLayerInput > 0 {
				if l.PLEGate, e = mat(p+"inp_gate.weight", g4.HiddenSizePerLayerInput, hidden); e != nil {
					return e
				}
				if l.PLEProj, e = mat(p+"proj.weight", hidden, g4.HiddenSizePerLayerInput); e != nil {
					return e
				}
				if l.PostPLENorm, e = vnorm(p+"post_norm.weight", hidden); e != nil {
					return e
				}
			}
			l.LayerScalar = 1
			if sc, se := vec(p+"layer_output_scale.weight", 1); se == nil {
				l.LayerScalar = sc[0]
			}
			return nil
		}
		if err = parallelLayers(arch.NumLayers, loadG4); err != nil {
			return nil, err
		}
		// K=V (attention_k_eq_v) on the 12B global layers: V reuses K's projection
		// (v_norm(k_proj) — see loadG4/runLayersGemma4). Parity-gated against the HF
		// bf16 oracle by TestGemma4_12B_logitParity (argmax exact, cosine 0.990).
		return w, nil
	}

	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	// llama.cpp permutes the q/k weights (and their per-row biases / head-dim
	// norms) only for the NORM rope type — llama and its derivatives (incl.
	// mellum). The NEOX rope type (qwen2/qwen3/gemma) leaves q/k in HF order, so
	// no un-permutation is needed there.
	permuteQK := ggufQKPermuted(arch.Name)
	loadQK := func(name string, out, nHead int) (linalg.WeightMat, error) {
		if permuteQK {
			return permMat(name, out, hidden, nHead)
		}
		return mat(name, out, hidden)
	}
	// Load the layers in parallel: each is independent (its own linalg.WeightMat slots
	// over the read-only mmap), and the per-tensor dequant + re-quant is the load's
	// cost — fanning it out across cores turns a 12B GGUF's ~2 min load into seconds.
	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var err error
		if l.PreAttnNorm, err = vnorm(p+"attn_norm.weight", hidden); err != nil {
			return err
		}
		if l.QProj, err = loadQK(p+"attn_q.weight", qDim, arch.NumHeads); err != nil {
			return err
		}
		if l.KProj, err = loadQK(p+"attn_k.weight", kvDim, arch.NumKVHeads); err != nil {
			return err
		}
		if l.VProj, err = mat(p+"attn_v.weight", kvDim, hidden); err != nil {
			return err
		}
		if l.OProj, err = mat(p+"attn_output.weight", hidden, qDim); err != nil {
			return err
		}
		// q/k/v projection bias (Qwen2): the q and k biases are per-output-row, so
		// llama.cpp's RoPE row permutation applies to them exactly as to the q/k
		// weight rows (ggufInvPermute with in=1); the v bias is not permuted.
		if arch.QKVBias {
			qb, qe := vec(p+"attn_q.bias", qDim)
			kb, ke := vec(p+"attn_k.bias", kvDim)
			vb, ve := vec(p+"attn_v.bias", kvDim)
			if qe != nil || ke != nil || ve != nil {
				return fmt.Errorf("decoder(gguf): qkv bias layer %d: %v / %v / %v", i, qe, ke, ve)
			}
			if permuteQK {
				qb = ggufInvPermute(qb, qDim, 1, arch.NumHeads)
				kb = ggufInvPermute(kb, kvDim, 1, arch.NumKVHeads)
			}
			l.QBias, l.KBias, l.VBias = qb, kb, vb
		}
		// QK-norm (Mellum, Qwen3): per-head RMSNorm over head_dim, before RoPE.
		// llama.cpp permutes the q/k weights for its RoPE, so the matching
		// per-head-dim norm weights are un-permuted the same way.
		if arch.QKNorm {
			qn, kerr := vnorm(p+"attn_q_norm.weight", hd)
			if kerr != nil {
				return kerr
			}
			kn, kerr := vnorm(p+"attn_k_norm.weight", hd)
			if kerr != nil {
				return kerr
			}
			if permuteQK {
				qn, kn = ggufInvPermuteVec(qn), ggufInvPermuteVec(kn)
			}
			l.QNorm, l.KNorm = qn, kn
		}
		// Pre-MLP norm. GLM4_MOE has no ffn_norm — llama.cpp stores its (positionally
		// pre-MLP, Pre2) post_attention_layernorm under the ATTN_POST_NORM name; every
		// other family here uses ffn_norm.
		preMLPNorm := "ffn_norm.weight"
		if arch.Name == "glm4_moe" {
			preMLPNorm = "post_attention_norm.weight"
		}
		if l.PreMLPNorm, err = vnorm(p+preMLPNorm, hidden); err != nil {
			return err
		}
		// Sandwich norms (Gemma 3): a post-attention and a post-FFN RMSNorm in
		// addition to the pre-norms. GGUF names them post_attention_norm /
		// post_ffw_norm.
		if arch.NormPlacement == NormSandwich4 {
			if l.PostAttnNorm, err = vnorm(p+"post_attention_norm.weight", hidden); err != nil {
				return err
			}
			if l.PostMLPNorm, err = vnorm(p+"post_ffw_norm.weight", hidden); err != nil {
				return err
			}
		}
		if arch.MoE != nil && i >= arch.FirstKDense {
			// Sparse MoE: router + stacked per-expert SwiGLU at the narrower
			// moe_intermediate_size. Mellum stops there; GLM/DeepSeek add an
			// e_score_correction_bias (exp_probs_b) and an always-on shared expert
			// (ffn_*_shexp, ungated). GLM's first_k_dense_replace prefix (i < FirstKDense)
			// has no router and falls through to the dense FFN below.
			expInter := arch.MoE.IntermediateDim
			if l.Router, err = mat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden); err != nil {
				return err
			}
			if arch.MoE.RouterSigmoid { // DeepSeek/GLM e_score_correction_bias (a bias term)
				if l.RouterBias, err = vec(p+"exp_probs_b.bias", arch.MoE.NumExperts); err != nil {
					return err
				}
			}
			gate, gerr := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			up, uerr := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
			down, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
			if gerr != nil || uerr != nil || derr != nil {
				return fmt.Errorf("decoder(gguf): experts layer %d: %v / %v / %v", i, gerr, uerr, derr)
			}
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for e := range arch.MoE.NumExperts {
				l.Experts[e] = expertWeights{Gate: gate[e], Up: up[e], Down: down[e]}
			}
			// Shared always-on expert (GLM: ffn_*_shexp). Ungated for GLM/DeepSeek;
			// Qwen2-MoE-style families would also carry a sigmoid gate (not in GGUF here).
			if arch.MoE.SharedIntermediateDim > 0 {
				sInter := arch.MoE.SharedIntermediateDim
				if l.SharedExpert.Gate, err = mat(p+"ffn_gate_shexp.weight", sInter, hidden); err != nil {
					return err
				}
				if l.SharedExpert.Up, err = mat(p+"ffn_up_shexp.weight", sInter, hidden); err != nil {
					return err
				}
				if l.SharedExpert.Down, err = mat(p+"ffn_down_shexp.weight", hidden, sInter); err != nil {
					return err
				}
				if !arch.MoE.SharedUngated {
					sg, serr := vec(p+"ffn_gate_inp_shexp.weight", hidden)
					if serr != nil {
						return serr
					}
					l.SharedGate = linalg.WrapF32(sg, 1, hidden)
				}
			}
			return nil
		}
		if l.GateProj, err = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.UpProj, err = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); err != nil {
			return err
		}
		if l.DownProj, err = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); err != nil {
			return err
		}
		return nil
	}
	if sink != nil {
		// Sequential load → write → free: peak RAM is one layer, not the model.
		for i := 0; i < arch.NumLayers; i++ {
			if err := loadLayer(i); err != nil {
				return nil, err
			}
			sink.layer(&w.Layers[i])
			if sink.err != nil {
				return nil, sink.err
			}
			w.Layers[i] = LayerWeights{} // release before the next layer
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// ggufQKPermuted reports whether llama.cpp's GGUF conversion permuted this
// architecture's q/k weights (and their per-row biases and head-dim norms). It
// permutes only for the NORM rope type — llama and its derivatives, including
// mellum; the NEOX rope type (qwen2/qwen3/gemma and the other modern families)
// leaves q/k in HF rotate_half order, so no un-permutation is needed. Unknown
// archs default to NEOX (no permute), the common modern case.
func ggufQKPermuted(archName string) bool {
	switch archName {
	case "llama", "mellum":
		return true
	default:
		return false
	}
}

// ggufInvPermuteVec inverts llama.cpp's q/k RoPE permutation for a single
// head_dim-wide vector (a QK-norm weight, shared across heads): HF position
// (s*half + j) comes from GGUF position (2*j + s). Mirrors ggufInvPermute with
// nHead=1 and no input dimension.
func ggufInvPermuteVec(v []float32) []float32 {
	hd := len(v)
	half := hd / 2
	res := make([]float32, hd)
	for s := range 2 {
		for j := range half {
			res[s*half+j] = v[2*j+s]
		}
	}
	return res
}

// ggufInvPermute inverts llama.cpp's q/k weight permutation. llama.cpp stores
// the projection rows in interleaved-pair order for its RoPE; this package uses
// HF's rotate_half (first-half / second-half) order. For head h, the HF row
// (s*half + j) — s∈{0,1} selecting the half, j the position within it — comes
// from the GGUF row (2*j + s). w is [out, in] row-major; out = nHead*headDim.
func ggufInvPermute(w []float32, out, in, nHead int) []float32 {
	hd := out / nHead
	half := hd / 2
	res := make([]float32, len(w))
	for h := range nHead {
		for s := range 2 {
			for j := range half {
				hfRow := h*hd + s*half + j
				ggufRow := h*hd + 2*j + s
				copy(res[hfRow*in:hfRow*in+in], w[ggufRow*in:ggufRow*in+in])
			}
		}
	}
	return res
}

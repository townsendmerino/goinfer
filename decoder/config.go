package decoder

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
)

// Config captures the Gemma 3 architecture constants the forward pass
// depends on. Field tags follow the HF config.json schema so a checkpoint's
// config drives the loader rather than hardcoded constants — the same
// config-driven approach encoder.Config uses, which is what lets one code
// path serve both 270M and 1B (and beyond) unchanged.
//
// Values that vary per layer (the 5:1 local:global attention pattern) are
// derived from SlidingWindowPattern at load time, not stored per layer.
type Config struct {
	ModelType            string  `json:"model_type"`              // "gemma3_text" — selects the arch adapter
	VocabSize            int     `json:"vocab_size"`              // 262144
	HiddenDim            int     `json:"hidden_size"`             // 640 (270M)
	NumLayers            int     `json:"num_hidden_layers"`       // 18 (270M)
	NumHeads             int     `json:"num_attention_heads"`     // 4 (270M)
	NumKVHeads           int     `json:"num_key_value_heads"`     // 1 (270M) — GQA
	HeadDim              int     `json:"head_dim"`                // 256 (270M); note heads*headDim != hidden
	IntermediateDim      int     `json:"intermediate_size"`       // 2048 (270M) — GeGLU
	MaxPositions         int     `json:"max_position_embeddings"` // 32768
	RMSNormEps           float64 `json:"rms_norm_eps"`
	RoPELocalBase        float64 `json:"rope_local_base_freq"`   // 10000
	RoPEGlobalBase       float64 `json:"rope_theta"`             // 1000000
	SlidingWindow        int     `json:"sliding_window"`         // 512 (270M)
	SlidingWindowPattern int     `json:"sliding_window_pattern"` // 6 → 5 local : 1 global
	QueryPreAttnScalar   float64 `json:"query_pre_attn_scalar"`  // 256 (270M)
	UseQKNorm            bool    `json:"use_qk_norm"`            // true in Gemma 3
	HiddenActivation     string  `json:"hidden_activation"`      // Gemma: "gelu_pytorch_tanh"
	HiddenAct            string  `json:"hidden_act"`             // Llama/Qwen: "silu" (different JSON key)
	AttentionBias        bool    `json:"attention_bias"`         // Qwen2/GPT-2 add q/k/v/o bias; Llama-3/Qwen3 don't
	UseSlidingWindow     bool    `json:"use_sliding_window"`     // Qwen2: gate for sliding-window attention (usually false)

	// Mixture-of-experts. NormTopKProb is a *bool so an absent field can default
	// to true (HF's MixtralConfig default). Mixtral names the expert count
	// num_local_experts; Mellum names it num_experts and gives the experts their
	// own (narrower) FFN width in moe_intermediate_size.
	NumLocalExperts     int   `json:"num_local_experts"`
	NumExperts          int   `json:"num_experts"`
	NumExpertsPerTok    int   `json:"num_experts_per_tok"`
	NormTopKProb        *bool `json:"norm_topk_prob"`
	MoeIntermediateSize int   `json:"moe_intermediate_size"`
	// SharedExpertIntermediateSize is the always-on shared expert's FFN width
	// (Qwen-MoE/Qwen2-MoE). 0/absent ⇒ no shared expert.
	SharedExpertIntermediateSize int `json:"shared_expert_intermediate_size"`

	// RopeScaling is HF's rope_scaling object (llama3 / linear / yarn / …).
	// Plain Llama-3.0 and Qwen3 leave it null; Llama-3.1+/3.2 set it. Kept raw
	// and decoded by parseRopeScaling (G4: linear + llama3 + yarn supported).
	RopeScaling json.RawMessage `json:"rope_scaling"`

	// QuantizationConfig is HF's quantization_config object — for safetensors
	// shipped pre-quantized (GPTQ: packed int4 qweight/qzeros/scales/g_idx).
	// Decoded by parseGPTQ; absent for full-precision checkpoints.
	QuantizationConfig json.RawMessage `json:"quantization_config"`

	// RopeParameters is the newer per-attention-type RoPE config (Mellum):
	// {"full_attention": {...}, "sliding_attention": {...}}, each a rope_theta +
	// a rope_scaling-style object. Decoded by parseRopeParameters.
	RopeParameters json.RawMessage `json:"rope_parameters"`

	// PartialRotaryFactor is the fraction of head_dim RoPE rotates (Phi: 0.4);
	// 0/absent means full rotary. Consumed via Config.rotaryDim.
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`

	// GPT-2 (multi-model-plan G5) uses a different config vocabulary: n_embd /
	// n_head / n_layer / n_positions / n_inner / layer_norm_epsilon /
	// activation_function instead of hidden_size etc. The gpt2 adapter reads
	// these directly.
	NEmbd              int     `json:"n_embd"`
	NHead              int     `json:"n_head"`
	NLayer             int     `json:"n_layer"`
	NPositions         int     `json:"n_positions"`
	NInner             int     `json:"n_inner"` // null/0 ⇒ 4*n_embd
	LayerNormEpsilon   float64 `json:"layer_norm_epsilon"`
	ActivationFunction string  `json:"activation_function"` // GPT-2: "gelu_new"

	// LayerTypes is the per-layer attention kind ("sliding_attention" /
	// "full_attention"). Gemma 3's checkpoints carry the local:global pattern
	// here explicitly; sliding_window_pattern is often null in config.json, so
	// this is the authoritative source when present (see IsGlobalLayer).
	LayerTypes []string `json:"layer_types"`

	// EOSTokenID is the checkpoint's end-of-sequence id(s). HF stores it as
	// either a scalar or a list, so it's kept raw and decoded by EOSIDs.
	EOSTokenID json.RawMessage `json:"eos_token_id"`

	// Gemma 2 fields that MUST be absent/zero in a Gemma 3 checkpoint.
	// ValidateAssumptions rejects a checkpoint that still sets them so we
	// fail loudly rather than silently skip soft-capping.
	FinalLogitSoftcap float64 `json:"final_logit_softcapping"`
	AttnLogitSoftcap  float64 `json:"attn_logit_softcapping"`
}

// IsGlobalLayer reports whether layer i is a global (full) attention layer
// vs a local (sliding-window) one. Gemma 3 carries this per-layer in
// LayerTypes ("full_attention" vs "sliding_attention"), which is the
// authoritative source. Only if LayerTypes is absent do we fall back to the
// SlidingWindowPattern arithmetic (a global layer as the last of each group;
// pattern=6 → layers 5, 11, 17 → 3 global of 18 in the 270M).
//
// Getting this right is load-bearing: local and global layers use different
// RoPE bases (10k vs 1e6), so a misclassified layer silently corrupts logits.
func (c *Config) IsGlobalLayer(i int) bool {
	if i >= 0 && i < len(c.LayerTypes) {
		return c.LayerTypes[i] == "full_attention"
	}
	p := c.SlidingWindowPattern
	if p <= 0 {
		return true // no pattern configured → all-global (degenerate)
	}
	return (i+1)%p == 0
}

// headDim returns the per-head dimension, falling back to hidden/heads when
// config.json omits head_dim. Gemma and Qwen3 always set it explicitly (and
// for Gemma heads*head_dim != hidden, so the field is load-bearing there);
// many Llama/Mistral configs omit it, where hidden_size/num_attention_heads
// is the definition.
func (c *Config) headDim() int {
	if c.HeadDim > 0 {
		return c.HeadDim
	}
	if c.NumHeads > 0 {
		return c.HiddenDim / c.NumHeads
	}
	return 0
}

// ValidateAssumptions fails loudly on any config the scaffolded forward pass
// is not built to honor. Mirrors encoder.Config.ValidateAssumptions: pin the
// assumptions at load time rather than produce junk logits at run time.
func (c *Config) ValidateAssumptions() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder: missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder: num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder: vocab_size is zero")
	case c.HiddenActivation != "" && c.HiddenActivation != "gelu_pytorch_tanh":
		return fmt.Errorf("decoder: hidden_activation=%q unsupported (gelu_pytorch_tanh / GeGLU only)", c.HiddenActivation)
	case c.FinalLogitSoftcap != 0 || c.AttnLogitSoftcap != 0:
		return fmt.Errorf("decoder: soft-capping set (final=%v attn=%v) — that's a Gemma 2 checkpoint; this path is Gemma 3 (QK-norm) only",
			c.FinalLogitSoftcap, c.AttnLogitSoftcap)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder: rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPELocalBase <= 0 || c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder: rope base must be >0 (local=%v global=%v)", c.RoPELocalBase, c.RoPEGlobalBase)
	case len(c.LayerTypes) > 0 && len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder: layer_types has %d entries, want num_hidden_layers=%d", len(c.LayerTypes), c.NumLayers)
	}
	for i, lt := range c.LayerTypes {
		if lt != "sliding_attention" && lt != "full_attention" {
			return fmt.Errorf("decoder: layer_types[%d]=%q unsupported (want sliding_attention/full_attention)", i, lt)
		}
	}
	return nil
}

// validateQwen3 pins the assumptions the qwen3 forward path makes (dense,
// SwiGLU, GQA, single-base RoPE). The Qwen3-MoE model_type isn't registered, so
// reaching here already implies a dense checkpoint; this guards the rest.
func (c *Config) validateQwen3() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.HeadDim == 0:
		return fmt.Errorf("decoder(qwen3): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.HeadDim)
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(qwen3): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(qwen3): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(qwen3): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(qwen3): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(qwen3): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(qwen3): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	}
	return nil
}

// validateQwen2 pins the assumptions the qwen2 forward path makes: like llama
// (dense, SwiGLU, GQA, single-base RoPE, no QK-norm, derived head_dim) but with
// q/k/v projection bias. Sliding-window attention (use_sliding_window) is a
// follow-up — reject it rather than silently run full attention.
func (c *Config) validateQwen2() error {
	if err := c.validateLlama(); err != nil {
		return err
	}
	if c.UseSlidingWindow {
		return fmt.Errorf("decoder(qwen2): use_sliding_window=true not yet supported (full-attention checkpoints only)")
	}
	return nil
}

// validateMixtral pins the Mixtral assumptions: the llama dense constraints
// (reused) plus a valid MoE config — top-k experts of num_local_experts, both
// positive and k ≤ E.
func (c *Config) validateMixtral() error {
	if err := c.validateLlama(); err != nil {
		return err
	}
	switch {
	case c.NumLocalExperts <= 0:
		return fmt.Errorf("decoder(mixtral): num_local_experts must be >0, got %d", c.NumLocalExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumLocalExperts:
		return fmt.Errorf("decoder(mixtral): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumLocalExperts)
	}
	return nil
}

// validateQwen2Moe pins the Qwen2-MoE assumptions: the qwen2 dense constraints
// (llama + q/k/v bias) plus a valid sparse MoE on every layer (num_experts /
// num_experts_per_tok / moe_intermediate_size) and an always-on shared expert
// (shared_expert_intermediate_size). decoder_sparse_step must be 1 (every layer
// sparse) — the only configuration shipped by the Qwen-MoE family.
func (c *Config) validateQwen2Moe() error {
	if err := c.validateLlama(); err != nil {
		return err
	}
	switch {
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(qwen2_moe): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(qwen2_moe): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen2_moe): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case c.SharedExpertIntermediateSize <= 0:
		return fmt.Errorf("decoder(qwen2_moe): shared_expert_intermediate_size must be >0, got %d", c.SharedExpertIntermediateSize)
	}
	return nil
}

// validateMellum pins the assumptions the mellum adapter makes: a sparse MoE on
// every layer (num_experts / num_experts_per_tok / moe_intermediate_size), the
// per-attention-type rope_parameters (full_attention YaRN + sliding plain), and
// the explicit layer_types interleave. The dense FFN axes (RMSNorm, SwiGLU,
// derived head_dim) are the same as llama and resolved there.
func (c *Config) validateMellum() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.NumKVHeads == 0:
		return fmt.Errorf("decoder(mellum): missing core dims (hidden/layers/heads)")
	case c.NumExperts <= 0:
		return fmt.Errorf("decoder(mellum): num_experts must be >0, got %d", c.NumExperts)
	case c.NumExpertsPerTok <= 0 || c.NumExpertsPerTok > c.NumExperts:
		return fmt.Errorf("decoder(mellum): num_experts_per_tok %d out of range (1..%d)", c.NumExpertsPerTok, c.NumExperts)
	case c.MoeIntermediateSize <= 0:
		return fmt.Errorf("decoder(mellum): moe_intermediate_size must be >0, got %d", c.MoeIntermediateSize)
	case len(c.RopeParameters) == 0:
		return fmt.Errorf("decoder(mellum): rope_parameters required (full_attention + sliding_attention)")
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("decoder(mellum): layer_types has %d entries, want %d", len(c.LayerTypes), c.NumLayers)
	}
	return nil
}

// validateGPT2 pins the assumptions the gpt2 forward path makes: LayerNorm,
// learned absolute positions (no RoPE), a non-gated GELU MLP, fused q/k/v with
// bias, and tied embeddings. GPT-2's config uses the n_embd/n_head/n_layer keys.
func (c *Config) validateGPT2() error {
	switch {
	case c.NEmbd == 0 || c.NLayer == 0 || c.NHead == 0:
		return fmt.Errorf("decoder(gpt2): missing required dim (n_embd=%d n_layer=%d n_head=%d)", c.NEmbd, c.NLayer, c.NHead)
	case c.NEmbd%c.NHead != 0:
		return fmt.Errorf("decoder(gpt2): n_embd %d not divisible by n_head %d", c.NEmbd, c.NHead)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(gpt2): vocab_size is zero")
	case c.NPositions == 0:
		return fmt.Errorf("decoder(gpt2): n_positions is zero (need the learned position table size)")
	case c.LayerNormEpsilon <= 0:
		return fmt.Errorf("decoder(gpt2): layer_norm_epsilon must be >0, got %v", c.LayerNormEpsilon)
	case c.ActivationFunction != "" && c.ActivationFunction != "gelu_new" && c.ActivationFunction != "gelu":
		return fmt.Errorf("decoder(gpt2): activation_function=%q unsupported (gelu_new/gelu)", c.ActivationFunction)
	}
	return nil
}

// validateLlama pins the assumptions the llama forward path makes (dense,
// SwiGLU, GQA, single-base RoPE, no QK-norm). It differs from validateQwen3 by
// allowing head_dim to be derived (headDim()). RoPE scaling (rope_scaling) is
// handled by the adapter via parseRopeScaling (G4: linear + llama3). Attention
// bias (Qwen2/GPT-2 q/k/v/o bias) is rejected — a later add.
// Plain Llama-2/3 / Mistral checkpoints pass.
func (c *Config) validateLlama() error {
	switch {
	case c.HiddenDim == 0 || c.NumLayers == 0 || c.NumHeads == 0 || c.headDim() == 0:
		return fmt.Errorf("decoder(llama): missing required dim (hidden=%d layers=%d heads=%d headDim=%d)",
			c.HiddenDim, c.NumLayers, c.NumHeads, c.headDim())
	case c.NumKVHeads == 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("decoder(llama): num_heads %d not a multiple of num_kv_heads %d (GQA)", c.NumHeads, c.NumKVHeads)
	case c.VocabSize == 0:
		return fmt.Errorf("decoder(llama): vocab_size is zero")
	case c.IntermediateDim == 0:
		return fmt.Errorf("decoder(llama): intermediate_size is zero")
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return fmt.Errorf("decoder(llama): hidden_act=%q unsupported (silu/SwiGLU only)", c.HiddenAct)
	case c.RMSNormEps <= 0:
		return fmt.Errorf("decoder(llama): rms_norm_eps must be >0, got %v", c.RMSNormEps)
	case c.RoPEGlobalBase <= 0:
		return fmt.Errorf("decoder(llama): rope_theta must be >0, got %v", c.RoPEGlobalBase)
	case c.AttentionBias:
		return fmt.Errorf("decoder(llama): attention_bias=true (q/k/v/o bias) not yet supported")
	}
	return nil
}

// rotaryDim returns the number of head dims RoPE rotates: partial_rotary_factor
// × head_dim when set (Phi), else 0 (the descriptor reads that as full head_dim).
func (c *Config) rotaryDim() int {
	if c.PartialRotaryFactor > 0 && c.PartialRotaryFactor < 1 {
		return int(c.PartialRotaryFactor * float64(c.headDim()))
	}
	return 0
}

// EOSIDs returns the configured end-of-sequence token ids, handling both the
// scalar (eos_token_id: 1) and list (eos_token_id: [1, 106]) JSON shapes HF
// emits. Empty when the field is absent.
func (c *Config) EOSIDs() []int {
	if len(c.EOSTokenID) == 0 {
		return nil
	}
	var one int
	if err := json.Unmarshal(c.EOSTokenID, &one); err == nil {
		return []int{one}
	}
	var many []int
	if err := json.Unmarshal(c.EOSTokenID, &many); err == nil {
		return many
	}
	return nil
}

// resolveEOSIDs returns the ids that end generation: config.json's
// eos_token_id, plus any extra ids from generation_config.json. The latter is
// HF's authoritative generation source and often lists more than config.json —
// Qwen3's config.json carries only <|im_end|> (151645) while its
// generation_config adds <|endoftext|> (151643), and both must stop a chat
// turn. Deduped, config.json's ids first. generation_config is best-effort
// (absent file → ignored).
func resolveEOSIDs(dir string, cfg *Config) []int {
	seen := map[int]bool{}
	var out []int
	add := func(ids []int) {
		for _, id := range ids {
			if !seen[id] {
				seen[id] = true
				out = append(out, id)
			}
		}
	}
	add(cfg.EOSIDs())
	add(eosFromGenerationConfig(os.DirFS(dir), "generation_config.json"))
	return out
}

// eosFromGenerationConfig reads eos_token_id from generation_config.json,
// reusing EOSIDs' scalar-or-list handling. Returns nil when the file is
// absent or has no eos_token_id.
func eosFromGenerationConfig(fsys fs.FS, name string) []int {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil
	}
	var g struct {
		EOSTokenID json.RawMessage `json:"eos_token_id"`
	}
	if json.Unmarshal(b, &g) != nil {
		return nil
	}
	gc := Config{EOSTokenID: g.EOSTokenID}
	return gc.EOSIDs()
}

// loadConfig reads and parses config.json from fsys.
func loadConfig(fsys fs.FS, name string) (*Config, error) {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("decoder: read %s: %w", name, err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("decoder: parse %s: %w", name, err)
	}
	return &c, nil
}

package decoder

// Generated capability matrix (docs/task-capability-matrix.md).
//
// This in-package test resolves every registry model_type against a minimal
// representative Config, reads the FAMILY-CONSTANT traits off the resulting
// Architecture, and renders two artifacts:
//
//   - docs/capability-matrix.md   — a human table grouped by coverage axis
//   - docs/capability-matrix.json — the machine-readable view
//
// The registry is the single source of truth: the columns are DERIVED from the
// resolved descriptor (drift-proof) plus a tiny presentation-only annotation map
// (familyDoc) for facts with no Architecture equivalent — display name, one-line
// description, loaders, and modality (the last two are loader-code facts, NOT
// descriptor fields, so they are hand-annotated and clearly marked as such).
//
// Run `go test ./decoder -run CapabilityMatrix -update` to regenerate; the plain
// run is a freshness gate that fails if the committed files are stale.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"
)

// updateMatrix, when set, makes the test WRITE the generated artifacts instead
// of comparing against them. No -update flag pre-exists in this package.
var updateMatrix = flag.Bool("update", false, "rewrite the generated capability-matrix artifacts instead of asserting they are fresh")

// representativeConfig returns a minimal, canned *Config for a registered
// model_type such that resolveArchitecture succeeds. Field values are harvested
// from the per-family _test.go fixtures (kept tiny). Aliases that share an
// adapter return a tailored config where the descriptor diverges (e.g.
// deepseek_v2 vs deepseek_v3 routing flavor). Returns nil for an unknown key.
func representativeConfig(modelType string) *Config {
	switch modelType {
	case "gemma3", "gemma3_text":
		// Explicit LayerTypes (one sliding + one global) so slidingWindowColumn's
		// per-layer probe sees the interleave; SlidingWindowPattern alone yields 0
		// globals within NumLayers=2 and would misread as all-layer.
		return &Config{
			ModelType: "gemma3_text", VocabSize: 100, HiddenDim: 8, NumLayers: 2,
			NumHeads: 2, NumKVHeads: 1, HeadDim: 4, IntermediateDim: 16, RMSNormEps: 1e-6,
			RoPELocalBase: 10000, RoPEGlobalBase: 1000000, SlidingWindow: 512,
			SlidingWindowPattern: 6, QueryPreAttnScalar: 4, HiddenActivation: "gelu_pytorch_tanh",
			LayerTypes: []string{"sliding_attention", "full_attention"},
		}
	case "gemma4":
		// Explicit LayerTypes (see gemma3 note) so the interleave probe sees a global.
		return &Config{
			ModelType: "gemma4", HiddenDim: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2,
			HeadDim: 16, IntermediateDim: 128, VocabSize: 256, RMSNormEps: 1e-6,
			RoPELocalBase: 10000, RoPEGlobalBase: 1000000, SlidingWindow: 1024,
			SlidingWindowPattern: 6, QueryPreAttnScalar: 16, GlobalHeadDim: 32,
			NumGlobalKVHeads: 1, AttentionKEqV: true, PartialRotaryFactor: 0.25,
			FinalLogitSoftcap: 30, // real Gemma-4 softcaps (gemma4_load_test); needed for FeatLogitSoftcap
			LayerTypes:        []string{"sliding_attention", "full_attention"},
		}
	case "gemma4_text":
		// The 26B-A4B MoE variant (enable_moe_block): the parallel dense+MoE FFN on
		// every layer, proportional global RoPE via the nested rope_parameters
		// (partial_rotary_factor 0.25 lives there, NOT top-level). Matches the tiny
		// checkpoint (testdata/gemma4-moe-tiny): v_proj on every layer (no K=V),
		// global_head_dim 512, PLE-free.
		return &Config{
			ModelType: "gemma4_text", HiddenDim: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2,
			HeadDim: 16, IntermediateDim: 48, VocabSize: 256, RMSNormEps: 1e-6,
			RoPELocalBase: 10000, RoPEGlobalBase: 1000000, SlidingWindow: 4,
			GlobalHeadDim: 512, FinalLogitSoftcap: 30,
			EnableMoeBlock: true, NumExperts: 8, TopKExperts: 2, MoeIntermediateSize: 16,
			RopeParameters:   json.RawMessage(`{"full_attention":{"rope_type":"proportional","partial_rotary_factor":0.25,"rope_theta":1000000.0},"sliding_attention":{"rope_type":"default","rope_theta":10000.0}}`),
			HiddenActivation: "gelu_pytorch_tanh",
			LayerTypes:       []string{"sliding_attention", "full_attention"},
		}
	case "qwen3":
		return &Config{
			ModelType: "qwen3", VocabSize: 128, HiddenDim: 16, NumLayers: 2, NumHeads: 4,
			NumKVHeads: 2, HeadDim: 4, IntermediateDim: 32, RMSNormEps: 1e-5,
			RoPEGlobalBase: 1000000, HiddenAct: "silu",
		}
	case "qwen2":
		return &Config{
			ModelType: "qwen2", VocabSize: 128, HiddenDim: 16, NumLayers: 2, NumHeads: 4,
			NumKVHeads: 2, IntermediateDim: 32, RMSNormEps: 1e-5, RoPEGlobalBase: 500000,
			HiddenAct: "silu",
		}
	case "qwen2_5_vl":
		// The qwen2_5_vl adapter degenerates to qwen2 for the text path; a top-level
		// rope_theta (no nested mrope_section) resolves with m-RoPE inactive.
		return &Config{
			ModelType: "qwen2_5_vl", VocabSize: 128, HiddenDim: 16, NumLayers: 2, NumHeads: 4,
			NumKVHeads: 2, IntermediateDim: 32, RMSNormEps: 1e-5, RoPEGlobalBase: 1000000,
			HiddenAct: "silu",
		}
	case "qwen2_moe":
		return &Config{
			ModelType: "qwen2_moe", VocabSize: 128, HiddenDim: 16, NumLayers: 2, NumHeads: 4,
			NumKVHeads: 2, IntermediateDim: 32, RMSNormEps: 1e-5, RoPEGlobalBase: 1000000,
			HiddenAct: "silu", NumExperts: 8, NumExpertsPerTok: 2, MoeIntermediateSize: 16,
			SharedExpertIntermediateSize: 16,
		}
	case "llama":
		return &Config{
			ModelType: "llama", VocabSize: 128, HiddenDim: 16, NumLayers: 2, NumHeads: 4,
			NumKVHeads: 2, IntermediateDim: 32, RMSNormEps: 1e-5, RoPEGlobalBase: 500000,
			HiddenAct: "silu",
		}
	case "mistral":
		return &Config{
			ModelType: "mistral", VocabSize: 128, HiddenDim: 16, NumLayers: 2, NumHeads: 4,
			NumKVHeads: 2, IntermediateDim: 32, RMSNormEps: 1e-5, RoPEGlobalBase: 1000000,
			HiddenAct: "silu", SlidingWindow: 4096,
		}
	case "gpt2":
		return &Config{
			ModelType: "gpt2", VocabSize: 64, NEmbd: 16, NHead: 4, NLayer: 2,
			NPositions: 32, LayerNormEpsilon: 1e-5, ActivationFunction: "gelu_new",
		}
	case "mixtral":
		return &Config{
			ModelType: "mixtral", VocabSize: 128, HiddenDim: 16, NumLayers: 2, NumHeads: 4,
			NumKVHeads: 2, IntermediateDim: 32, RMSNormEps: 1e-5, RoPEGlobalBase: 1000000,
			HiddenAct: "silu", NumLocalExperts: 8, NumExpertsPerTok: 2,
		}
	case "mellum":
		cfg := &Config{
			ModelType: "mellum", VocabSize: 256, HiddenDim: 64, NumLayers: 4, NumHeads: 4,
			NumKVHeads: 2, HeadDim: 16, IntermediateDim: 128, RMSNormEps: 1e-6, HiddenAct: "silu",
			NumExperts: 8, NumExpertsPerTok: 2, MoeIntermediateSize: 32, SlidingWindow: 1024,
			RopeParameters: json.RawMessage(`{"full_attention":{"rope_type":"yarn","rope_theta":500000.0,"factor":16.0,"original_max_position_embeddings":8192,"beta_fast":32.0,"beta_slow":1.0,"attention_factor":1.2772588722239782},"sliding_attention":{"rope_type":"default","rope_theta":500000.0}}`),
		}
		// 3:1 sliding/full interleave over NumLayers.
		for i := 0; i < cfg.NumLayers; i++ {
			if (i+1)%4 == 0 {
				cfg.LayerTypes = append(cfg.LayerTypes, "full_attention")
			} else {
				cfg.LayerTypes = append(cfg.LayerTypes, "sliding_attention")
			}
		}
		return cfg
	case "qwen3_5_moe", "qwen3_5_moe_text":
		return &Config{
			ModelType: modelType, VocabSize: 256, HiddenDim: 64, NumLayers: 4, NumHeads: 4,
			NumKVHeads: 2, HeadDim: 16, HiddenAct: "silu", RMSNormEps: 1e-6,
			LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
			LinearConvKernelDim: 4, LinearKeyHeadDim: 16, LinearValueHeadDim: 16,
			LinearNumKeyHeads: 2, LinearNumValueHeads: 4,
			NumExperts: 4, NumExpertsPerTok: 2, MoeIntermediateSize: 32, SharedExpertIntermediateSize: 32,
			RopeParameters: json.RawMessage(`{"rope_type":"default","rope_theta":1000000.0,"partial_rotary_factor":0.25}`),
		}
	case "glm4_moe":
		return &Config{
			ModelType: "glm4_moe", HiddenDim: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2,
			HeadDim: 16, VocabSize: 256, IntermediateDim: 128, RMSNormEps: 1e-6,
			NRoutedExperts: 8, NumExpertsPerTok: 2, NSharedExperts: 1, MoeIntermediateSize: 32,
			FirstKDenseReplace: 1, RoutedScalingFactor: 1.0, UseQKNorm: true, PartialRotaryFactor: 0.5,
			RopeParameters: json.RawMessage(`{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":0.5}`),
		}
	case "granitemoehybrid":
		return &Config{
			ModelType: "granitemoehybrid", HiddenDim: 64, NumLayers: 4, NumHeads: 4, NumKVHeads: 2,
			VocabSize: 256, IntermediateDim: 128, RMSNormEps: 1e-6,
			MambaNHeads: 8, MambaDHead: 16, MambaDState: 16, MambaDConv: 4, MambaNGroups: 1,
			NumLocalExperts: 8, NumExpertsPerTok: 2, SharedIntermediateSize: 128,
			LayerTypes:          []string{"mamba", "attention", "mamba", "mamba"},
			EmbeddingMultiplier: 12.0, AttentionMultiplier: 0.5, ResidualMultiplier: 0.22, LogitsScaling: 6.0,
			RopeParameters: json.RawMessage(`{"rope_type":"default","rope_theta":10000.0}`),
		}
	case "nemotron_h":
		return &Config{
			ModelType: "nemotron_h", HiddenDim: 64, NumLayers: 5, NumHeads: 4, NumKVHeads: 2,
			HeadDim: 16, VocabSize: 256, IntermediateDim: 128, LayerNormEpsilon: 1e-5,
			MambaNumHeads: 8, MambaHeadDim: 8, SSMStateSize: 16, NGroups: 1, ConvKernel: 4,
			LayersBlockType: []string{"mamba", "attention", "mlp", "mamba", "attention"},
		}
	case "deepseek_v2":
		// V2-Lite: no q-LoRA (QLoRARank 0), softmax routing (no ScoringFunc), no grouping.
		return &Config{
			ModelType: "deepseek_v2", HiddenDim: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 4,
			VocabSize: 128, IntermediateDim: 128, MoeIntermediateSize: 32, RMSNormEps: 1e-6,
			QLoRARank: 0, KVLoRARank: 16, QKNopeHeadDim: 16, QKRopeHeadDim: 8, VHeadDim: 16,
			NRoutedExperts: 8, NSharedExperts: 2, NumExpertsPerTok: 2, FirstKDenseReplace: 1,
			RoPEGlobalBase: 10000.0,
		}
	case "deepseek_v3":
		// V3: q-LoRA, sigmoid + group-limited routing (n_group/topk_group).
		return &Config{
			ModelType: "deepseek_v3", HiddenDim: 64, NumLayers: 4, NumHeads: 4, NumKVHeads: 4,
			VocabSize: 128, IntermediateDim: 128, MoeIntermediateSize: 32, RMSNormEps: 1e-6,
			QLoRARank: 24, KVLoRARank: 16, QKNopeHeadDim: 16, QKRopeHeadDim: 8, VHeadDim: 16,
			NRoutedExperts: 8, NSharedExperts: 1, NumExpertsPerTok: 2, NGroup: 2, TopkGroup: 1,
			FirstKDenseReplace: 1, RoutedScalingFactor: 2.5,
			RopeParameters: json.RawMessage(`{"rope_theta":10000.0,"rope_type":"default"}`),
		}
	case "kimi_k2":
		return &Config{
			ModelType: "kimi_k2", HiddenDim: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 8,
			VocabSize: 128, IntermediateDim: 128, MoeIntermediateSize: 32, RMSNormEps: 1e-6,
			QLoRARank: 24, KVLoRARank: 16, QKNopeHeadDim: 16, QKRopeHeadDim: 8, VHeadDim: 16,
			// Real Kimi K2 has 384 routed experts (registry.go:39). The representative must carry the
			// real count, not a tiny 16, or the generated hardware matrix shows Kimi WebGPU-resident
			// even though the real model exceeds the router-kernel cap and declines (M22, the
			// residentBackendMoECap check in features.go). Same representativeConfig-accuracy class as
			// C6's Mellum yarn.
			NRoutedExperts: 384, NSharedExperts: 1, NumExpertsPerTok: 4, NGroup: 1, TopkGroup: 1,
			FirstKDenseReplace: 1, RoutedScalingFactor: 2.827, ScoringFunc: "sigmoid",
			RopeParameters: json.RawMessage(`{"rope_theta":50000.0,"rope_type":"default"}`),
		}
	case "phi3":
		return &Config{
			ModelType: "phi3", HiddenDim: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2,
			VocabSize: 128, IntermediateDim: 128, RMSNormEps: 1e-5, RoPEGlobalBase: 10000,
		}
	case "llama4_text":
		return &Config{
			ModelType: "llama4_text", HiddenDim: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 4, HeadDim: 8,
			VocabSize: 128, IntermediateDim: 128, IntermediateSizeMLP: 192, RMSNormEps: 1e-5,
			NumLocalExperts: 4, NumExpertsPerTok: 1, MoeLayers: []int{1, 3}, NoRopeLayers: []int{1, 1, 0, 1},
			UseQKNorm: true, AttnTemperatureTuning: true, FloorScale: 4, AttnScaleL4: 0.1,
			RopeParameters: json.RawMessage(`{"rope_theta":10000.0,"rope_type":"default"}`),
		}
	case "gpt_oss":
		return &Config{
			ModelType: "gpt_oss", HiddenDim: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 2, HeadDim: 8,
			VocabSize: 128, MoeIntermediateSize: 32, NumLocalExperts: 4, NumExpertsPerTok: 2,
			SlidingWindow: 4, RMSNormEps: 1e-5, RoPEGlobalBase: 150000,
			RopeScaling: json.RawMessage(`{"rope_type":"yarn","factor":32,"beta_fast":32,"beta_slow":1,"original_max_position_embeddings":4096,"truncate":false}`),
		}
	}
	return nil
}

// coverageAxis classifies a resolved Architecture by its sequence mixer — the
// primary organizing axis of the matrix.
func coverageAxis(a *Architecture) string {
	switch {
	case a.mla != nil:
		return "latent-KV (MLA)"
	case a.granite != nil:
		return "state-space hybrid (Mamba-2)"
	case a.nemotron != nil:
		return "state-space hybrid (Mamba-2, single-op)"
	case a.qwen35 != nil:
		return "gated-linear hybrid (Gated DeltaNet)"
	default:
		return "softmax-GQA"
	}
}

// familyDoc carries PRESENTATION-ONLY facts with no Architecture equivalent.
// DisplayName + OneLineDesc are descriptive; Loaders + Modality are LOADER-CODE
// facts (which on-disk formats / tensor schemas a family supports, and whether a
// vision tower exists), hand-annotated here because they are NOT descriptor
// fields. Keyed by the canonical model_type (one key per registry alias).
type familyDoc struct {
	DisplayName string
	OneLineDesc string
	Loaders     string // hand-annotated loader-code fact (not from Architecture)
	Modality    string // hand-annotated loader-code fact (not from Architecture)
}

var familyDocs = map[string]familyDoc{
	"gemma3":           {"Gemma 3", "Google Gemma 3 dense (270M/1B/4B/12B/27B)", "safetensors, GGUF", "text (+ vision via VL text_config)"},
	"gemma3_text":      {"Gemma 3", "Google Gemma 3 dense (270M/1B/4B/12B/27B)", "safetensors, GGUF", "text (+ vision via VL text_config)"},
	"gemma4":           {"Gemma 4", "Google Gemma 4 dense + E-models (per-layer attention deltas, PLE)", "safetensors, GGUF", "text"},
	"gemma4_text":      {"Gemma 4", "Google Gemma 4 text (incl. 26B-A4B parallel dense+MoE FFN, enable_moe_block)", "safetensors, GGUF", "text"},
	"qwen3":            {"Qwen3", "Alibaba Qwen3 dense (QK-norm, no bias)", "safetensors, GGUF", "text"},
	"qwen2":            {"Qwen2 / Qwen2.5", "Alibaba Qwen2/2.5 dense (q/k/v bias)", "safetensors, GGUF", "text"},
	"qwen2_5_vl":       {"Qwen2.5-VL", "Qwen2.5-VL text decoder (qwen2 + m-RoPE)", "safetensors", "text (+ vision tower)"},
	"qwen2_moe":        {"Qwen2-MoE", "Qwen1.5/2 MoE (sparse + always-on shared expert)", "safetensors, GGUF", "text"},
	"llama":            {"Llama", "Meta Llama 2/3 dense (single-base RoPE)", "safetensors, GGUF, GPTQ, AWQ", "text"},
	"mistral":          {"Mistral", "Mistral dense (all-layer sliding window)", "safetensors, GGUF", "text"},
	"gpt2":             {"GPT-2", "GPT-2/NeoX (LayerNorm, learned positions, non-gated GELU)", "safetensors, GGUF", "text"},
	"mixtral":          {"Mixtral", "Mistral + sparse MoE FFN (router + top-k experts)", "safetensors, GGUF", "text"},
	"mellum":           {"Mellum2", "JetBrains Mellum2 code model (MoE + sliding/full interleave + YaRN)", "safetensors, GGUF", "text"},
	"qwen3_5_moe":      {"Qwen3.5-MoE", "Qwen3.5/3.6 hybrid: Gated DeltaNet + softmax + MoE", "safetensors, GGUF", "text"},
	"qwen3_5_moe_text": {"Qwen3.5-MoE", "Qwen3.5/3.6 hybrid: Gated DeltaNet + softmax + MoE", "safetensors, GGUF", "text"},
	"glm4_moe":         {"GLM-4.5/4.6", "Zhipu GLM-4.5/4.6 DeepSeek-style MoE (sigmoid routing + dense prefix)", "safetensors, GGUF", "text"},
	"granitemoehybrid": {"Granite-4.0-H", "IBM Granite-4.0-H: Mamba-2 + attention hybrid + MoE-on-every-layer", "safetensors, GGUF", "text"},
	"nemotron_h":       {"Nemotron-H", "NVIDIA Nemotron-H single-op-per-block hybrid (mamba | attention | relu² MLP)", "safetensors, GGUF", "text"},
	"deepseek_v2":      {"DeepSeek-V2", "DeepSeek-V2/V2-Lite: MLA + DeepSeekMoE (softmax routing)", "safetensors, GGUF", "text"},
	"deepseek_v3":      {"DeepSeek-V3", "DeepSeek-V3: MLA + DeepSeekMoE (sigmoid + group-limited routing)", "safetensors, GGUF", "text"},
	"kimi_k2":          {"Kimi K2", "Moonshot Kimi K2 / K2.5 / K2.6 / K2.7-Code (DeepseekV3 arch: MLA + DeepSeekMoE — same arch across the K2.x line)", "safetensors, GGUF", "text"},
	"phi3":             {"Phi-3 / Phi-4", "Microsoft Phi-3/Phi-4 dense (fused qkv/gate-up, partial rotary)", "safetensors, GGUF", "text"},
	"llama4_text":      {"Llama 4", "Meta Llama 4 (Scout/Maverick) text decoder: iRoPE (RoPE/NoPE interleave) + L2 QK-norm + attn-temp + dense/MoE interleave (top-1 sigmoid + shared)", "safetensors, GGUF", "text"},
	"gpt_oss":          {"gpt-oss", "OpenAI gpt-oss 20b/120b sparse MoE: per-head attention sinks + clamped interleaved-SwiGLU + alternating sliding/full + YaRN (MXFP4 experts, CPU-only)", "GGUF", "text"},
}

// capabilityRow is one family's row in the matrix (alias group → one row). All
// columns except Loaders/Modality/DisplayName/OneLineDesc are DERIVED from the
// resolved Architecture.
type capabilityRow struct {
	Name          string   `json:"name"` // resolved Architecture.Name (the group key)
	Aliases       []string `json:"model_types"`
	DisplayName   string   `json:"display_name"`
	OneLineDesc   string   `json:"description"`
	CoverageAxis  string   `json:"coverage_axis"`
	MoE           string   `json:"moe"`
	SlidingWindow string   `json:"sliding_window"`
	QKNorm        bool     `json:"qk_norm"`
	RoPEStyle     string   `json:"rope_style"`
	Norm          string   `json:"norm"`
	Activation    string   `json:"activation"`
	TiedLMHead    bool     `json:"tied_lm_head"`
	Loaders       string   `json:"loaders"`
	Modality      string   `json:"modality"`
	GPUResident   bool     `json:"gpu_residency_eligible"`
	Parity        string   `json:"parity"` // joined from testdata/parity_manifest.json by Name
}

// parityColumn renders a family's parity cell from its manifest entry: a short
// method label + argmax%/cosine_min for validated families with a numeric oracle,
// just the method label for purely qualitative gates (coherent-generation on a real
// model with no feasible oracle), and "pending" otherwise. Method labels:
//
//   - full-oracle / real-oracle / weight-diff: cosine vs a RELEASED model.
//   - tiny-oracle: cosine vs the family's tiny HF-seeded golden (every family's first
//     gate) — the strongest NUMERIC oracle when no released model is small enough to
//     diff (e.g. Llama 4 Scout is 109B / Q2_K-only). "tiny-golden+coherent" additionally
//     ran a real model qualitatively, rendered with a " +coherent" suffix so the cell
//     shows BOTH the tiny numeric proof and that real weights were exercised.
func parityColumn(fam familyParity) string {
	if string(fam.Status) != `"validated"` {
		return "pending"
	}
	var method string
	_ = json.Unmarshal(fam.Method, &method)
	suffix := ""
	switch method {
	case "full-forward-oracle":
		method = "full-oracle"
	case "weightDiff":
		method = "weight-diff"
	case "real-model-oracle":
		method = "real-oracle"
	case "tiny-golden":
		method = "tiny-oracle"
	case "tiny-golden+coherent":
		method, suffix = "tiny-oracle", " +coherent"
	case "coherent-generation":
		return "coherent-gen" // qualitative real-model gate, no numeric oracle at all
	}
	var metrics struct {
		ArgmaxPct float64 `json:"argmax_pct"`
		CosineMin float64 `json:"cosine_min"`
	}
	_ = json.Unmarshal(fam.Metrics, &metrics)
	return fmt.Sprintf("%s %.1f%%/%.5f%s", method, metrics.ArgmaxPct, metrics.CosineMin, suffix)
}

// loadParityManifest reads testdata/parity_manifest.json for the matrix join.
func loadParityManifest() (*parityManifest, error) {
	raw, err := os.ReadFile(parityManifestPath)
	if err != nil {
		return nil, err
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// moeColumn renders the MoE column from the resolved descriptor. Expert/top-k
// counts vary by checkpoint, so only the family-constant structure is printed.
func moeColumn(a *Architecture) string {
	if a.MoE == nil {
		return "dense"
	}
	if a.MoE.SharedIntermediateDim > 0 {
		return "sparse +shared"
	}
	return "sparse, no-shared"
}

// slidingWindowColumn renders the sliding-window column. all-layer vs interleave
// is distinguished by probing the resolved descriptor's per-layer global/local
// classification across [0, NumLayers): all-local = all-layer sliding (mistral),
// a global/local mix = interleave (gemma/gemma4/mellum/qwen3_5).
func slidingWindowColumn(a *Architecture) string {
	if a.SlidingWindow == 0 {
		return "none"
	}
	nGlobal := 0
	for i := 0; i < a.NumLayers; i++ {
		if a.isGlobalLayer(i) {
			nGlobal++
		}
	}
	switch {
	case nGlobal == 0:
		return "all-layer" // mistral
	case nGlobal == a.NumLayers:
		return "none" // degenerate
	default:
		return "interleave" // gemma/gemma4/mellum
	}
}

// ropeStyleColumn derives the RoPE style from the descriptor.
func ropeStyleColumn(a *Architecture) string {
	switch {
	case a.LearnedPosEmbed:
		return "learned/none"
	case a.RoPEGlobalBase <= 0:
		return "none"
	case a.MRopeSection != nil:
		return "m-RoPE"
	case a.RotaryDim > 0 && a.RotaryDim < a.HeadDim:
		// The partial-ness is family-constant; the absolute dim is per-checkpoint.
		return "partial"
	case a.RoPELocalBase != a.RoPEGlobalBase:
		return "dual-base"
	default:
		return "full"
	}
}

// activationColumn renders the activation, marking the gated vs non-gated split.
func activationColumn(a *Architecture) string {
	if a.NonGatedMLP {
		return a.Act.String() + " (non-gated)"
	}
	// Gated: SiLU→SwiGLU, GELU-tanh→GeGLU (the conventional names).
	switch a.Act {
	case ActSiLU:
		return "SwiGLU"
	case ActGeluTanh:
		return "GeGLU"
	default:
		return a.Act.String() + " (gated)"
	}
}

// normColumn renders norm kind + placement.
func normColumn(a *Architecture) string {
	return a.Norm.String() + ", " + a.NormPlacement.String()
}

// buildMatrix resolves every registry key, groups aliases by resolved
// Architecture.Name, and returns the rows sorted deterministically (by coverage
// axis, then display name, then name). The second return is the list of any keys
// that failed to resolve.
func buildMatrix(t *testing.T) ([]capabilityRow, error) {
	t.Helper()
	keys := make([]string, 0, len(registry))
	for k := range registry {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	manifest, err := loadParityManifest()
	if err != nil {
		return nil, fmt.Errorf("load parity manifest: %w", err)
	}

	rows := map[string]*capabilityRow{} // by Architecture.Name
	for _, mt := range keys {
		cfg := representativeConfig(mt)
		if cfg == nil {
			return nil, fmt.Errorf("model_type %q has no representativeConfig", mt)
		}
		doc, ok := familyDocs[mt]
		if !ok {
			return nil, fmt.Errorf("model_type %q has no familyDoc", mt)
		}
		arch, _, err := resolveArchitecture(cfg)
		if err != nil {
			return nil, fmt.Errorf("model_type %q failed to resolve: %w", mt, err)
		}
		r, exists := rows[arch.Name]
		if !exists {
			parity := "pending"
			if fam, ok := manifest.Families[arch.Name]; ok {
				parity = parityColumn(fam)
			}
			r = &capabilityRow{
				Name:          arch.Name,
				DisplayName:   doc.DisplayName,
				OneLineDesc:   doc.OneLineDesc,
				CoverageAxis:  coverageAxis(arch),
				MoE:           moeColumn(arch),
				SlidingWindow: slidingWindowColumn(arch),
				QKNorm:        arch.QKNorm,
				RoPEStyle:     ropeStyleColumn(arch),
				Norm:          normColumn(arch),
				Activation:    activationColumn(arch),
				TiedLMHead:    arch.TiedLMHead,
				Loaders:       doc.Loaders,
				Modality:      doc.Modality,
				GPUResident:   arch.decodeRunnerEligible(),
				Parity:        parity,
			}
			rows[arch.Name] = r
		}
		r.Aliases = append(r.Aliases, mt)
	}

	out := make([]capabilityRow, 0, len(rows))
	for _, r := range rows {
		sort.Strings(r.Aliases)
		out = append(out, *r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CoverageAxis != out[j].CoverageAxis {
			return out[i].CoverageAxis < out[j].CoverageAxis
		}
		if out[i].DisplayName != out[j].DisplayName {
			return out[i].DisplayName < out[j].DisplayName
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

const matrixHeader = "<!-- GENERATED by go test ./decoder -run CapabilityMatrix -update — DO NOT EDIT -->"

func yn(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// renderMarkdown renders the matrix as a Markdown doc grouped by coverage axis.
func renderMarkdown(rows []capabilityRow) []byte {
	var b strings.Builder
	b.WriteString(matrixHeader + "\n\n")
	b.WriteString("# goinfer capability matrix\n\n")
	b.WriteString("Generated from the `decoder` registry — the Go adapters are the single\n")
	b.WriteString("source of truth. Each row is one model family (model_type aliases that share\n")
	b.WriteString("an adapter are grouped). Per-checkpoint dims (hidden size, layer/expert\n")
	b.WriteString("counts) are intentionally excluded — they vary by checkpoint. Regenerate\n")
	b.WriteString("with `go test ./decoder -run CapabilityMatrix -update`.\n\n")
	b.WriteString("**Parity** shows each family's strongest validation as `method argmax%/cosine_min`: ")
	b.WriteString("`full-oracle`/`real-oracle`/`weight-diff` diff against a RELEASED model; ")
	b.WriteString("`tiny-oracle` against the family's tiny HF-seeded golden (used when no released ")
	b.WriteString("model is small enough to diff); a `+coherent` suffix means a real model also ran ")
	b.WriteString("qualitatively; `coherent-gen` is a real-model coherence check with no numeric ")
	b.WriteString("oracle; `pending` is not yet recorded. Source: `testdata/parity_manifest.json`.\n\n")

	// Group by coverage axis, preserving the sorted order.
	axes := []string{}
	byAxis := map[string][]capabilityRow{}
	for _, r := range rows {
		if _, ok := byAxis[r.CoverageAxis]; !ok {
			axes = append(axes, r.CoverageAxis)
		}
		byAxis[r.CoverageAxis] = append(byAxis[r.CoverageAxis], r)
	}

	cols := "| Family | model_type(s) | MoE | Sliding window | QK-norm | RoPE | Norm | Activation | Tied head | Loaders | Modality | GPU-resident | Parity |\n"
	sep := "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

	for _, axis := range axes {
		b.WriteString("## " + axis + "\n\n")
		for _, r := range byAxis[axis] {
			b.WriteString("> **" + r.DisplayName + "** — " + r.OneLineDesc + "\n\n")
		}
		b.WriteString(cols)
		b.WriteString(sep)
		for _, r := range byAxis[axis] {
			b.WriteString(fmt.Sprintf("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
				r.DisplayName,
				strings.Join(r.Aliases, "`, `"),
				r.MoE,
				r.SlidingWindow,
				yn(r.QKNorm),
				r.RoPEStyle,
				r.Norm,
				r.Activation,
				yn(r.TiedLMHead),
				r.Loaders,
				r.Modality,
				yn(r.GPUResident),
				r.Parity,
			))
		}
		b.WriteString("\n")
	}
	return []byte(b.String())
}

// renderJSON renders the matrix as deterministic JSON (one object per family).
func renderJSON(rows []capabilityRow) ([]byte, error) {
	out, err := json.MarshalIndent(rows, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(out, '\n'), nil
}

func TestCapabilityMatrix(t *testing.T) {
	rows, err := buildMatrix(t)
	if err != nil {
		t.Fatalf("build matrix: %v", err)
	}

	md := renderMarkdown(rows)
	js, err := renderJSON(rows)
	if err != nil {
		t.Fatalf("render json: %v", err)
	}

	const mdPath = "../docs/capability-matrix.md"
	const jsPath = "../docs/capability-matrix.json"

	if *updateMatrix {
		if err := os.WriteFile(mdPath, md, 0o644); err != nil {
			t.Fatalf("write %s: %v", mdPath, err)
		}
		if err := os.WriteFile(jsPath, js, 0o644); err != nil {
			t.Fatalf("write %s: %v", jsPath, err)
		}
		t.Logf("wrote %s and %s", mdPath, jsPath)
		return
	}

	checkFresh := func(path string, want []byte) {
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v (capability matrix stale — regenerate: go test ./decoder -run CapabilityMatrix -update)", path, err)
		}
		// stripCR so the comparison is EOL-agnostic — a Windows checkout stores the committed doc
		// as CRLF while `want` is always LF-generated. Applies to the .json as much as the .md: it
		// is generated text under the same gate, so it has the same exposure. (See stripCR in
		// hardware_matrix_test.go; .gitattributes pins both docs to eol=lf as the other half.)
		if !bytes.Equal(stripCR(got), want) {
			t.Fatalf("%s out of date — capability matrix stale — regenerate: go test ./decoder -run CapabilityMatrix -update", path)
		}
	}
	checkFresh(mdPath, md)
	checkFresh(jsPath, js)
}

// TestCapabilityMatrix_CoverageComplete asserts every registry key has both a
// representativeConfig and a familyDoc, so adding a family without these fails CI.
func TestCapabilityMatrix_CoverageComplete(t *testing.T) {
	for mt := range registry {
		if representativeConfig(mt) == nil {
			t.Errorf("model_type %q has no representativeConfig", mt)
		}
		if _, ok := familyDocs[mt]; !ok {
			t.Errorf("model_type %q has no familyDoc", mt)
		}
	}
}

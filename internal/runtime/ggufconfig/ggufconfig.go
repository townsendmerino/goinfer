package ggufconfig

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/townsendmerino/aikit/embed"
)

// ggufConfig synthesizes a Config from GGUF metadata, dispatching on
// general.architecture, with the file's recommended sampling (ggufSampling). The resolved Config feeds
// resolveArchitecture, so a GGUF reuses the same per-family adapter as the safetensors path.
func ggufConfig(g *embed.GGUFFile) (cfg *Config, err error) {
	arch, _ := g.Str("general.architecture")
	// Hostile metadata can drive a family builder into an integer divide-by-zero (a missing head_count makes hidden/0) or a
	// similar panic before ValidateDims runs. Any such panic becomes a typed error, so the loader keeps its never-panic
	// contract for every architecture, current and future. Unbounded per-layer allocations are guarded explicitly by
	// ggufLayerCount: a huge makeslice is a fatal OOM that recover cannot catch.
	defer func() {
		if r := recover(); r != nil {
			cfg, err = nil, fmt.Errorf("decoder(gguf): malformed metadata for architecture %q: %v", arch, r)
		}
		if err == nil && cfg != nil {
			cfg.Sampling = ggufSampling(g)
		}
	}()
	if key := ggufHadamardKey(g); key != "" {
		return nil, unsupported("decoder(gguf): this file's weights are Hadamard-rotated (metadata %q) and need a matching "+
			"activation transform on every forward pass that goinfer does not implement, so any read of them would generate garbage; not loaded", key)
	}
	if adapter, ok := ggufRegistry[arch]; ok {
		return adapter(g)
	}
	return nil, unsupported("decoder(gguf): architecture %q unsupported (have: %s)", arch, knownGGUFArchitectures())
}

// ggufHadamardKey is the first metadata key under llama.cpp's "prism.hadamard." prefix (the marker PrismML's Bonsai 2 files carry for a
// blockwise Hadamard rotation folded into the weights), "" when there is none. Any key under the prefix declines: the file is rotated
// whatever its tensor types, a third-party requantization included.
func ggufHadamardKey(g *embed.GGUFFile) string {
	first := ""
	for k := range g.Metadata {
		if strings.HasPrefix(k, "prism.hadamard.") && (first == "" || k < first) {
			first = k
		}
	}
	return first
}

// ggufConfigAdapter builds a *Config from a GGUF file's metadata for one general.architecture value.
type ggufConfigAdapter func(*embed.GGUFFile) (*Config, error)

// ggufRegistry maps GGUF general.architecture → its Config builder, keeping GGUF architecture registration in the same
// table-driven file as safetensors registry above.
var ggufRegistry = map[string]ggufConfigAdapter{
	"llama":          ggufLlamaConfig,
	"qwen2":          ggufQwen2Config,
	"qwen3":          ggufQwen3Config,
	"gemma":          func(g *embed.GGUFFile) (*Config, error) { return ggufGemma12Config(g, "gemma") },
	"gemma2":         func(g *embed.GGUFFile) (*Config, error) { return ggufGemma12Config(g, "gemma2") },
	"gemma3":         ggufGemmaConfig,
	"gemma3_text":    ggufGemmaConfig,
	"gemma4":         ggufGemma4Config,
	"mellum":         ggufMellumConfig,
	"qwen35moe":      ggufQwen35Config,
	"qwen35":         ggufQwen35DenseConfig,
	"qwen3moe":       ggufQwen3MoeConfig,
	"laguna":         ggufLagunaConfig,
	"flint":          ggufFlintConfig,
	"glm4moe":        ggufGlm4MoeConfig,
	"granitehybrid":  ggufGraniteConfig,
	"granite":        ggufGraniteDenseConfig,
	"nemotron_h":     ggufNemotronConfig,
	"nemotron_h_moe": ggufNemotronConfig, // Nemotron 3 Nano — same HF model_type "nemotron_h", llama.cpp gives it its own GGUF arch string
	"deepseek2":      ggufDeepseekConfig,
	"phi3":           ggufPhi3Config,
	"llama4":         ggufLlama4Config,
	"gpt-oss":        ggufGptOssConfig,
	"inkling2":       func(g *embed.GGUFFile) (*Config, error) { return ggufInkling2Config(g, "inkling2") },
	"inkling2_loop":  func(g *embed.GGUFFile) (*Config, error) { return ggufInkling2Config(g, "inkling2_loop") },
}

func knownGGUFArchitectures() string {
	return strings.Join(slices.Sorted(maps.Keys(ggufRegistry)), ", ")
}

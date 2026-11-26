package ggufconfig

import (
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// TestGGUFConfig_hostileMetadata_M16 covers the family builders the old corpus
// never reached: a huge block_count must be a typed error (not a multi-TB makeslice)
// before the per-layer allocation, and a missing head_count that drives hidden/0 must
// be a typed error (not an integer divide panic). ggufConfig is the fuzz contract's
// entry point, so it must honor "never panic" for every architecture.
func TestGGUFConfig_hostileMetadata_M16(t *testing.T) {
	emb := []ggufTensorDecl{embedTensor(8, 32)}
	cases := []struct {
		name string
		kvs  []ggufKV
	}{
		{"granite huge block_count", []ggufKV{
			kvStr("general.architecture", "granitehybrid"),
			kvU32("granitehybrid.embedding_length", 8),
			kvU64("granitehybrid.block_count", 1<<40),
			kvU32("granitehybrid.attention.head_count", 4),
		}},
		{"nemotron huge block_count", []ggufKV{
			kvStr("general.architecture", "nemotron_h"),
			kvU32("nemotron_h.embedding_length", 8),
			kvU64("nemotron_h.block_count", 1<<40),
			kvU32("nemotron_h.attention.head_count", 4),
		}},
		{"llama4 huge block_count", []ggufKV{
			kvStr("general.architecture", "llama4"),
			kvU32("llama4.embedding_length", 8),
			kvU64("llama4.block_count", 1<<40),
			kvU32("llama4.attention.head_count", 4),
		}},
		{"phi3 zero head_count divides hidden", []ggufKV{
			kvStr("general.architecture", "phi3"),
			kvU32("phi3.embedding_length", 8),
			kvU32("phi3.block_count", 2),
			kvU32("phi3.attention.head_count", 0),
			kvU32("phi3.feed_forward_length", 16),
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := embed.OpenGGUFBytes(buildGGUF(tc.kvs, emb))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			defer g.Close()
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ggufConfig panicked on hostile metadata: %v", r)
				}
			}()
			if _, err := ggufConfig(g); err == nil {
				t.Fatalf("expected a typed error, got nil")
			}
		})
	}
}

// ggufSeeds builds one tiny, valid GGUF per supported architecture so the mutator
// starts from headers that reach each family config builder.
func ggufSeeds() [][]byte {
	emb := embedTensor(8, 32)
	dense := func(arch string) []byte {
		return buildGGUF([]ggufKV{
			kvStr("general.architecture", arch),
			kvU32(arch+".embedding_length", 8),
			kvU32(arch+".block_count", 2),
			kvU32(arch+".attention.head_count", 4),
			kvU32(arch+".attention.head_count_kv", 2),
			kvU32(arch+".attention.key_length", 2),
			kvU32(arch+".feed_forward_length", 16),
			kvU32(arch+".attention.sliding_window", 4),
			kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-6),
			kvF32(arch+".rope.freq_base", 10000),
			kvU32("tokenizer.ggml.eos_token_id", 1),
		}, []ggufTensorDecl{emb})
	}
	gemma4 := buildGGUF([]ggufKV{
		kvStr("general.architecture", "gemma4"),
		kvU32("gemma4.embedding_length", 8),
		kvU32("gemma4.block_count", 2),
		kvU32("gemma4.attention.head_count", 4),
		kvU32("gemma4.attention.head_count_kv", 2),
		kvU32("gemma4.attention.key_length", 4),
		kvU32("gemma4.attention.key_length_swa", 2),
		kvU32("gemma4.feed_forward_length", 16),
		kvU32("gemma4.attention.sliding_window", 4),
		kvU32("gemma4.attention.shared_kv_layers", 1),
		kvU32("gemma4.embedding_length_per_layer_input", 4),
		kvBoolArr("gemma4.attention.sliding_window_pattern", []bool{true, false}),
		kvF32("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
	}, []ggufTensorDecl{emb})
	mellum := buildGGUF([]ggufKV{
		kvStr("general.architecture", "mellum"),
		kvU32("mellum.embedding_length", 8),
		kvU32("mellum.block_count", 2),
		kvU32("mellum.attention.head_count", 4),
		kvU32("mellum.attention.head_count_kv", 2),
		kvU32("mellum.attention.key_length", 2),
		kvU32("mellum.feed_forward_length", 16),
		kvU32("mellum.expert_feed_forward_length", 8),
		kvU32("mellum.expert_count", 4),
		kvU32("mellum.expert_used_count", 2),
		kvU32("mellum.attention.sliding_window", 4),
		kvF32("mellum.attention.layer_norm_rms_epsilon", 1e-6),
		kvF32("mellum.rope.freq_base", 10000),
		kvBoolArr("mellum.attention.sliding_window_pattern", []bool{true, false}),
	}, []ggufTensorDecl{emb})

	var seeds [][]byte
	// Every dispatchable architecture gets a seed so the fuzzer exercises its family builder: the div-by-zero / makeslice
	// sites M16 hardened live inside these. A generic dense seed is enough to drive each builder's entry; the fuzzer
	// mutates from there.
	for _, a := range []string{
		"llama", "qwen2", "qwen3", "gemma3", "phi3",
		"deepseek2", "glm4moe", "granitehybrid", "nemotron_h", "qwen35moe", "llama4",
	} {
		seeds = append(seeds, dense(a))
	}
	seeds = append(seeds, gemma4, mellum)
	seeds = append(seeds, gU32(0x46554747), buildGGUF(nil, nil)) // raw-edge: truncated, header-only
	// hostile: a well-formed header claiming absurd tensor + kv counts with no payload — the
	// count→make DoS surface a parser must bound against the file size before allocating.
	hostileCounts := slices.Clone(gU32(0x46554747))       // "GGUF"
	hostileCounts = append(hostileCounts, gU32(3)...)     // version
	hostileCounts = append(hostileCounts, gU64(1<<48)...) // tensor_count
	hostileCounts = append(hostileCounts, gU64(1<<48)...) // kv_count
	seeds = append(seeds, hostileCounts)
	return seeds
}

// FuzzGGUFConfig fuzzes goinfer's GGUF metadata → Config interpretation (the
// per-family config builders). With aikit's container parse hardened (v1.2.1),
// arbitrary bytes either fail aikit's typed parse or reach ggufConfig, which must
// never panic. And the dim gate's contract: if ValidateDims accepts a config,
// its core dims must be safe to allocate (positive, within bounds) — the gate is
// what stands between a hostile uint64→int overflow and a make() panic/OOM.
func FuzzGGUFConfig(f *testing.F) {
	for _, s := range ggufSeeds() {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		g, err := embed.OpenGGUFBytes(raw)
		if err != nil {
			return // aikit rejected the container — not goinfer's layer
		}
		defer g.Close()
		cfg, err := ggufConfig(g)
		if err != nil {
			return // a typed interpretation error is the correct outcome
		}
		if ValidateDims(cfg) == nil { // accepted ⇒ must be safe to allocate
			if cfg.NumLayers <= 0 || cfg.NumLayers > MaxLayers ||
				cfg.HiddenDim <= 0 || cfg.HiddenDim > MaxHidden ||
				cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 ||
				cfg.VocabSize <= 0 || cfg.VocabSize > MaxVocabSize {
				t.Fatalf("ValidateDims accepted an unsafe config: %+v", cfg)
			}
		}
	})
}

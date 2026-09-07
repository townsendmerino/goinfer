package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/embed"
)

// R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): found while wiring up the new
// request-time memory guard (AdmitPrefillMemory, prefill_budget.go) — Config.MaxPositions came
// back 0 for a real, freshly-downloaded qwen2.5-coder-0.5b GGUF, silently disabling both the new
// guard and the pre-existing contextLengthError/clampMaxTokens checks (C-18/C-20). The cause: 16
// of the 18 GGUF architecture config builders never read "context_length" into
// Config.MaxPositions at all — only ggufPhi3Config did, by what looks like accident rather than
// design (nothing about that family is special). This is a regression gate for every one of
// them, driven through the real dispatch table (ggufConfig), not the individual functions, so a
// future architecture that forgets the field fails here too.
//
// Mutation: comment out any one architecture's `MaxPositions: u("context_length"),` line — that
// architecture's subtest goes red with MaxPositions=0 (want 4096); every other subtest stays
// green, isolating exactly which family broke.
func TestGGUFConfig_everyArchitectureReadsMaxPositions(t *testing.T) {
	const wantCtx = 4096
	emb := []ggufTensorDecl{embedTensor(8, 32)}

	// dense mirrors ggufSeeds' own minimal-valid-GGUF shape (gguf_dims_test.go) plus a
	// context_length key — every field a dense or MoE family builder needs to reach its
	// MaxPositions assignment without erroring or panicking.
	dense := func(arch string) []byte {
		return buildGGUF([]ggufKV{
			kvStr("general.architecture", arch),
			kvU32(arch+".context_length", wantCtx),
			kvU32(arch+".embedding_length", 8),
			kvU32(arch+".block_count", 2),
			kvU32(arch+".attention.head_count", 4),
			kvU32(arch+".attention.head_count_kv", 2),
			kvU32(arch+".attention.key_length", 2),
			kvU32(arch+".feed_forward_length", 16),
			kvU32(arch+".attention.sliding_window", 4),
			kvF32(arch+".attention.layer_norm_rms_epsilon", 1e-6),
			kvF32(arch+".rope.freq_base", 10000),
			// granitehybrid/nemotron_h's Mamba head-dim math divides by ssm.time_step_rank;
			// every other architecture ignores these two keys, so setting them unconditionally
			// is harmless.
			kvU32(arch+".ssm.time_step_rank", 1),
			kvU32(arch+".ssm.inner_size", 2),
			kvU32("tokenizer.ggml.eos_token_id", 1),
		}, emb)
	}
	gemma4 := buildGGUF([]ggufKV{
		kvStr("general.architecture", "gemma4"),
		kvU32("gemma4.context_length", wantCtx),
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
	}, emb)
	mellum := buildGGUF([]ggufKV{
		kvStr("general.architecture", "mellum"),
		kvU32("mellum.context_length", wantCtx),
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
	}, emb)

	cases := []struct {
		name string
		raw  []byte
	}{
		{"llama", dense("llama")},
		{"qwen2", dense("qwen2")},
		{"qwen3", dense("qwen3")},
		{"gemma3", dense("gemma3")},
		{"gemma4", gemma4},
		{"mellum", mellum},
		{"qwen35moe", dense("qwen35moe")},
		{"qwen35", dense("qwen35")},
		{"qwen3moe", dense("qwen3moe")},
		{"laguna", dense("laguna")},
		{"glm4moe", dense("glm4moe")},
		{"granitehybrid", dense("granitehybrid")},
		{"granite", dense("granite")},
		{"nemotron_h", dense("nemotron_h")},
		{"deepseek2", dense("deepseek2")},
		{"phi3", dense("phi3")},
		{"llama4", dense("llama4")},
		{"gpt-oss", dense("gpt-oss")},
	}
	if len(cases) != 18 {
		t.Fatalf("this covers %d architectures; ggufConfig's switch dispatches 18 functions (19 case labels, nemotron_h_moe aliases nemotron_h) — update this table, not just its count", len(cases))
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := embed.OpenGGUFBytes(tc.raw)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			defer g.Close()
			cfg, err := ggufConfig(g)
			if err != nil {
				t.Fatalf("ggufConfig(%s): %v", tc.name, err)
			}
			if cfg.MaxPositions != wantCtx {
				t.Errorf("%s: MaxPositions = %d, want %d (context_length silently dropped)", tc.name, cfg.MaxPositions, wantCtx)
			}
		})
	}
}

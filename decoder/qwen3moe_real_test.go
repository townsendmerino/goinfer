//go:build realckpt

// Real-checkpoint T3 for Qwen3-MoE (Qwen3-30B-A3B, model_type "qwen3_moe"), F1 of docs/completed/task-families-2026-09.md.
// The tiny random-weight HF oracle proved the loader/forward shapes; this proves the released weights, which is a different
// claim: a tiny fixture cannot catch a wrong tensor name, a transposed expert stack, or a router read from the wrong key,
// every one of which produces correct shapes and plausible values.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen3MoeReal_oracle -v -timeout 90m
package decoder

import "testing"

func TestQwen3MoeReal_oracle(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN3MOE_HF")
	// int8 WEIGHTS, f32 ACTIVATIONS as the starting quant, not int8int8: this family routes 8 of 128 experts (6.25%), the same
	// order of sparsity as nemotron_h's MoE variant, which hit a real router-flip cliff at int8 activations
	// (docs/completed/queue-correctness.md G4). Start from the safer quant and measure int8int8 separately if this passes.
	realLogitOracleQuant(t, ckpt, "../testdata/qwen3moe_real_golden.json.gz", "qwen3_moe", "qwen3_moe",
		"HF bf16 (Qwen/Qwen3-30B-A3B; int8 weights, f32 activations)", "int8")
}

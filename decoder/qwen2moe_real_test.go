//go:build realckpt

// Real-checkpoint T3 for Qwen2-MoE (Qwen1.5-MoE-A2.7B, model_type "qwen2_moe") — the T3
// promotion of the qwen2_moe family from tiny-golden to a released checkpoint. The existing
// tiny-golden gate (TestQwen2Moe_forwardParity) already loads a real-but-tiny-random checkpoint
// (katuni4ka/tiny-random-qwen1.5-moe) for structural coverage; this proves the adapter on
// released weights at the real expert count — a tiny fixture cannot catch a wrong tensor name,
// a transposed expert stack, or a shared-expert gate wired to the wrong tensor, every one of
// which produces correct shapes and plausible values. int8 weights, f32 activations: 14.3B
// total params at f32 would need ~57 GB, too tight on this box; the int8-vs-bf16 shape is the
// one deepseek_v2/v3, mellum, and qwen3_moe already use. Fixture: scripts/pin_qwen2moe_real.py.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen2MoeReal_oracle -v -timeout 90m
package decoder

import "testing"

func TestQwen2MoeReal_oracle(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN2MOE_HF")
	realLogitOracleQuant(t, ckpt, "../testdata/qwen2moe_real_golden.json", "qwen2_moe", "qwen2_moe",
		"HF bf16 (Qwen/Qwen1.5-MoE-A2.7B; int8 weights, f32 activations)", "int8")
}

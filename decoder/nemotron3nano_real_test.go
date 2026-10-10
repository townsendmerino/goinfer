//go:build realckpt

// Real-checkpoint T3 for Nemotron 3 Nano (nemotron_h MoE, 30B-A3B) — step 3 of the
// family-onboarding checklist (docs/parity-coverage-policy.md).
//
// The family's dense parent (nemotron_h, 9B) already has a T3 row; this is the MoE variant,
// which exercises everything the dense checkpoint cannot: the noaux_tc router (sigmoid +
// e_score_correction_bias + group-limited top-k over 128 experts, 6 active), NON-GATED relu²
// experts, the shared expert at its own intermediate size (3712, which is NOT
// n_shared_experts*moe_intermediate_size — the real config ships a distinct value), and the
// 52-block MEMEM* pattern interleaving Mamba-2 / MoE-FFN / NoPE-attention.
//
// T1 (tiny random-weight HF oracle, cosine 1.000000) proved the shapes. This proves the released
// weights, which is a different claim: a tiny fixture cannot catch a wrong tensor NAME, a
// transposed expert stack, or a router bias read from the wrong key — every one of which
// produces correct shapes and plausible values.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestNemotron3NanoReal -v -timeout 90m
package decoder

import "testing"

func TestNemotron3NanoReal_oracle(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_NEMOTRON3NANO_HF")
	// int8 WEIGHTS, f32 ACTIVATIONS, not the int8int8 every other family uses: this model routes 6 of 128 experts, far
	// sparser than the comparable MoE families here, and quantizing activations perturbs the router enough to flip which
	// experts run, a discrete change no averaging smooths (the expert-flip cliff recorded for granite's MoE stack). The
	// forward is correct; the sensitivity is real. So this is a deployment fact, not a threshold dodge: DO NOT run this
	// family's MoE variant with int8 activations. Measured cosines: docs/code-notes/decoder.md#TestNemotron3NanoReal_oracle.int8.
	realLogitOracleQuant(t, ckpt, "../testdata/nemotron3nano_real_golden.json.gz", "nemotron_h", "nemotron_h",
		"HF bf16 (NVIDIA-Nemotron-3-Nano-30B-A3B-BF16; int8 weights, f32 activations)", "int8")
}

//go:build realckpt

// Real-checkpoint T3 for Qwen3-Next (qwen3_next, 80B-A3B hybrid) — the row the family has been
// missing since 2026-08-17, when it was blocked as "no full reference forward of a 163GB bf16
// model fits 62GB".
//
// THAT BLOCKER WAS ABOUT CO-RESIDENCY, AND CO-RESIDENCY IS NOT REQUIRED. The reference and
// goinfer never have to be alive at the same instant: scripts/pin_qwen3next_real.py writes the
// bf16 logits to a JSON golden through accelerate's disk offload, and this test reads that file
// back in a separate process. Pinning offline is how every tiny golden in this repo is already
// made; the earlier slice route generalised the SLICE when what needed generalising was the
// PINNING.
//
// WHY THIS IS NOT COVERED BY qwen3_5_moe's T3, though the two share forward_qwen35.go. The
// shared-path proxy in docs/parity-coverage-policy.md needs the same forward file(s) AND the same
// deps_hash. qwen3_next has decoder/qwen3next.go of its own and a distinct hash, and that file is
// exactly where its three real deltas live — computed layer_types from full_attention_interval,
// flat partial_rotary_factor, and the fused in_proj_qkvz/in_proj_ba split. A proxy row would have
// asserted that qwen3_5_moe's oracle covered code qwen3_5_moe never executes.
//
// QUANT IS int4 BY CAPACITY, NOT BY CHOICE, and the number this produces has to be read knowing
// that. 80B at int8 is ~80 GB against 62 GB of RAM; int4 is ~40 GB and fits. There is no int8 run
// to fall back to on this box, so a low cosine here cannot be re-run at higher precision to
// separate quant noise from a defect — the per-layer SHAPE has to do that instead (smooth and
// non-monotonic is noise; a cliff is a defect, and a defect cannot recover). The decision rule
// was pre-registered in docs/queue-correctness.md G5 before this ran.
//
// The sparsity is the reason to expect trouble: 10 of 512 experts is 1.95% active, sparsest in the
// table by a factor of two. nemotron3nano at 6/128 (4.7%) measured 0.978 with int8 ACTIVATIONS
// against 0.9977 with f32 ones, and its recorded cause is the expert-flip cliff.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen3NextReal_oracle -v -timeout 120m
package decoder

import "testing"

func TestQwen3NextReal_oracle(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN3NEXT_HF")
	// int4 WEIGHTS, f32 ACTIVATIONS. Activations are deliberately NOT quantized: at 10/512
	// routing, perturbing the router is the one thing measured to break this model class, and
	// nemotron3nano already paid for that lesson at four times the density.
	realLogitOracleQuant(t, ckpt, "../testdata/qwen3next_real_golden.json", "qwen3_next", "qwen3_next",
		"HF bf16 (Qwen/Qwen3-Next-80B-A3B-Instruct, full model via accelerate disk offload; "+
			"int4 weights, f32 activations)", "int4")
}

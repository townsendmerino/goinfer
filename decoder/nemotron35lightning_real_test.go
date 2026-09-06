//go:build realckpt

// Real-checkpoint gate for NVIDIA-Nemotron-3.5-Lightning-30B-A3B (nemotron_h MoE) — F2 of
// docs/task-families-2026-09.md. NOT a new family: Phase 0 found this checkpoint's config.json
// identical to the already-T3'd Nemotron 3 Nano's (docs/completed/queue-correctness.md G4) in
// every architecturally meaningful field, including the exact 52-block layer pattern (23 mamba /
// 23 moe / 6 attention, same order). This gate exists to confirm the ACTUALLY TRAINED weights
// behave the way that identical architecture predicts — a tiny fixture cannot catch a wrong
// tensor name, a transposed expert stack, or a router bias read from the wrong key.
//
// This goes through the same realLogitOracleQuant helper every other real-checkpoint gate uses,
// so it DOES call emitParityRow like the others — but it is deliberately left OUT of
// cmd/gate/parity.go's emitGates list (the manifest's "nemotron_h" row is keyed by registry
// model_type, not by checkpoint, and is already `validated` from Nano's T3, cosine 0.997668;
// TestNemotron3NanoReal_oracle itself isn't in emitGates either — the row was populated once by a
// direct run + manual merge, not by the routine sweep). A `go run ./cmd/gate parity` sweep will
// still run this gate (it's in parityRealckptGates) and PASS/FAIL/skip on it, but won't touch the
// manifest; running it directly with GOINFER_MANIFEST_EMIT=1 would still emit a PARITY_ROW line,
// and merging that WOULD overwrite Nano's specific numbers with Lightning's — a deliberate choice
// for whoever runs it by hand, not something this gate silently does on a routine sweep. This
// run's result is recorded in docs/task-families-2026-09.md's F2 section as confirmatory
// evidence for the same family.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestNemotron35LightningReal -v -timeout 90m
package decoder

import "testing"

func TestNemotron35LightningReal_oracle(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_NEMOTRON35LIGHTNING_HF")
	// Same quant finding as Nano is expected to transfer (identical router shape, same 6-of-128
	// sparsity) but is MEASURED here independently, not assumed: int8 weights + f32 activations,
	// per Nano's own recorded sensitivity to int8 activations at this sparsity.
	realLogitOracleQuant(t, ckpt, "../testdata/nemotron35lightning_real_golden.json", "nemotron_h", "nemotron_h",
		"HF bf16 (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16; int8 weights, f32 activations)", "int8")
}

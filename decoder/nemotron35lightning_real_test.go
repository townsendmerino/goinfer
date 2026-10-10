//go:build realckpt

// Real-checkpoint gate for NVIDIA-Nemotron-3.5-Lightning-30B-A3B (nemotron_h MoE), F2 of
// docs/completed/task-families-2026-09.md. NOT a new family: its config.json is identical to the already-T3'd
// Nemotron 3 Nano's (docs/completed/queue-correctness.md G4) in every architecturally meaningful field, including the
// 52-block layer pattern. This gate confirms the ACTUALLY TRAINED weights behave as that identical architecture
// predicts; a tiny fixture cannot catch a wrong tensor name, a transposed expert stack or a router bias read from the
// wrong key.
//
// It goes through realLogitOracleQuant, so it calls emitParityRow, but it is deliberately OUT of cmd/gate/parity.go's
// emitGates list: the manifest's "nemotron_h" row is keyed by registry model_type, not by checkpoint, and is already
// `validated` from Nano's T3. A `go run ./cmd/gate parity` sweep still runs it (it is in parityRealckptGates) but does
// not touch the manifest; running it by hand with GOINFER_MANIFEST_EMIT=1 and merging the PARITY_ROW line WOULD
// overwrite Nano's numbers with Lightning's, which is a deliberate choice for whoever does it. The result is recorded in
// docs/completed/task-families-2026-09.md's F2 section as confirmatory evidence.
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

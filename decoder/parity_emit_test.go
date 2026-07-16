package decoder

// Canonical parity-row emitter (docs/task-parity-coverage.md Item 1d). The
// real-checkpoint gates (decoder/*real_test.go, qwen35 gates) that measure argmax/
// cosine against a real oracle call emitParityRow at the end of a passing run; the
// parity sweep (scripts/parity_sweep.sh EMIT_MANIFEST=1) captures the PARITY_ROW
// lines and merges them into testdata/parity_manifest.json (the -merge-rows mode in
// parity_manifest_test.go) — so validating a family records its metrics instead of a
// hand-edit. This file carries NO build tag so the realckpt gates can call it.
//
// Discipline: a row is only emitted by a gate that genuinely measured against an
// oracle, and only when that gate PASSED (t.Failed() guards it) — never a threshold,
// never a coherence-only gate. Off by default (no env, no asset) so plain `go test`
// and CI never touch the manifest.

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// emitParityRow writes one canonical manifest row to stdout for the sweep to merge —
// but ONLY when GOINFER_MANIFEST_EMIT is set AND the calling gate has not failed (so a
// numerics regression emits nothing). The line is `PARITY_ROW {compact-json}` with the
// family, method (full-forward-oracle | real-model-oracle | weightDiff | layer-slice),
// the oracle it validated against, and the measured metrics. validated_at/date/machine
// are NOT stamped here — the sweep's merge stamps those so every row from one run agrees.
// This emitter carries no build tag so the realckpt gates can call it, so it reads as
// unused in the default (no-realckpt) build:
//
//lint:ignore U1000 called only by the //go:build realckpt gates (phi3/deepseek/qwen35_real_test.go).
func emitParityRow(t *testing.T, family, method, reference string, argmaxPct, cosineMin, cosineMean float64) {
	t.Helper()
	if os.Getenv("GOINFER_MANIFEST_EMIT") == "" || t.Failed() {
		return
	}
	// Fixed-precision metrics keep the JSON readable + the merge deterministic (Go's
	// float marshaling would print 1.0 as "1"). Strings go through json.Marshal for
	// correct quoting/escaping.
	metrics := fmt.Sprintf(`{"argmax_pct":%.1f,"cosine_min":%.5f,"cosine_mean":%.5f}`, argmaxPct, cosineMin, cosineMean)
	fam, _ := json.Marshal(family)
	meth, _ := json.Marshal(method)
	ref, _ := json.Marshal(reference)
	fmt.Printf("PARITY_ROW {\"family\":%s,\"method\":%s,\"reference\":%s,\"metrics\":%s}\n", fam, meth, ref, metrics)
}

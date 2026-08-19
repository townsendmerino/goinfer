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
	"strings"
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
	// THE METHOD IS VOCABULARY, NOT FREE TEXT (B15). mellum's gate emitted "real-oracle" —
	// one word short of the T3 name "real-model-oracle" — and the merge wrote it into the
	// manifest verbatim, where it read as a method no tier rule recognises. A string typed
	// at ~20 call sites and never compared to a list is a defect waiting for a rename, so
	// it is checked HERE, at the source, rather than only by the tier gate downstream.
	if !knownParityMethod(method) {
		t.Fatalf("emitParityRow(%q): method %q is not in the manifest vocabulary %v — "+
			"a row with an unrecognised method corrupts the manifest and no tier rule can "+
			"classify it", family, method, sortedKeys(parityMethods))
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

// parityMethods is the closed vocabulary of manifest `method` values: the T3 methods that can
// carry status "validated" (t3Methods, in parity_manifest_test.go, which is the authority the
// tier gate reads) plus the sub-T3 methods that are honest evidence at status "experimental".
// Keeping BOTH sets in one place is the point — the emitter must accept a tiny-golden row (it
// is real evidence) while the merge must not let one become "validated".
var parityMethods = map[string]bool{
	"full-forward-oracle":  true, // T3
	"real-model-oracle":    true, // T3
	"weightDiff":           true, // T3
	"tiny-golden":          true, // sub-T3
	"tiny-golden+coherent": true, // sub-T3
	"layer-slice":          true, // sub-T3: real weights, partial depth
}

// knownParityMethod reports whether m is in the vocabulary. "shared-path (via X)" is accepted
// by prefix the same way isT3Method accepts it, since X is a family name.
func knownParityMethod(m string) bool {
	return parityMethods[m] || strings.HasPrefix(m, "shared-path (via ")
}

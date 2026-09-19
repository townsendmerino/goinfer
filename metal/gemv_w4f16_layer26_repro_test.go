//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestW4F16Lane_layer26Reproduction is the KEEPER reproducer for the open bug this brief's
// investigation (docs/measurements/w4f16-decode-investigation-2026-09-19.md) found: R1's W4F16
// decode lane (GOINFER_METAL_DECODE_LANE=w4f16) diverges catastrophically at layer 26 of
// qwen2.5-coder-1.5b's 28 — cosine goes NEGATIVE — even from a bit-for-bit CLEAN, unperturbed
// input taken from the shipped W4A8 path. This rules out the two hypotheses the investigation
// tested and falsified first (a "massive activation" precision-loss theory, and a model-level
// chaos/sensitivity-to-small-input-differences theory — see the record for both). The
// investigation localized the entry point to layer 26's FFN block specifically (attention there
// is fine, cosine 0.99999995) — most likely the gate/up GEMV (cosine 0.9976 in isolation, with at
// least one output row ~31% off) — but did not find the root mechanism before the investigation
// was parked. This test exists so whoever picks it up next starts from a known-good, minimal
// repro instead of rebuilding one.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestW4F16Lane_layer26Reproduction -v
func TestW4F16Lane_layer26Reproduction(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint twice)")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}

	// Ground truth: the shipped W4A8 path, greedy-decode-shaped (one token, position 0), through
	// layer 25. Captures x entering layer 26 and the final result after the whole stack.
	m1, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	rf1 := m1.ResidentForwardForTest().(*metalResident)
	r1 := rf1.r
	copy(r1.x.Floats(), m1.EmbedResidentForTest(785))
	r1.setPos(0)
	for l := 0; l < 26; l++ {
		e := r1.q.Begin()
		r1.encodeLayer(e, l)
		e.End()
	}
	x1 := append([]float32(nil), r1.x.Floats()...) // clean, ground-truth input to layer 26
	for l := 26; l < r1.nL; l++ {
		e := r1.q.Begin()
		r1.encodeLayer(e, l)
		e.End()
	}
	baseline := append([]float32(nil), r1.x.Floats()...)
	m1.Close()

	// The W4F16 lane, fed the EXACT SAME clean x1 at the layer-26 boundary (not its own,
	// independently-drifted layer-25 output) — isolates the kernels themselves from any
	// accumulated cross-layer drift.
	t.Setenv("GOINFER_METAL_DECODE_LANE", "w4f16")
	m2, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load2: %v", err)
	}
	defer m2.Close()
	rf2 := m2.ResidentForwardForTest().(*metalResident)
	r2 := rf2.r
	if !r2.canUseF16Lane(26) {
		t.Fatalf("layer 26 unexpectedly not eligible for the f16 lane — investigation's premise no longer holds, re-check")
	}
	copy(r2.x.Floats(), x1)
	r2.setPos(0)
	for l := 26; l < r2.nL; l++ {
		e := r2.q.Begin()
		r2.encodeLayer(e, l)
		e.End()
	}
	f16Result := append([]float32(nil), r2.x.Floats()...)

	var maxAbs float64
	var dot, m1n, m2n float64
	for i := range baseline {
		if d := math.Abs(float64(baseline[i] - f16Result[i])); d > maxAbs {
			maxAbs = d
		}
		dot += float64(baseline[i]) * float64(f16Result[i])
		m1n += float64(baseline[i]) * float64(baseline[i])
		m2n += float64(f16Result[i]) * float64(f16Result[i])
	}
	cos := dot / (math.Sqrt(m1n)*math.Sqrt(m2n) + 1e-30)
	t.Logf("layers 26-27 from a clean input: maxAbs=%.6g cosine=%.8f", maxAbs, cos)

	// The bug: this SHOULD read close to 1.0 (matching every earlier layer's ~0.99999+ fidelity)
	// and instead goes catastrophically negative. Failing on purpose, loudly, so this test
	// cannot silently start passing (e.g. from an unrelated refactor) without someone noticing —
	// see the investigation record for what "fixed" should look like (cosine back above ~0.999).
	if cos < 0.9 {
		t.Errorf("KNOWN OPEN BUG (see docs/measurements/w4f16-decode-investigation-2026-09-19.md): "+
			"cosine %.6f is catastrophically low from a CLEAN input at layers 26-27 — expected ~1.0. "+
			"Do not paper over this by loosening the threshold; find and fix the mechanism instead.", cos)
	}
}

//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoeSwigluWiring_C15 is the dedicated bug-B gate that the e2e MoE parity cosine can no longer
// serve (audit C-15): once cuda's f32tof16 matches the canonical cross-backend representation, the
// correct dispatch and a gate/up swap both land at ~0.9978 on mixtral-tiny's RANDOM experts, because
// silu(up)*gate ≈ silu(gate)*up in magnitude and the difference washes out through down-proj +
// combine + argmax. This test instead exercises launchGluSplit — the ONE place the gate/up split
// convention lives, shared by the routed/shared/gemma-4 dispatches — with CRAFTED gate≠up and reads
// back the pre-quant SwiGLU output, where a swap is unmistakable. Scale- and fixture-independent.
func TestMoeSwigluWiring_C15(t *testing.T) {
	const path = "../testdata/mixtral-tiny"
	requireDeviceAndFixture(t, path)
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		t.Skip("model did not go resident")
	}

	// Craft gate and up with DIFFERENT per-element patterns so silu(gate)*up is clearly distinct from
	// the swapped silu(up)*gate. n ≤ moeInter (128 for mixtral-tiny).
	const n = 64
	gate := make([]float32, n)
	up := make([]float32, n)
	for i := range gate {
		gate[i] = 0.5 + float32(i%7)*0.30 // 0.5..2.3
		up[i] = 2.0 - float32(i%5)*0.40   // 0.4..2.0, a different pattern
	}
	got, err := rf.MoeSwigluForTest(gate, up)
	if err != nil {
		t.Fatalf("MoeSwigluForTest: %v", err)
	}

	silu := func(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }
	var maxErr, swapDiff float64
	for i := range got {
		want := silu(gate[i]) * up[i] // gate-first (correct)
		swap := silu(up[i]) * gate[i] // gate/up swapped (bug B)
		if d := math.Abs(float64(got[i]) - float64(want)); d > maxErr {
			maxErr = d
		}
		if d := math.Abs(float64(want) - float64(swap)); d > swapDiff {
			swapDiff = d
		}
	}
	// The kernel uses __expf (fast intrinsic) vs Go's f64 exp, so allow a small elementwise tolerance.
	if maxErr > 5e-3 {
		t.Errorf("SwiGLU output != silu(gate)*up (maxErr %.4g) — gate/up wiring is broken (C-15 bug B: a gOff/uOff swap computes silu(up)*gate)", maxErr)
	}
	// Guard the test itself: the crafted input must make a swap genuinely detectable (not gate≈up).
	if swapDiff < 0.1 {
		t.Fatalf("crafted gate/up too similar (swap would differ by only %.4g) — this test could not detect a swap", swapDiff)
	}
	t.Logf("SwiGLU gate/up wiring OK: maxErr %.2e; a swap would perturb the output by up to %.3f", maxErr, swapDiff)
}

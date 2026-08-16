//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentHiddenCapture is the gate on the resident hidden-state seam (P10 increment 4,
// docs/spec/08): a resident target must be able to hand a block drafter the layer outputs it
// taps, and those must be the SAME hidden states the CPU seam produces.
//
// Why this test and not a unit check: the seam is only useful if a drafter can consume it, and
// the thing that would silently break it is not a crash but a wrong LAYER — an off-by-one in the
// tap convention, or capturing before the residual add instead of after. Both produce
// plausible-looking vectors. Only a cross-check against decoder.Model.ForwardCapture, which is
// itself gated against HF (TestDFlash_targetEndToEnd, 15/15 drafted ids), catches that.
//
// Tolerance: the resident path is int4 and the CPU reference here is int4 too, but they are
// different kernels, so this is a cosine gate rather than bit-equality — the same bar the wired
// gate uses for logits. A wrong tap layer reads as cosine ~0.0-0.5, not 0.99, so the floor
// separates the failure that matters by a wide margin.
func TestResidentHiddenCapture(t *testing.T) {
	requireHeavyModel(t)
	gguf := os.Getenv("GOINFER_CUDA_MODEL")
	if gguf == "" {
		gguf = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skip("no model")
	}

	mc, err := decoder.Load(gguf, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	if missing := mc.MissingResidentFeatures(decoder.ResidentBackendFeatures("cuda")); len(missing) > 0 {
		t.Skipf("model needs unimplemented feature(s) %v", missing)
	}
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("resident did not engage")
	}
	r, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("resident is %T, want *cudaResident", rf)
	}

	// Five taps spread over the stack, the shape DFlash/DSpark use (their Qwen3 configs tap
	// [1,9,17,25,33] of 36). Derived from the layer count so this runs on any test model.
	n := r.nLayers
	taps := []int{1, n / 4, n / 2, (3 * n) / 4, n - 3}
	for i := 1; i < len(taps); i++ { // keep strictly ascending on small models
		if taps[i] <= taps[i-1] {
			taps[i] = taps[i-1] + 1
		}
	}
	if taps[len(taps)-1] >= n {
		t.Skipf("model too shallow (%d layers) for a 5-tap probe", n)
	}
	if err := r.SetHiddenCapture(taps); err != nil {
		t.Fatalf("SetHiddenCapture(%v): %v", taps, err)
	}
	t.Logf("taps %v of %d layers", taps, n)

	// The seam must be OFF by default and cost nothing until armed — assert the disarm path
	// too, since a seam that cannot be turned off is a decode-time tax on every other family.
	defer func() {
		if err := r.SetHiddenCapture(nil); err != nil {
			t.Errorf("disarm: %v", err)
		}
		if got := r.HiddenCapture(); got != nil {
			t.Errorf("HiddenCapture after disarm = %v, want nil", got)
		}
	}()

	mcpu, err := decoder.Load(gguf, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	defer mcpu.Close()

	prompt := []int{785, 6722, 315, 9625, 374} // "The capital of France is"
	cache := mcpu.NewCache(len(prompt) + 2)
	for i, tok := range prompt {
		_, wantHidden, err := mcpu.ForwardCapture(tok, cache, taps)
		if err != nil {
			t.Fatalf("cpu ForwardCapture: %v", err)
		}
		emb := mc.EmbedResidentForTest(tok)
		if _, err := rf.Forward(emb, i); err != nil {
			t.Fatalf("resident forward: %v", err)
		}
		got := r.HiddenCapture()
		if len(got) != len(taps) {
			t.Fatalf("pos %d: captured %d rows, want %d", i, len(got), len(taps))
		}
		for s, tap := range taps {
			if len(got[s]) != len(wantHidden[s]) {
				t.Fatalf("pos %d tap %d: %d wide, want %d", i, tap, len(got[s]), len(wantHidden[s]))
			}
			cos := hidCapCosine(got[s], wantHidden[s])
			if i == len(prompt)-1 {
				t.Logf("pos %d layer %2d: cosine %.6f", i, tap, cos)
			}
			if cos < 0.99 {
				t.Errorf("pos %d layer %d: resident capture vs CPU cosine %.6f < 0.99 — wrong layer, or captured before the residual add",
					i, tap, cos)
			}
		}
	}
}

func hidCapCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

package decoder

import (
	"slices"
	"testing"
)

// TestForwardCaptureSeam gates the 05 hidden-state seam (increment 1): ForwardCapture
// must (a) leave the token output byte-identical to plain forward (read-only), and
// (b) return the actual residual stream — verified by feeding the last layer's capture
// back through final-norm + lm-head and reproducing the logits exactly. The low/mid
// captures use the same copy path, so the last-layer exactness covers them.
func TestForwardCaptureSeam(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	L, hidden := m.w.arch.NumLayers, m.w.arch.HiddenDim
	layers := []int{1, L / 2, L - 1} // EAGLE-3-style low / mid / high
	prompt := specPrompts[0]

	// Reference: plain forward over the prompt; keep the last token's logits.
	cref := m.NewCache(len(prompt) + 4)
	var refLogits []float32
	for _, id := range prompt {
		lg, ferr := m.forward(id, cref)
		if ferr != nil {
			t.Fatalf("forward: %v", ferr)
		}
		refLogits = slices.Clone(lg) // forward reuses scr.logits — clone
	}

	// Capture run: same prompt on a fresh cache; capture on the final token.
	ccap := m.NewCache(len(prompt) + 4)
	var capLogits []float32
	var hid [][]float32
	for i, id := range prompt {
		if i < len(prompt)-1 {
			if _, ferr := m.forward(id, ccap); ferr != nil {
				t.Fatalf("forward: %v", ferr)
			}
			continue
		}
		lg, h, cerr := m.ForwardCapture(id, ccap, layers)
		if cerr != nil {
			t.Fatalf("ForwardCapture: %v", cerr)
		}
		capLogits, hid = slices.Clone(lg), h
	}

	// (a) read-only: capture must not perturb the token output.
	if !slices.Equal(capLogits, refLogits) {
		t.Fatalf("ForwardCapture changed the logits (capture not read-only)")
	}
	// shapes
	if len(hid) != len(layers) {
		t.Fatalf("captured %d layers, want %d", len(hid), len(layers))
	}
	for i, h := range hid {
		if len(h) != hidden {
			t.Fatalf("capture[%d] len %d, want hidden %d", i, len(h), hidden)
		}
	}
	// (b) the high (last-layer) capture, through final-norm + lm-head, == the logits.
	recomp := slices.Clone(m.logitsFromHidden(slices.Clone(hid[len(hid)-1]), ccap))
	if !slices.Equal(recomp, capLogits) {
		var maxAbs float32
		for i := range recomp {
			if d := recomp[i] - capLogits[i]; d > maxAbs || -d > maxAbs {
				maxAbs = d
			}
		}
		t.Fatalf("last-layer capture != residual stream (final-norm+head maxAbs %.3e)", maxAbs)
	}
	t.Logf("hidden-state seam OK: captured layers %v (hidden %d), read-only + last-layer exact", layers, hidden)
}

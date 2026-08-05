package decoder

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestValidateShapes_catchesArchMismatch is the break-it-first gate for audit C-06:
// a .giw whose tensor rows disagree with the architecture must be REJECTED, because
// the forward writes tensor.Rows() floats into a config-sized scratch slice (moeMLP's
// make([]float32, NumExperts); the qDim/kvDim/vocab decodeScratch buffers) — a heap
// write past the slice from caller-supplied bytes.
func TestValidateShapes_catchesArchMismatch(t *testing.T) {
	const (
		vocab, hidden, nH, nKV, hd, nE = 32, 8, 2, 1, 4, 4
	)
	arch := &Architecture{
		VocabSize: vocab, HiddenDim: hidden, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd,
		MoE: &MoEConfig{NumExperts: nE, TopK: 2},
	}
	f := func(rows, cols int) linalg.WeightMat {
		return linalg.WrapF32(make([]float32, rows*cols), rows, cols)
	}
	good := func() *Weights {
		return &Weights{
			arch:   arch,
			Embed:  f(vocab, hidden),
			LMHead: f(vocab, hidden),
			Layers: []LayerWeights{{
				QProj: f(nH*hd, hidden), KProj: f(nKV*hd, hidden), VProj: f(nKV*hd, hidden),
				DownProj: f(hidden, hidden), Router: f(nE, hidden),
			}},
		}
	}
	// The honest bundle passes.
	if e := validateShapes(good(), arch); e != nil {
		t.Fatalf("valid bundle rejected: %v", e)
	}
	// Each tampered dimension must be caught.
	for _, tc := range []struct {
		name   string
		mutate func(*Weights)
		want   string
	}{
		{"router rows = NumExperts+K", func(w *Weights) { w.Layers[0].Router = f(nE+2, hidden) }, "Router"},
		{"embed vocab wrong", func(w *Weights) { w.Embed = f(vocab+1, hidden) }, "Embed"},
		{"lmhead vocab wrong", func(w *Weights) { w.LMHead = f(vocab*2, hidden) }, "LMHead"},
		{"qproj rows wrong", func(w *Weights) { w.Layers[0].QProj = f(nH*hd+1, hidden) }, "QProj"},
		{"kproj rows wrong", func(w *Weights) { w.Layers[0].KProj = f(nKV*hd+3, hidden) }, "KProj"},
	} {
		w := good()
		tc.mutate(w)
		e := validateShapes(w, arch)
		if e == nil {
			t.Errorf("%s: not rejected", tc.name)
			continue
		}
		if !strings.Contains(e.Error(), tc.want) {
			t.Errorf("%s: error %q does not name %q", tc.name, e.Error(), tc.want)
		}
	}
}

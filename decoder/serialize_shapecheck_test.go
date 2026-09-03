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
		vocab, hidden, nH, nKV, hd, nE, inter, moeInter = 32, 8, 2, 1, 4, 4, 16, 12
	)
	arch := &Architecture{
		VocabSize: vocab, HiddenDim: hidden, NumHeads: nH, NumKVHeads: nKV, HeadDim: hd,
		IntermediateDim: inter,
		NumLayers:       1, // one layer in the bundle below; the C-06 layer-count check needs them to agree
		MoE:             &MoEConfig{NumExperts: nE, TopK: 2, IntermediateDim: moeInter},
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
		{"oproj rows wrong", func(w *Weights) { w.Layers[0].OProj = f(hidden+1, nH*hd) }, "OProj"},
		{"gateproj rows wrong", func(w *Weights) { w.Layers[0].GateProj = f(inter+1, hidden) }, "GateProj"},
		{"upproj rows wrong", func(w *Weights) { w.Layers[0].UpProj = f(inter*2, hidden) }, "UpProj"},
		{"expert gate rows wrong", func(w *Weights) {
			w.Layers[0].Experts = []expertWeights{{Gate: f(moeInter+1, hidden)}}
		}, "expert 0 Gate"},
		{"expert down rows wrong", func(w *Weights) {
			w.Layers[0].Experts = []expertWeights{{Down: f(hidden+1, moeInter)}}
		}, "expert 0 Down"},
		{"too few layers (OOB read)", func(w *Weights) { w.Layers = nil }, "layer count"},
		{"too many layers", func(w *Weights) { w.Layers = append(w.Layers, LayerWeights{}) }, "layer count"},
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

// TestValidateShapes_gemma4PLE is a synthetic gate for the model-level Per-Layer-Embedding tail:
// PerLayerModelProj and PerLayerProjNorm were checked against the wrong axis (HiddenDim instead
// of NumLayers*HiddenSizePerLayerInput) and only a real gemma4-E2B checkpoint — a heavy,
// asset-gated test — ever exercised the branch, so both bugs shipped unnoticed. This runs on
// every invocation with no checkpoint required.
func TestValidateShapes_gemma4PLE(t *testing.T) {
	const hidden, pleDim, nLayers = 16, 4, 2
	pleTotal := nLayers * pleDim
	arch := &Architecture{
		HiddenDim: hidden,
		NumLayers: nLayers,
		gemma4:    &gemma4Params{HiddenSizePerLayerInput: pleDim},
	}
	f := func(rows, cols int) linalg.WeightMat {
		return linalg.WrapF32(make([]float32, rows*cols), rows, cols)
	}
	good := func() *Weights {
		return &Weights{
			arch:               arch,
			Layers:             make([]LayerWeights, nLayers),
			PerLayerTokenEmbed: f(1, pleTotal), // only Rows()>0 gates the branch; shape unchecked
			PerLayerModelProj:  f(pleTotal, hidden),
			PerLayerProjNorm:   make([]float32, pleDim),
		}
	}
	if e := validateShapes(good(), arch); e != nil {
		t.Fatalf("valid gemma4 PLE bundle rejected: %v", e)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Weights)
		want   string
	}{
		{"PerLayerModelProj rows swapped with hidden", func(w *Weights) { w.PerLayerModelProj = f(hidden, pleTotal) }, "PerLayerModelProj"},
		{"PerLayerModelProj cols wrong", func(w *Weights) { w.PerLayerModelProj = f(pleTotal, hidden+1) }, "PerLayerModelProj"},
		{"PerLayerProjNorm sized to HiddenDim instead of pleDim", func(w *Weights) { w.PerLayerProjNorm = make([]float32, hidden) }, "PerLayerProjNorm"},
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

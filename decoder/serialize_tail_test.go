package decoder

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// mat builds a zeroed [rows, cols] f32 weight — WrapF32 requires the exact backing length.
func mat(rows, cols int) linalg.WeightMat {
	return linalg.WrapF32(make([]float32, rows*cols), rows, cols)
}

// M-11: validateShapes did not cover the v6 completeness tail, and there is a REAL trigger
// rather than a hypothetical one.
//
// A gpt-oss sidecar .giw written before v6 is sink-free — the R3 "bundle LOADED CLEAN"
// artifact. It is still within giwMinReadV=3, still judged "fresh" by mtime (prequant compares
// timestamps, which cannot see a missing tensor), reads AttnSinks as nil, passes
// validateShapes, and then panics at forward_gptoss.go's `lw.AttnSinks[qh]` on the first
// request — in the decode goroutine, where no handler recovers it.
//
// `vec`'s "got == 0 means absent, and absent is allowed" is correct for a family that omits a
// field and exactly wrong for one whose forward dereferences it unconditionally. `req` is that
// distinction, and this pins it.
func TestValidateShapes_gptOssTailIsRequired(t *testing.T) {
	// A minimal gpt-oss-shaped arch and bundle. Built by hand rather than loaded, because the
	// defect is a bundle that CANNOT be produced by the current writer — it is what an older
	// one left behind.
	arch := &Architecture{
		Name: "gpt_oss", Norm: NormRMS, NormEps: 1e-5, AttnScale: 0.25,
		HiddenDim: 8, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 2, VocabSize: 16,
		gptoss: &gptOssParams{},
		MoE:    &MoEConfig{NumExperts: 1, TopK: 1, IntermediateDim: 4},
	}
	full := func() *Weights {
		return &Weights{Layers: []LayerWeights{{
			AttnSinks: make([]float32, arch.NumHeads),
			Experts: []expertWeights{{
				Gate: mat(arch.MoE.IntermediateDim, arch.HiddenDim),
				Up:   mat(arch.MoE.IntermediateDim, arch.HiddenDim),
				Down: mat(arch.HiddenDim, arch.MoE.IntermediateDim),

				GateBias: make([]float32, arch.MoE.IntermediateDim),
				UpBias:   make([]float32, arch.MoE.IntermediateDim),
				DownBias: make([]float32, arch.HiddenDim),
			}},
		}}}
	}

	// The premise: a COMPLETE bundle passes. Without this every case below could pass for an
	// unrelated reason and the test would assert nothing about the tail.
	if e := validateShapes(full(), arch); e != nil {
		t.Fatalf("a complete gpt-oss bundle is rejected: %v — every case below would pass for "+
			"the wrong reason", e)
	}

	for name, tc := range map[string]struct {
		break_ func(*Weights)
		names  string
	}{
		// THE REAL ONE: exactly what a pre-v6 sidecar hands back.
		"AttnSinks absent (a pre-v6 sidecar)": {
			func(w *Weights) { w.Layers[0].AttnSinks = nil }, "AttnSinks"},
		"AttnSinks short": {
			func(w *Weights) { w.Layers[0].AttnSinks = make([]float32, 1) }, "AttnSinks"},
		"expert GateBias absent": {
			func(w *Weights) { w.Layers[0].Experts[0].GateBias = nil }, "GateBias"},
		"expert UpBias absent": {
			func(w *Weights) { w.Layers[0].Experts[0].UpBias = nil }, "UpBias"},
		"expert DownBias absent": {
			func(w *Weights) { w.Layers[0].Experts[0].DownBias = nil }, "DownBias"},
		"expert DownBias short": {
			func(w *Weights) { w.Layers[0].Experts[0].DownBias = make([]float32, 1) }, "DownBias"},
	} {
		t.Run(name, func(t *testing.T) {
			w := full()
			tc.break_(w)
			e := validateShapes(w, arch)
			if e == nil {
				t.Fatal("accepted; the bundle loads clean and panics at the first forward (M-11)")
			}
			if !strings.Contains(e.Error(), tc.names) {
				t.Errorf("error %q does not name %q", e, tc.names)
			}
		})
	}

	// NOT required for other families — the same fields are legitimately nil everywhere else,
	// and a fix that demanded them universally would refuse every non-gpt-oss bundle.
	t.Run("absent is fine without the gptoss marker", func(t *testing.T) {
		plain := *arch
		plain.gptoss = nil
		w := full()
		w.Layers[0].AttnSinks = nil
		w.Layers[0].Experts[0].GateBias = nil
		w.Layers[0].Experts[0].UpBias = nil
		w.Layers[0].Experts[0].DownBias = nil
		if e := validateShapes(w, &plain); e != nil {
			t.Errorf("a non-gpt-oss bundle is rejected for missing gpt-oss fields: %v", e)
		}
	})
}

//go:build gpu

package kernels

import (
	"strings"
	"testing"
)

// TestNewDecodeRunner_layerNormOnlyWhereItReaches: a LayerNorm model takes LayerNormQuantWGSL at every norm rmsQuant builds
// (the pre-norms and the final norm); every other norm site binds an RMSNorm kernel. Each such site, set on a LayerNorm
// plan, must be refused before anything is built.
func TestNewDecodeRunner_layerNormOnlyWhereItReaches(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer c.Close()
	w := c.zbuf(64)
	for name, l := range map[string]RunLayer{
		"QK-norm (q)":    {QNorm: w},
		"QK-norm (k)":    {KNorm: w},
		"V-norm":         {VNorm: true},
		"K=V":            {GKEqV: true},
		"post-attn norm": {PostAttnNorm: w},
		"post-MLP norm":  {PostMLPNorm: w},
		"post-PLE norm":  {PostPLENorm: w},
	} {
		m := RunModel{LmHead: &ResidentW8A8{}, LayerNorm: true, Layers: []RunLayer{l}}
		r, err := c.newDecodeRunner(m, 64, 2, 2, 32, 128, 0, 1e-5, 1, false)
		if err == nil {
			r.release()
		}
		if err == nil || !strings.Contains(err.Error(), "would run as RMSNorm") {
			t.Errorf("%s on a LayerNorm model: err %v, want the refusal naming RMSNorm", name, err)
		}
	}
	whole := RunModel{LmHead: &ResidentW8A8{}, LayerNorm: true, QKNormWhole: true, Layers: []RunLayer{{}}}
	r, err := c.newDecodeRunner(whole, 64, 2, 2, 32, 128, 0, 1e-5, 1, false)
	if err == nil {
		r.release()
	}
	if err == nil || !strings.Contains(err.Error(), "would run as RMSNorm") {
		t.Errorf("whole-vector QK-norm on a LayerNorm model: err %v, want the refusal naming RMSNorm", err)
	}
}

// TestRunModelToPrefill_declinesLayerNormAndParallel: a plain dense RunModel takes the batched pass, and the same model with
// only RunModel.LayerNorm, or only RunModel.Parallel, set does not, since the pass runs RMSNorm and the sequential block.
// Each flag is set alone, so neither rides another decline (Pairwise, on every real model that sets them). No device:
// RunModelToPrefill reads the plan's fields only.
func TestRunModelToPrefill_declinesLayerNormAndParallel(t *testing.T) {
	base := RunModel{LmHead: &ResidentW8A8{}}
	if _, _, ok := RunModelToPrefill(&base, 64); !ok {
		t.Fatal("the plain dense baseline declined: the test cannot isolate LayerNorm or Parallel")
	}
	ln, par := base, base
	ln.LayerNorm, par.Parallel = true, true
	if _, _, ok := RunModelToPrefill(&ln, 64); ok {
		t.Error("RunModelToPrefill admitted a LayerNorm model: the batched pass would run RMSNorm")
	}
	if _, _, ok := RunModelToPrefill(&par, 64); ok {
		t.Error("RunModelToPrefill admitted a parallel-block model: the batched pass would run the block sequentially")
	}
}

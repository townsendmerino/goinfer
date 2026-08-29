package decoder

import (
	"os"
	"testing"
)

// TestMellum2SliceShape reports which FFN path the sliced checkpoint actually
// takes. The prefill profile showed NO moeMLP frames, which has two very
// different explanations — the MoE work is real but attributed to anonymous
// worker goroutines, or the experts did not load and the DENSE FFN ran instead.
// Mellum2 makes those hard to tell apart by cost alone: top-k 8 x
// moe_intermediate 896 = 7168 = intermediate_size exactly, so the MoE and dense
// paths have IDENTICAL MAC counts and differ only in batching efficiency.
func TestMellum2SliceShape(t *testing.T) {
	path := os.Getenv("GOINFER_MELLUM_CKPT")
	if path == "" {
		t.Skip("set GOINFER_MELLUM_CKPT")
	}
	requireHeavyModel(t)
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	a := m.w.arch
	t.Logf("arch=%s layers=%d MoE=%v", a.Name, len(m.w.Layers), a.MoE != nil)
	if a.MoE != nil {
		t.Logf("MoE: experts=%d topk=%d interDim=%d sharedInterDim=%d",
			a.MoE.NumExperts, a.MoE.TopK, a.MoE.IntermediateDim, a.MoE.SharedIntermediateDim)
	}
	for i := range m.w.Layers {
		lw := &m.w.Layers[i]
		t.Logf("layer %d: Experts=%d Router=%v denseGateProj=%d denseUpProj=%d",
			i, len(lw.Experts), lw.Router.Rows() != 0, lw.GateProj.Rows(), lw.UpProj.Rows())
	}
}

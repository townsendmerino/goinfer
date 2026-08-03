//go:build cuda

package cuda

import "testing"

// TestGemma4Graphs_locateDivergence runs ONE forward pass at a fixed position with per-layer residual
// snapshots, graphs-on vs graphs-off on the same model, and reports the FIRST layer whose residual
// r.x differs. Run under external GPU churn (where the full forward diverges). It answers the one
// question the micro-tests can't: does the divergence appear at a SPECIFIC layer immediately (a bad
// captured graph / a repeatable boundary) or ACCUMULATE over layers (drift)? Immediate-at-layer-k
// points to a fixable capture defect in layer k's segments; smooth accumulation from layer 0 points to
// a distributed effect (contention perturbing every layer's replay slightly).
func TestGemma4Graphs_locateDivergence(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoEGraphs(t, dir, true, false)
	defer mc.Close()
	r := rf.(*cudaResident)
	r.layerCap = true

	emb := mc.EmbedResidentForTest(42)

	run := func(useGraphs bool) [][]float32 {
		r.graphs = useGraphs
		r.layerCapBuf = nil
		if _, err := rf.Forward(emb, 0); err != nil {
			t.Fatalf("forward (graphs=%v): %v", useGraphs, err)
		}
		out := make([][]float32, len(r.layerCapBuf))
		copy(out, r.layerCapBuf)
		return out
	}

	graphed := run(true)
	live := run(false)

	firstDiv := -1
	for l := 0; l < len(live) && l < len(graphed); l++ {
		maxAbs, n := 0.0, 0
		for i := range live[l] {
			if live[l][i] != graphed[l][i] {
				n++
				d := float64(live[l][i] - graphed[l][i])
				if d < 0 {
					d = -d
				}
				if d > maxAbs {
					maxAbs = d
				}
			}
		}
		mark := ""
		if n > 0 && firstDiv < 0 {
			firstDiv = l
			mark = "  <-- FIRST DIVERGENCE"
		}
		t.Logf("layer %2d: %d/%d differ, maxAbs=%.6g%s", l, n, len(live[l]), maxAbs, mark)
	}
	if firstDiv < 0 {
		t.Logf("no per-layer divergence (isolation, or churn not active)")
	} else {
		t.Logf("FIRST DIVERGENCE at layer %d of %d", firstDiv, len(live))
	}
}

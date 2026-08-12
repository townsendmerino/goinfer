//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestCUDA_graphSegAIsolated captures the REAL segA (a heterogeneous RAW chain: rms writes the
// quantized activation aq/aSc, then the Q/K/V GEMVs read aq/aSc — genuine inter-kernel data
// dependencies between DIFFERENT kernels) and compares graph replay vs live issue for ONE layer, in
// isolation (a single executor round-trip, not the full 48-layer forward).
//
// This is the minimal-unit probe. Earlier clean-under-churn chains were either the SAME kernel
// repeated (aikit vadd dependent chain) or INDEPENDENT same-kernel launches (fRms/fScaleVec) — neither
// exercises a heterogeneous dependent chain, the one structure segA has that they don't. If segA
// diverges here under concurrent GPU load, the bug reproduces without the full forward and this is
// where to dump the captured DAG (cuGraphGetNodes/GetEdges). If it stays bit-exact, the divergence
// needs the full-forward scale/interleaving and the minimal unit is elsewhere.
//
// Run under external GPU churn (another process). In isolation it passes; the diagnostic is under load.
func TestCUDA_graphSegAIsolated(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoECache(t, dir, false)
	defer mc.Close()
	r := rf.(*cudaResident)
	Ly := &r.layers[0]

	seed := make([]float32, r.hidden)
	for i := range seed {
		seed[i] = float32(i%13) - 6
	}
	// segA reads r.x (+ per-layer weights) and writes r.qB/r.kB/r.vB. Compare qB — the query rows the
	// downstream rope/attention consume.
	readQB := func() []float32 {
		h := make([]float32, Ly.qDim)
		_ = gpu.Download(r.qB, h)
		return h
	}
	chain := func() error { return r.segA(Ly, 0) }

	// A missing capture edge is exposed only *probabilistically* per replay, so one replay can pass by
	// luck. Replay a single captured segA REPS times under churn (the full forward does ~1920/run) and
	// check every replay against the live reference — volume is what turns a latent missing edge into a
	// near-certain hit.
	const REPS = 2000
	badReplays, firstMaxAbs := 0, 0.0
	err := r.do(func() error {
		if e := gpu.Upload(r.x, seed); e != nil {
			return e
		}
		if e := chain(); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		live := readQB()

		g, e := r.stream.Capture(chain)
		if e != nil {
			return e
		}
		defer g.Close()
		for range REPS {
			if e := gpu.Upload(r.x, seed); e != nil { // reset the input each replay
				return e
			}
			if e := g.Replay(); e != nil {
				return e
			}
			if e := r.stream.Sync(); e != nil {
				return e
			}
			got := readQB()
			diff := false
			for i := range live {
				if live[i] != got[i] {
					diff = true
					d := float64(live[i] - got[i])
					if d < 0 {
						d = -d
					}
					if badReplays == 0 && d > firstMaxAbs {
						firstMaxAbs = d
					}
				}
			}
			if diff {
				badReplays++
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("segA isolate: %v", err)
	}
	t.Logf("segA isolated ×%d replays (qDim=%d): %d divergent replays, first maxAbsDiff=%.6g", REPS, Ly.qDim, badReplays, firstMaxAbs)
	if badReplays != 0 {
		t.Errorf("segA graph replay diverged from live in ISOLATION on %d/%d replays (maxAbs=%.6g) — capture has fewer edges than the real deps", badReplays, REPS, firstMaxAbs)
	}
}

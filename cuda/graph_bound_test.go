//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestCUDA_graphReplayBound is the fail-fast in front of the CUDA-graphs forward restructure
// (Step 2). launch_cost bounded a SINGLE live launch at ~10 µs (FFI/purego-bound, grid-independent).
// It did NOT measure graph replay — and the whole lever rests on one unproven claim: that replaying a
// captured K-kernel segment collapses K host crossings into ~one, rather than still paying per-node
// GPU-side dispatch K times. If replay ≈ live at real segment size, the restructure buys nothing and
// we bank a negative BEFORE the invasive launchToken surgery.
//
// It captures a representative static chain (K launches of a trivial elementwise kernel — the crossing
// cost is kernel-independent, same basis launch_cost used), then compares:
//   - live:  issue K launches per iteration, one sync at the end, ÷N
//   - graph: capture the K-chain once, Replay() per iteration, one sync at the end, ÷N
//
// and spot-checks that a captured chain is BIT-EXACT to the live chain (the numerical foundation the
// real bit-exact gate builds on). Model-independent, so it runs on the tiny fixture.
func TestCUDA_graphReplayBound(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoECache(t, dir, false)
	defer mc.Close()
	r := rf.(*cudaResident)

	// K ≈ one static segment's launch count. A real Gemma-4 MoE layer has ~47 static launches split
	// across 3 segments (A/B/C) ≈ ~15 each; 16 is representative of the per-segment collapse.
	const K = 16
	const N = 20000
	cfg := LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	// s^K stays ~1 so the spot-check values don't under/overflow; the exact factor is irrelevant —
	// live and graph apply the identical op sequence, so bit-exact equality is the claim under test.
	const s = float32(0.9999)
	chain := func() error {
		for range K {
			if e := r.launch(r.fScaleVec, cfg, Arg(r.x), gpu.ArgValue(s), gpu.ArgValue(int32(r.hidden))); e != nil {
				return e
			}
		}
		return nil
	}

	var liveUs, graphUs float64
	var liveOut, graphOut []float32
	seed := make([]float32, r.hidden)
	for i := range seed {
		seed[i] = float32(i%7) - 3 // deterministic, non-trivial
	}

	err := r.do(func() error {
		// --- bit-exact spot check: same seed, live chain vs captured-then-replayed chain ---
		if e := gpu.Upload(r.x, seed); e != nil {
			return e
		}
		if e := chain(); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		liveOut = make([]float32, r.hidden)
		if e := gpu.Download(r.x, liveOut); e != nil {
			return e
		}

		g, e := r.stream.Capture(chain)
		if e != nil {
			return e
		}
		defer g.Close()
		if e := gpu.Upload(r.x, seed); e != nil { // reset to the SAME seed
			return e
		}
		if e := g.Replay(); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		graphOut = make([]float32, r.hidden)
		if e := gpu.Download(r.x, graphOut); e != nil {
			return e
		}

		// --- timing: live K-launch chain vs single graph Replay, per iteration ---
		for range 200 { // warm
			_ = chain()
		}
		_ = r.stream.Sync()
		start := time.Now()
		for range N {
			_ = chain()
		}
		_ = r.stream.Sync()
		liveUs = time.Since(start).Seconds() * 1e6 / float64(N)

		for range 200 { // warm
			_ = g.Replay()
		}
		_ = r.stream.Sync()
		start = time.Now()
		for range N {
			_ = g.Replay()
		}
		_ = r.stream.Sync()
		graphUs = time.Since(start).Seconds() * 1e6 / float64(N)
		return nil
	})
	if err != nil {
		t.Fatalf("bound: %v", err)
	}

	// Bit-exact: a captured chain must equal the live chain element-for-element.
	for i := range liveOut {
		if liveOut[i] != graphOut[i] {
			t.Fatalf("graph replay NOT bit-exact to live at [%d]: live=%v graph=%v", i, liveOut[i], graphOut[i])
		}
	}

	perLaunchLive := liveUs / K
	savementPerSeg := liveUs - graphUs
	t.Logf("K=%d launches/segment", K)
	t.Logf("  live chain:   %.2f µs/iter  (%.2f µs/launch)", liveUs, perLaunchLive)
	t.Logf("  graph replay: %.2f µs/iter  (%.2f× cheaper than live)", graphUs, liveUs/graphUs)
	t.Logf("  saved/segment: %.2f µs", savementPerSeg)

	// Project to a 26B-shaped decode: ~48 MoE layers × 3 static segments = ~144 replays/token.
	// Each replay saves (liveUs - graphUs) over live issue; the per-token wall-clock delta is the
	// share of the ~59 ms/token budget this lever can actually reclaim.
	const segsPerToken = 48 * 3
	projMs := savementPerSeg * float64(segsPerToken) / 1000.0
	t.Logf("  projection (48 layers × 3 segments = %d replays/token): ~%.1f ms/token reclaimed", segsPerToken, projMs)
	switch {
	case liveUs/graphUs >= 3:
		t.Logf("  VERDICT: replay is %.1f× cheaper — the crossing collapse is REAL. The restructure is justified; "+
			"proceed to launchToken segmentation. Projected ~%.0f ms/token off a ~59 ms budget ≈ %.2f× speedup.",
			liveUs/graphUs, projMs, 59.0/(59.0-projMs))
	case liveUs/graphUs >= 1.5:
		t.Logf("  VERDICT: replay %.1f× cheaper — MODEST. Proceed only if the segmentation overhead (below the "+
			"launch count) stays small; the margin is thin.", liveUs/graphUs)
	default:
		t.Logf("  VERDICT: replay ~= live (%.2f×) — graph replay does NOT collapse the crossings at this segment "+
			"size (per-node GPU dispatch dominates). BANK NEGATIVE: do not restructure launchToken.", liveUs/graphUs)
	}
}

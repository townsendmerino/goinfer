//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestCUDA_graphLiveNoSyncOrdering is the VALID inter-operation ordering test — no sync between the
// interleaved ops, which is the regime the full forward runs in and the one my earlier volume tests
// wrongly serialized away (a trailing per-iteration Sync hides an inter-op race exactly like
// CUDA_LAUNCH_BLOCKING does; the per-layer-drain probe made the forward divergence vanish, proving the
// race is inter-operation and sync-maskable).
//
// It builds a NON-COMMUTATIVE chain on r.x, alternating a LIVE launch and a GRAPH REPLAY with NO sync
// between them, and one sync only at the very end:
//
//	live:   x *= 2      (fScaleVec)
//	graph:  x += 1      (captured fRes(x, ones))
//
// From x=0 the exact result after N steps is 2^N - 1 (f32-exact for N<=23). If cuGraphLaunch and
// cuLaunchKernel fail to serialize on the one stream under contention, the ×2 and +1 interleave wrong
// and the final value diverges from 2^N-1. Multiplication/addition don't commute, so an ordering slip
// is visible; a correctly-ordered stream is exact regardless of contention.
//
// Run under external GPU churn. This is the primitive-level analogue of the forward's
// replay<->live gaps, with the sync removed so the race is actually observable.
func TestCUDA_graphLiveNoSyncOrdering(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoECache(t, dir, false)
	defer mc.Close()
	r := rf.(*cudaResident)

	n := r.hidden
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	const N = 20 // 2^20-1 = 1048575, exact in f32
	want := float32((int64(1) << N) - 1)
	cfg := LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}

	const CHAINS = 400
	bad, firstGot := 0, float32(0)
	err := r.do(func() error {
		oneBuf := r.af(n)
		if e := gpu.Upload(oneBuf, ones); e != nil {
			return e
		}
		// graph: x += 1 (fRes adds oneBuf into x)
		g, e := r.stream.Capture(func() error {
			return r.launch(r.fRes, g1cfg(n, 256), Arg(r.x), Arg(oneBuf), gpu.ArgValue(int32(n)))
		})
		if e != nil {
			return e
		}
		defer g.Close()

		zero := make([]float32, n)
		for c := 0; c < CHAINS; c++ {
			if e := gpu.Upload(r.x, zero); e != nil {
				return e
			}
			// N steps, NO sync between ops on the stream
			for step := 0; step < N; step++ {
				if e := r.launch(r.fScaleVec, cfg, Arg(r.x), gpu.ArgValue(float32(2)), gpu.ArgValue(int32(n))); e != nil { // live x*=2
					return e
				}
				if e := g.Replay(); e != nil { // graph x+=1
					return e
				}
			}
			if e := r.stream.Sync(); e != nil { // single sync at the very end
				return e
			}
			got := make([]float32, n)
			if e := gpu.Download(r.x, got); e != nil {
				return e
			}
			if got[0] != want {
				if bad == 0 {
					firstGot = got[0]
				}
				bad++
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("nosync ordering: %v", err)
	}
	t.Logf("live<->graph NO-SYNC chains ×%d (N=%d steps, want %.0f): %d divergent, first got=%.0f", CHAINS, N, want, bad, firstGot)
	if bad != 0 {
		t.Errorf("%d/%d chains diverged (first got %.0f, want %.0f) — live<->graph ops not serialized on the stream under contention", bad, CHAINS, firstGot, want)
	}
}

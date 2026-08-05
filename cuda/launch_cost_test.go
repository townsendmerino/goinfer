//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestCUDA_launchCost bounds the per-launch host cost — Step 0 for the dispatch-overhead lever.
// The refined 26B decomposition put ~19 ms/token across ~600 launches ≈ 32 µs/launch, several times
// a normal CUDA launch (~5 µs). That points at the purego FFI crossing (cgo-free: every
// cuLaunchKernel is a dlopen'd-symbol call + Go-side arg packing), NOT GPU-side dispatch. If
// confirmed, CUDA GRAPHS (capture once, replay in one call — collapses N crossings into one) are the
// tool, not batching (which only cuts dispatch COUNT). The split: launch a cheap kernel at the real
// grid vs a minimal 1×1 grid — if per-launch is ~equal and both ~30 µs, the cost is launch-bound
// (FFI + packing), not compute-bound, and graphs win.
//
// Model-independent (the FFI cost is the same for any model), so it runs on the tiny fixture.
func TestCUDA_launchCost(t *testing.T) {
	const dir = "../testdata/gemma4-moe-tiny"
	mc, rf := loadG4MoECache(t, dir, false)
	defer mc.Close()
	r := rf.(*cudaResident)

	// Check 3 (graph-capturable fraction): count dispatches for one forward, divide by layers →
	// per-layer count. Only rope_kv + attention are per-token DYNAMIC (pos/nKeys), so they stay out
	// of graphs; the static fraction bounds the lever. Run at a filled window so attention's shared-mem
	// geometry has stabilized (pos ≥ window) — representative of steady-state decode.
	nLayers := len(r.layers)
	r.launchN = 0
	if _, err := r.Forward(mc.EmbedResidentForTest(1), 0); err != nil {
		t.Fatalf("forward: %v", err)
	}
	perLayer := float64(r.launchN) / float64(nLayers)
	const dynamicPerLayer = 2.0 // rope_kv + attention
	staticFrac := (perLayer - dynamicPerLayer) / perLayer
	t.Logf("dispatch count: %d launches / %d layers = %.0f/layer; static (graph-capturable) = %.0f/layer = %.0f%%",
		r.launchN, nLayers, perLayer, perLayer-dynamicPerLayer, staticFrac*100)

	const N = 20000
	// scale_vec (fScaleVec) is a trivial elementwise kernel: dst[i] *= s. Launch it N times with one
	// sync and divide → per-launch host cost (arg packing + purego call + driver dispatch + a tiny
	// GPU op). Two geometries to separate crossing from device work.
	bench := func(cfg LaunchConfig) float64 {
		var us float64
		_ = r.do(func() error {
			for i := 0; i < 200; i++ {
				_ = r.launch(r.fScaleVec, cfg, Arg(r.x), gpu.ArgValue(float32(1.0)), gpu.ArgValue(int32(r.hidden)))
			}
			_ = r.stream.Sync()
			start := time.Now()
			for i := 0; i < N; i++ {
				_ = r.launch(r.fScaleVec, cfg, Arg(r.x), gpu.ArgValue(float32(1.0)), gpu.ArgValue(int32(r.hidden)))
			}
			_ = r.stream.Sync()
			us = time.Since(start).Seconds() * 1e6 / float64(N)
			return nil
		})
		return us
	}

	minimal := bench(LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1})     // ~no GPU work
	bigGrid := bench(LaunchConfig{GridX: 512, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}) // 512 blocks dispatched; n guards all but 1 → driver sets up the whole grid, ~no real compute

	// Native (cgo) CUDA launch is ~5 µs. If the cgo-free crossing sits well above that AND barely
	// moves with a 512× larger grid, the cost is the purego FFI crossing + Go-side arg packing, not
	// GPU-side dispatch — the case CUDA GRAPHS collapse (replay N crossings as one call).
	const nativeUs = 5.0
	t.Logf("per-launch host cost through the cgo-free (purego) stack:")
	t.Logf("  minimal 1×1 grid:   %.2f µs/launch", minimal)
	t.Logf("  512-block grid:     %.2f µs/launch  (%.2f× the minimal)", bigGrid, bigGrid/minimal)
	t.Logf("  native cgo launch ≈ %.1f µs for reference", nativeUs)
	switch {
	case minimal > 2*nativeUs && bigGrid/minimal < 1.5:
		t.Logf("  VERDICT: FFI/LAUNCH-BOUND — per-launch is %.1f× native and ~grid-independent, so it's the purego "+
			"crossing + arg packing, NOT GPU dispatch. CUDA GRAPHS are the tool (capture the ~96%%-static per-layer "+
			"segments, replay in one call). No gocudrv fork: cudasys already exposes StreamBeginCapture/EndCapture/"+
			"GraphInstantiate/GraphLaunch. Batching cuts dispatch COUNT only (~20→13/layer; expertsDown don't share "+
			"activation) — the smaller win.", minimal/nativeUs)
	case bigGrid/minimal >= 1.5:
		t.Logf("  VERDICT: grid scaling %.1f× — GPU-side dispatch is a real share; batching helps meaningfully too.", bigGrid/minimal)
	default:
		t.Logf("  VERDICT: per-launch ~native — the dispatch estimate was off; re-examine.")
	}
}

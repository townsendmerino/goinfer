//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// TestRouteGptOssGrowsPoolPastMoERoute measures the gap audit-2026-09-10 G-13(b) closed.
//
// BuildResident pays the deferred local-memory reservation before sizing the expert cache by
// launching the kernel with the most per-thread scratch (see TestMoERouteFirstLaunchReservation for
// the mechanism). It launched moe_route, 4416 B/thread. route_gptoss declares 4608, and it is the
// router on gpt-oss, the model the expert cache exists for. This replays the old warm-up (moe_route
// alone), then reads what route_gptoss's first launch takes on top. That figure is what allocSlots
// could not see on gpt-oss before the fix.
//
// The fix's own property is checked after that: once both are forced, a launch shaped like a real
// gpt-oss-20b token (nE=32, k=4) must cost nothing.
//
// The reservation belongs to the context, so this measures only if nothing earlier in the process
// launched either kernel. Run it alone.
func TestRouteGptOssGrowsPoolPastMoERoute(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	read := func() int64 {
		f, _, e := dev.Context().MemInfo()
		if e != nil {
			t.Fatalf("MemInfo: %v", e)
		}
		return int64(f)
	}
	pipe := func(ptx []byte, name string) Pipeline {
		mod, e := dev.CompileLibrary(ptx)
		if e != nil {
			t.Fatalf("CompileLibrary for %s: %v", name, e)
		}
		p, e := dev.NewComputePipeline(mod, name)
		if e != nil {
			t.Fatalf("NewComputePipeline(%s): %v", name, e)
		}
		return p
	}
	fRoute := pipe(moePTXOrOverride(), "moe_route")
	fGptOss := pipe(gptOssActPTX, "route_gptoss")
	q := dev.NewCommandQueue()
	one := LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 1, BlockY: 1, BlockZ: 1}
	cost := func(what string, launch func() error) int64 {
		before := read()
		if e := launch(); e != nil {
			t.Fatalf("%s: launch: %v", what, e)
		}
		if e := q.Sync(); e != nil {
			t.Fatalf("%s: sync: %v", what, e)
		}
		c := before - read()
		t.Logf("  %-44s cost %11d B (%.1f MiB)", what, c, float64(c)/(1<<20))
		return c
	}
	logits, bias := afn(dev, 32), afn(dev, 32)
	idx, wgt := aun(dev, 4), afn(dev, 4)
	gptOss := func(nE, k int32) func() error {
		return func() error {
			return q.Launch(fGptOss, one, Arg(logits), Arg(bias), Arg(idx), Arg(wgt),
				gpu.ArgValue(nE), gpu.ArgValue(k))
		}
	}

	// backend.go's warm-up before G-13(b): moe_route alone, with its arguments.
	route := cost("moe_route (the old warm-up)", func() error {
		return q.Launch(fRoute, one, Arg(logits), Arg(logits), Arg(idx), Arg(wgt),
			gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)),
			gpu.ArgValue(int32(0)), gpu.ArgValue(float32(1)),
			gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)))
	})
	if route == 0 {
		t.Skipf("could not evaluate: the local-memory pool was already reserved before this test ran, "+
			"so moe_route's first launch read 0 B. Run it alone (-run '^%s$').", t.Name())
	}
	grow := cost("route_gptoss after moe_route (the gap)", gptOss(1, 1))
	onToken := cost("route_gptoss at nE=32 k=4, both forced", gptOss(32, 4))

	if grow <= 0 {
		t.Errorf("route_gptoss's first launch after moe_route took %d B. It declares more local memory "+
			"than moe_route (4608 against 4416 B/thread), and backend.go's G-13(b) comment says that "+
			"left the pool to grow after allocSlots on gpt-oss. A non-positive reading refutes that "+
			"comment: correct it rather than this assertion", grow)
	}
	if onToken != 0 {
		t.Errorf("with moe_route and route_gptoss both forced, a gpt-oss-shaped launch still took %d B: "+
			"forcing both does not cover the pool, so the expert cache would be sized against "+
			"memory about to be taken", onToken)
	}
}

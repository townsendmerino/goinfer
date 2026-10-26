//go:build darwin

package gpudev

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// workSrc is a fixed, modest per-thread compute kernel — the SAME GPU work in all three regimes, so
// the difference between their timings is PURE submission/handshake overhead, not compute. `iters`
// tunes the segment length so each "layer" has realistic-ish GPU time.
const workSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void work(device float* buf[[buffer(0)]], constant uint& iters[[buffer(1)]], uint i[[thread_position_in_grid]]) {
    float x = buf[i];
    for (uint k=0;k<iters;k++) x = x*1.0000001f + 0.0000001f;
    buf[i] = x;
}
`

// TestSharedEventHandshake is Step-6 Step-0 option (3): does a ONE-submit-per-token command buffer
// with per-layer CPU⇄GPU MTLSharedEvent handshakes recover the throughput that per-layer submit+wait
// loses? It prices three regimes over N segments (N = a 28-layer decode's inter-layer boundaries) of
// identical GPU work:
//
//	(1) baseline   — one command buffer, N dispatches, one commit+wait (the pre-encode trunk shape).
//	(2) per-layer  — N command buffers, N commit+wait cycles (the naive-paging regime; goinfer's
//	    TestPageCost_submissionStructure measured this at +106% / ~0.557ms per extra submit).
//	(3) shared-evt — one command buffer; after each segment the GPU signals and waits, a spinning CPU
//	    acks between (where the real path would read the router idx + stage experts). One submit/token.
//
// The delta (3)-(1) is the per-boundary HANDSHAKE cost; (2)-(1) is the per-boundary SUBMIT cost.
//
// FINDING (authoritative number is the REAL forward, not this synthetic): measured on goinfer's real
// qwen2.5-1.5b decode via encodeLayer + these bindings, the shared-event handshake (~0.26 ms/boundary)
// is ≈ a full per-layer submit (~0.23 ms/boundary) — it recovers ~0% of the per-layer loss; both
// synchronous regimes cost ~+45% over the single-submit baseline. The GPU stall at encodeWaitForEvent
// + the CPU round-trip cost about the same as committing a separate command buffer. This synthetic
// probe (trivial fixed segments) UNDERSTATES the handshake cost — light segments let the GPU hide part
// of the wait, so it can read a rosy "recovers ~35%"; do not trust that over the real-forward result.
// Conclusion for Step 6: neither synchronous shape is viable → speculative prefetch (encode last
// token's expert set, correct on miss) is the path. Kept as the bindings' regression + mechanism demo.
// Env-gated (perf probe, noisy in CI).
func TestSharedEventHandshake(t *testing.T) {
	if os.Getenv("GOINFER_HANDSHAKE_PROBE") == "" {
		t.Skip("set GOINFER_HANDSHAKE_PROBE=1 to run the shared-event handshake probe (perf timing)")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(workSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	work, err := d.NewComputePipeline(lib, "work")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const N = 28       // inter-layer boundaries of a 28-layer decode
	const grid = 65536 // threads per dispatch
	const iters = 128  // per-thread inner loop
	const dPerSeg = 12 // dispatches per segment ≈ a real decode layer's dispatch count — matters because
	// per-command-buffer SUBMIT cost scales with the buffer's dispatch count, while the per-boundary
	// HANDSHAKE cost does not; comparing them at 1 dispatch/segment understates the handshake's edge.
	buf := d.NewBufferLen(grid)
	uIters := NewBufferOf(d, []uint32{iters})
	ev := d.NewSharedEvent()
	queue := d.NewCommandQueue()

	// spinUntil busy-polls the event to `target` with a wall-clock bailout so a protocol bug hangs the
	// test in seconds instead of forever (a GPU waitUntilCompleted deadlock has no timeout otherwise).
	spinUntil := func(target uint64) bool {
		deadline := time.Now().Add(5 * time.Second)
		for n := 0; ev.Value() < target; n++ {
			if n&0xffff == 0 && time.Now().After(deadline) {
				return false
			}
		}
		return true
	}

	baseline := func() { // (1) one CB, N dispatches, one commit+wait
		e := queue.Begin()
		for range N {
			for range dPerSeg {
				e.Dispatch(work, grid, 256, buf, uIters)
			}
		}
		e.End()
	}
	perLayer := func() { // (2) N CBs, N commit+wait
		for range N {
			e := queue.Begin()
			for range dPerSeg {
				e.Dispatch(work, grid, 256, buf, uIters)
			}
			e.End()
		}
	}
	var base uint64            // monotonic event base (MTLSharedEvent must not go backwards, so never reset)
	sharedEvt := func() bool { // (3) one CB, per-boundary GPU signal/wait + CPU ack
		e := queue.Begin()
		for i := range N {
			for range dPerSeg {
				e.Dispatch(work, grid, 256, buf, uIters)
			}
			if i < N-1 {
				e.EventBoundary(ev, base+uint64(2*i+1), base+uint64(2*i+2))
			}
		}
		e.FinishEncoding()
		e.Commit()
		ok := true
		for i := range N - 1 { // CPU handshake, concurrent with GPU execution
			if !spinUntil(base + uint64(2*i+1)) { // GPU finished segment i?
				ok = false
				break
			}
			_ = buf.Floats()[0]               // host readback stand-in (the real path reads router idx here)
			ev.SetValue(base + uint64(2*i+2)) // ack → GPU proceeds to segment i+1
		}
		e.WaitDone()
		e.DrainPool()
		base += uint64(2 * N) // keep values monotonic across tokens
		return ok
	}

	bestMs := func(run func()) float64 {
		best := 1e18
		for r := range 4 {
			start := time.Now()
			run()
			ms := float64(time.Since(start).Microseconds()) / 1000.0
			if r > 0 && ms < best {
				best = ms
			}
		}
		return best
	}
	// warm + correctness: the shared-event regime must not hang (protocol correct).
	if !sharedEvt() {
		t.Fatal("shared-event handshake deadlocked (spin bailout) — the signal/wait protocol is wrong")
	}

	baseMs := bestMs(baseline)
	perLayerMs := bestMs(perLayer)
	var evMs float64
	{
		best := 1e18
		for r := range 4 {
			start := time.Now()
			if !sharedEvt() {
				t.Fatal("shared-event handshake deadlocked mid-measurement")
			}
			ms := float64(time.Since(start).Microseconds()) / 1000.0
			if r > 0 && ms < best {
				best = ms
			}
		}
		evMs = best
	}

	perSubmit := (perLayerMs - baseMs) / float64(N-1)
	perHandshake := (evMs - baseMs) / float64(N-1)
	t.Logf("N=%d segments (grid=%d iters=%d), best-of-3 warm", N, grid, iters)
	t.Logf("  (1) baseline   1 CB, %d dispatch : %6.3f ms", N, baseMs)
	t.Logf("  (2) per-layer  %d CB submit+wait : %6.3f ms  (+%.1f%%, ~%.3f ms/submit)", N, perLayerMs, (perLayerMs-baseMs)/baseMs*100, perSubmit)
	t.Logf("  (3) shared-evt 1 CB, %d handshakes: %6.3f ms  (+%.1f%%, ~%.3f ms/handshake)", N-1, evMs, (evMs-baseMs)/baseMs*100, perHandshake)
	t.Logf("synthetic: handshake %.3f ms vs submit %.3f ms per boundary (recovers %.0f%%) — but the REAL-forward "+
		"result is authoritative: handshake ≈ submit, ~0%% recovery, speculative prefetch is the path (see header)",
		perHandshake, perSubmit, (1-(evMs-baseMs)/(perLayerMs-baseMs))*100)
}

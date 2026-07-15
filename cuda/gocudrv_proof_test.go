//go:build cuda

package cuda

import (
	"context"
	_ "embed"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
)

// addonePTX is a trivial verifiable kernel compiled cgo-free by NVRTC (release 12.6,
// compute_75) — the Layer-A proof loads/launches it via gocudrv's dlopen'd libcuda.
//
//go:embed testdata/addone.ptx
var addonePTX []byte

// TestGocudrvLayerA is the spike's Layer-A end-to-end proof (docs/task-cuda-cgofree-spike.md):
// does the cgo-free plumbing — dlopen libcuda (gocudrv) → JIT PTX → alloc/H2D → launch →
// CUDA-event time → D2H → correct result — actually work on this 2070 SUPER? It also
// measures the two numbers the spike flags: the CUDA-event KERNEL time and the WALL time
// per launch (the delta is gocudrv's per-call channel-hop tax, §1.2). Skips cleanly with
// no device. Correctness here is a real (if trivial) GPU compute, argmax-parity is the
// megakernel's job. Run: CGO_ENABLED=0 go test -tags cuda ./cuda/ -run LayerA -v
func TestGocudrvLayerA(t *testing.T) {
	if err := gc.Init(); err != nil { // dlopen libcuda.so.1 + cuInit(0) — the cgo-free entry
		t.Skipf("cuInit failed (no driver): %v", err)
	}
	dev, err := gc.GetDevice(0)
	if err != nil {
		t.Skipf("no CUDA device (driver-only box needed): %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Skipf("no primary context: %v", err)
	}
	defer ctx.Close()
	bg := context.Background()

	t0 := time.Now()
	mod, err := ctx.LoadModule(addonePTX) // JIT the PTX via the driver (cold-start)
	if err != nil {
		t.Fatalf("LoadModule (JIT PTX): %v", err)
	}
	jitWall := time.Since(t0)
	fn, err := mod.Function("addone")
	if err != nil {
		t.Fatalf("Function: %v", err)
	}

	const n = 8192
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(i)
	}
	buf, err := gc.Alloc[float32](ctx, n)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	defer buf.Close()
	if err := gc.CopyHtoD(bg, buf, host); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	cfg := gc.LaunchConfig1D(n, 256)

	// warm (drop first-launch driver/JIT warmup, per the methodology)
	for i := 0; i < 5; i++ {
		if err := fn.LaunchOn(bg, stream, cfg, gc.Arg(buf), gc.ArgValue(int32(n))); err != nil {
			t.Fatalf("warm launch: %v", err)
		}
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("sync: %v", err)
	}

	// Two SEPARATE numbers, measured so neither contaminates the other (the naive
	// "events around a launch burst" conflates them: a trivial kernel drains faster than
	// gocudrv can enqueue, so the GPU STARVES and the event span just re-measures the
	// launch-limited wall — not kernel compute). So:
	//   (A) true GPU per-kernel time — events tightly around ONE synced launch (Elapsed is
	//       pure GPU work); best of many drops noise.
	//   (B) CPU per-launch cost — wall of an async enqueue BURST with no sync inside it.
	//       cuLaunchKernel is async and the driver queue is deep, so the GPU drains this
	//       trivial kernel in the background (no backpressure) and the burst wall is pure
	//       CPU: gocudrv's channel-hop to its LockOSThread executor + cuLaunchKernel. This
	//       is the per-dispatch tax the spike flags (§1.2).
	rounds := 8
	bestKernUs, bestEnqUs := 1e18, 1e18
	launched := 0

	// (A) true GPU per-kernel time
	for r := 0; r < rounds; r++ {
		start, _ := ctx.NewEvent()
		done, _ := ctx.NewEvent()
		_ = start.Record(stream)
		if err := fn.LaunchOn(bg, stream, cfg, gc.Arg(buf), gc.ArgValue(int32(n))); err != nil {
			t.Fatalf("launch: %v", err)
		}
		_ = done.Record(stream)
		if err := stream.Synchronize(bg); err != nil {
			t.Fatalf("sync: %v", err)
		}
		launched++
		kern, _ := start.Elapsed(done)
		if k := float64(kern.Nanoseconds()) / 1000; k < bestKernUs {
			bestKernUs = k
		}
	}

	// (B) CPU per-launch cost (channel-hop + enqueue), async burst, no sync inside
	const iters = 500
	for r := 0; r < rounds; r++ {
		tEnq := time.Now()
		for i := 0; i < iters; i++ {
			if err := fn.LaunchOn(bg, stream, cfg, gc.Arg(buf), gc.ArgValue(int32(n))); err != nil {
				t.Fatalf("launch: %v", err)
			}
		}
		enq := time.Since(tEnq)
		if err := stream.Synchronize(bg); err != nil {
			t.Fatalf("sync: %v", err)
		}
		launched += iters
		if e := float64(enq.Nanoseconds()) / float64(iters) / 1000; e < bestEnqUs {
			bestEnqUs = e
		}
	}

	// verify: each launch adds 1; buf[i] = i + (5 warm + launched).
	out := make([]float32, n)
	if err := gc.CopyDtoH(bg, out, buf); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	want := float32(1) + float32(5+launched) // host[1]=1
	if out[1] != want {
		t.Fatalf("wrong GPU result: out[1]=%g want %g (cgo-free launch produced incorrect compute)", out[1], want)
	}

	// Both numbers below are LAUNCH-OVERHEAD-bound: the addone kernel's real compute is
	// <1 us, so neither isolates GPU work — that's the point. They bracket the per-dispatch
	// tax the megakernel exists to amortize; the real GPU decode time is the Phase-2
	// megakernel measurement, not measurable from a trivial kernel.
	t.Logf("Layer A OK — cgo-free JIT+launch+event-time verified on the 2070 SUPER:")
	t.Logf("  JIT cold-start (LoadModule of embedded PTX): %.2f ms — does not wreck boot", float64(jitWall.Microseconds())/1000)
	t.Logf("  per-dispatch CPU cost (gocudrv channel-hop + cuLaunchKernel, async burst, best of 8×%d): %.2f us", iters, bestEnqUs)
	t.Logf("  one dependent synced launch (record→launch→sync, idle GPU incl. wake-up): %.2f us", bestKernUs)
	t.Logf("  ⇒ both are launch-bound (trivial-kernel compute <1us); at ~13 dispatches/layer this per-dispatch")
	t.Logf("     tax is exactly what collapsing to ~1-3 super-kernels/layer amortizes. Real GPU decode time = Phase 2.")
}

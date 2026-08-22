//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// quant_vec is the per-token activation quantizer on the decode path: one block of 256 threads, a
// maxabs reduction over N, then a scale-and-round pass. Its maxabs is a classic __syncthreads() tree
// across 8 warps — 8 barriers where a warp-shuffle needs one.
//
// COMPILED FROM THE SHIPPED PTX (`gluePTX`), never a private inline copy of the .cu. A benchmark
// that builds its own kernel measures a kernel nobody runs, and the embedded PTX is what the
// driver's JIT actually loads — so `./build_ptx.sh glue` is what moves this number, not a .cu edit.
//
// PRODUCTION GEOMETRY, not a round number: resident.go launches this as `onecfg(256, 256*4)` —
// GridX 1, BlockX 256, 1 KiB shared — over qDim-length activations. The sizes below bracket the real
// ones (gemma4 2048, Qwen3 4096, the 26B's 5120), plus one deliberately larger to separate the
// fixed reduction cost from the per-element scan: if a change helps only at 16384 it has not helped
// the decode path.
func benchQuantVec(b *testing.B, n int) {
	b.Helper()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	bg := context.Background()
	ctx := dev.Context()
	mod, err := ctx.LoadModule(gluePTX)
	if err != nil {
		b.Fatalf("LoadModule(gluePTX): %v", err)
	}
	fn, err := mod.Function("quant_vec")
	if err != nil {
		b.Fatalf("quant_vec: %v", err)
	}
	stream := mustStream(b, ctx)

	rng := rand.New(rand.NewSource(1))
	host := make([]float32, n)
	for i := range host {
		host[i] = float32(rng.NormFloat64())
	}
	dx := mustAlloc[float32](b, ctx, n)
	dq := mustAlloc[int32](b, ctx, n/4+1)
	dsc := mustAlloc[float32](b, ctx, 1)
	if err := gc.CopyHtoD(bg, dx, host); err != nil {
		b.Fatalf("CopyHtoD: %v", err)
	}

	cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4}
	launch := func() error {
		return fn.LaunchOn(bg, stream, cfg, gc.Arg(dx), gc.ArgValue(int32(n)), gc.Arg(dq), gc.Arg(dsc))
	}
	// One warm launch OUTSIDE the timer: the first launch of a JIT'd kernel pays module load, which
	// would otherwise land entirely in the b.N=1 probe and skew the whole run.
	if err := launch(); err != nil {
		b.Fatalf("warm launch: %v", err)
	}
	if err := stream.Synchronize(bg); err != nil {
		b.Fatalf("warm sync: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := launch(); err != nil {
			b.Fatalf("launch: %v", err)
		}
	}
	// Sync ONCE, inside the timer but outside the loop. A per-iteration Synchronize measures the
	// round trip of the sync rather than the kernel, and serializes exactly the pipelining that makes
	// a small dispatch-bound kernel cheap — the concurrency-instrument trap this repo has hit before.
	if err := stream.Synchronize(bg); err != nil {
		b.Fatalf("sync: %v", err)
	}
	b.StopTimer()
}

func BenchmarkQuantVec2048(b *testing.B)  { benchQuantVec(b, 2048) }
func BenchmarkQuantVec4096(b *testing.B)  { benchQuantVec(b, 4096) }
func BenchmarkQuantVec5120(b *testing.B)  { benchQuantVec(b, 5120) }
func BenchmarkQuantVec16384(b *testing.B) { benchQuantVec(b, 16384) }

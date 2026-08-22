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

// rmsnorm_quant is the fused norm+quantize on the decode path — pass 1 a sum of squares (untouched,
// float-non-associative), pass 2 a maxabs, pass 3 the pack. Only pass 2's reduction is safe to
// restructure, so only pass 2 is what this benchmark's number moves.
//
// Shared memory here is (H+256)*4, not 256*4: normed[] lives in shared alongside the reduction
// scratch, which is also why H cannot simply be raised without checking the launch's smem budget.
func benchRmsnormQuant(b *testing.B, h int) {
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
	fn, err := mod.Function("rmsnorm_quant")
	if err != nil {
		b.Fatalf("rmsnorm_quant: %v", err)
	}
	stream := mustStream(b, ctx)
	rng := rand.New(rand.NewSource(int64(h)))
	x, w := make([]float32, h), make([]float32, h)
	for i := range x {
		x[i], w[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	dx, dw := mustAlloc[float32](b, ctx, h), mustAlloc[float32](b, ctx, h)
	dq := mustAlloc[int32](b, ctx, h/4)
	dsc := mustAlloc[float32](b, ctx, 1)
	if err := gc.CopyHtoD(bg, dx, x); err != nil {
		b.Fatalf("CopyHtoD: %v", err)
	}
	if err := gc.CopyHtoD(bg, dw, w); err != nil {
		b.Fatalf("CopyHtoD: %v", err)
	}
	cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((h + 256) * 4)}
	launch := func() error {
		return fn.LaunchOn(bg, stream, cfg, gc.Arg(dx), gc.Arg(dw), gc.ArgValue(int32(h)),
			gc.ArgValue(float32(1e-6)), gc.ArgValue(int32(0)), gc.Arg(dq), gc.Arg(dsc))
	}
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
	if err := stream.Synchronize(bg); err != nil {
		b.Fatalf("sync: %v", err)
	}
	b.StopTimer()
}

func BenchmarkRmsnormQuant2048(b *testing.B) { benchRmsnormQuant(b, 2048) }
func BenchmarkRmsnormQuant4096(b *testing.B) { benchRmsnormQuant(b, 4096) }
func BenchmarkRmsnormQuant5120(b *testing.B) { benchRmsnormQuant(b, 5120) }

// glu_quant fuses the gated activation with the int8 quantize: d = act(g)*u, then maxabs, then pack.
// Only the maxabs reduction is restructured; the activation and the pack are untouched.
//
// act=1 is SwiGLU (llama/mistral/qwen), act=0 GeGLU (gemma). The benchmark runs SwiGLU because that
// is what the resident families on this box use; the reduction is identical either way.
func benchGluQuant(b *testing.B, inter int) {
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
	fn, err := mod.Function("glu_quant")
	if err != nil {
		b.Fatalf("glu_quant: %v", err)
	}
	stream := mustStream(b, ctx)
	rng := rand.New(rand.NewSource(int64(inter)))
	gu := make([]float32, 2*inter)
	for i := range gu {
		gu[i] = float32(rng.NormFloat64())
	}
	dgu := mustAlloc[float32](b, ctx, 2*inter)
	dscr := mustAlloc[float32](b, ctx, inter)
	dq := mustAlloc[int32](b, ctx, inter/4)
	dsc := mustAlloc[float32](b, ctx, 1)
	if err := gc.CopyHtoD(bg, dgu, gu); err != nil {
		b.Fatalf("CopyHtoD: %v", err)
	}
	cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4}
	launch := func() error {
		// (g, u, gOff, uOff, I, act, q, scale, dscratch) — MoE passes one buffer twice with
		// uOff=I, which is the shape exercised here.
		return fn.LaunchOn(bg, stream, cfg, gc.Arg(dgu), gc.Arg(dgu), gc.ArgValue(int32(0)),
			gc.ArgValue(int32(inter)), gc.ArgValue(int32(inter)), gc.ArgValue(int32(1)),
			gc.Arg(dq), gc.Arg(dsc), gc.Arg(dscr))
	}
	// Fatalf, NOT Skipf: a benchmark that silently skips on a launch error reports "ok" and measures
	// nothing, which is the skip-is-not-a-pass trap wearing a benchmark's clothes.
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
	if err := stream.Synchronize(bg); err != nil {
		b.Fatalf("sync: %v", err)
	}
	b.StopTimer()
}

func BenchmarkGluQuant4096(b *testing.B)  { benchGluQuant(b, 4096) }
func BenchmarkGluQuant11008(b *testing.B) { benchGluQuant(b, 11008) }

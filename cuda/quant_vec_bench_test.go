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

// ---- the BATCHED siblings (prefill_batched.ptx) ----
//
// Same three kernels, one block per row instead of one block total, so GridX is M. Production
// launches them at BlockX 256 from the drafter path (drafter.go). M=8 is the DFlash block size.

type gcArg = gc.KernelArg

func benchBatched(b *testing.B, name string, M, N int, args func(dx, dw *gc.Buffer[float32], dq *gc.Buffer[int32], dsc, dscr *gc.Buffer[float32]) []gcArg) {
	b.Helper()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	bg := context.Background()
	ctx := dev.Context()
	mod, err := ctx.LoadModule(prefillBatchedPTX)
	if err != nil {
		b.Fatalf("LoadModule(prefillBatchedPTX): %v", err)
	}
	fn, err := mod.Function(name)
	if err != nil {
		b.Fatalf("%s: %v", name, err)
	}
	stream := mustStream(b, ctx)
	rng := rand.New(rand.NewSource(int64(M * N)))
	host := make([]float32, 2*M*N)
	for i := range host {
		host[i] = float32(rng.NormFloat64())
	}
	dx := mustAlloc[float32](b, ctx, 2*M*N)
	dw := mustAlloc[float32](b, ctx, N)
	dq := mustAlloc[int32](b, ctx, M*N/4+1)
	dsc := mustAlloc[float32](b, ctx, M)
	dscr := mustAlloc[float32](b, ctx, M*N)
	if err := gc.CopyHtoD(bg, dx, host); err != nil {
		b.Fatalf("CopyHtoD: %v", err)
	}
	if err := gc.CopyHtoD(bg, dw, host[:N]); err != nil {
		b.Fatalf("CopyHtoD w: %v", err)
	}
	cfg := gc.LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((N + 256) * 4)}
	a := args(dx, dw, dq, dsc, dscr)
	launch := func() error { return fn.LaunchOn(bg, stream, cfg, a...) }
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

func BenchmarkRmsnormQuantBatched(b *testing.B) {
	benchBatched(b, "rmsnorm_quant_batched", 8, 4096, func(dx, dw *gc.Buffer[float32], dq *gc.Buffer[int32], dsc, dscr *gc.Buffer[float32]) []gcArg {
		return []gcArg{gc.Arg(dx), gc.Arg(dw), gc.ArgValue(int32(4096)), gc.ArgValue(float32(1e-6)),
			gc.ArgValue(int32(0)), gc.Arg(dq), gc.Arg(dsc)}
	})
}

func BenchmarkQuantVecBatched(b *testing.B) {
	benchBatched(b, "quant_vec_batched", 8, 4096, func(dx, dw *gc.Buffer[float32], dq *gc.Buffer[int32], dsc, dscr *gc.Buffer[float32]) []gcArg {
		return []gcArg{gc.Arg(dx), gc.ArgValue(int32(4096)), gc.Arg(dq), gc.Arg(dsc), gc.ArgValue(int32(8))}
	})
}

func BenchmarkGluQuantBatched(b *testing.B) {
	benchBatched(b, "glu_quant_batched", 8, 4096, func(dx, dw *gc.Buffer[float32], dq *gc.Buffer[int32], dsc, dscr *gc.Buffer[float32]) []gcArg {
		return []gcArg{gc.Arg(dx), gc.Arg(dx), gc.ArgValue(int32(0)), gc.ArgValue(int32(4096)),
			gc.ArgValue(int32(4096)), gc.ArgValue(int32(1)), gc.Arg(dq), gc.Arg(dsc), gc.Arg(dscr),
			gc.ArgValue(int32(8))}
	})
}

// splitkv_softmax is step 2 of the decode split-KV attention path: nH blocks of 128 threads, a
// softmax over nWin scores in place. It carries TWO tree reductions — a max and a sum — and only the
// max is safe to restructure; the sum is the softmax denominator, float-non-associative, and
// resident.go pins this kernel's block width at 128 to keep it byte-identical to attn_batched.
//
// nWin is the context window, so it is the variable that decides whether the reductions matter at
// all: the two strided passes are O(nWin) while the ladders are O(log 128) regardless.
func benchSplitkvSoftmax(b *testing.B, nH, nWin int) {
	b.Helper()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	bg := context.Background()
	ctx := dev.Context()
	mod, err := ctx.LoadModule(decodeSplitKVPTX)
	if err != nil {
		b.Fatalf("LoadModule(decodeSplitKVPTX): %v", err)
	}
	fn, err := mod.Function("splitkv_softmax")
	if err != nil {
		b.Fatalf("splitkv_softmax: %v", err)
	}
	stream := mustStream(b, ctx)
	rng := rand.New(rand.NewSource(int64(nWin)))
	host := make([]float32, nH*nWin)
	for i := range host {
		host[i] = float32(rng.NormFloat64() * 3)
	}
	dsc := mustAlloc[float32](b, ctx, nH*nWin)
	dinv := mustAlloc[float32](b, ctx, nH)
	if err := gc.CopyHtoD(bg, dsc, host); err != nil {
		b.Fatalf("CopyHtoD: %v", err)
	}
	cfg := gc.LaunchConfig{GridX: uint32(nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
		SharedMemBytes: 128 * 4}
	// sinks is nullptr in every resident caller today (only gpt-oss has one and it is not
	// resident-eligible), so the benchmark measures the shipped path.
	launch := func() error {
		return fn.LaunchOn(bg, stream, cfg, gc.Arg(dsc), gc.ArgValue(int32(nH)),
			gc.ArgValue(int32(nWin)), gc.Arg(dinv), gc.ArgDevicePtr(0))
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

func BenchmarkSplitkvSoftmax512(b *testing.B)  { benchSplitkvSoftmax(b, 32, 512) }
func BenchmarkSplitkvSoftmax2048(b *testing.B) { benchSplitkvSoftmax(b, 32, 2048) }

// attn_block_full is the DFlash drafter's non-causal attention: grid nH x M, BlockX 128, a two-pass
// softmax over nKeys = startPos + M. Its own comment records the kernel as L1TEX-throughput-saturated
// (the float4 K read was added for exactly that reason), so the reduction is a smaller share here
// than in any quant kernel — which is the prediction this benchmark exists to test rather than assume.
func benchAttnBlock(b *testing.B, nH, nKV, hd, startPos, M int) {
	b.Helper()
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	bg := context.Background()
	ctx := dev.Context()
	mod, err := ctx.LoadModule(attnBlockPTX)
	if err != nil {
		b.Fatalf("LoadModule(attnBlockPTX): %v", err)
	}
	fn, err := mod.Function("attn_block_full")
	if err != nil {
		b.Fatalf("attn_block_full: %v", err)
	}
	stream := mustStream(b, ctx)
	nKeys := startPos + M
	qDim, kvDim := nH*hd, nKV*hd
	rng := rand.New(rand.NewSource(int64(startPos)))
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	dq := mustAlloc[float32](b, ctx, M*qDim)
	dk := mustAlloc[float32](b, ctx, nKeys*kvDim)
	dv := mustAlloc[float32](b, ctx, nKeys*kvDim)
	dout := mustAlloc[float32](b, ctx, M*qDim)
	for buf, n := range map[*gc.Buffer[float32]]int{dq: M * qDim, dk: nKeys * kvDim, dv: nKeys * kvDim} {
		if err := gc.CopyHtoD(bg, buf, fill(n)); err != nil {
			b.Fatalf("CopyHtoD: %v", err)
		}
	}
	cfg := gc.LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((nKeys + 128) * 4)}
	launch := func() error {
		return fn.LaunchOn(bg, stream, cfg, gc.Arg(dq), gc.Arg(dk), gc.Arg(dv),
			gc.ArgValue(int32(nH)), gc.ArgValue(int32(nKV)), gc.ArgValue(int32(hd)),
			gc.ArgValue(int32(startPos)), gc.ArgValue(float32(1.0/11.3137)),
			gc.ArgValue(int32(0)), gc.ArgValue(int32(M)), gc.Arg(dout))
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

func BenchmarkAttnBlockFull512(b *testing.B) { benchAttnBlock(b, 32, 8, 128, 512, 8) }

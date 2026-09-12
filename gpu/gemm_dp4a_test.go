//go:build gpu

package gpu

import (
	"fmt"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
)

// runTiledKernel dispatches ONE explicit tiled-GEMM WGSL variant (bypassing
// ensureTiled's single cached pipeline slot, which always picks whichever variant
// Context.hasDP4A selected) — so a test can exercise BOTH the DP4A and the
// unpacked-scalar kernel in the same process regardless of what this machine's
// adapter actually probed. Mirrors MatmulW8A8Tiled's buffer/dispatch shape exactly;
// kept test-only rather than folded into gemm.go since production code never needs
// to run a variant other than the one hasDP4A picked.
func runTiledKernel(c *Context, code string, aq []int8, aScales []float32, rm *ResidentW8A8, M int) ([]float32, error) {
	sh, err := c.device.TryCreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:      "tiled-variant-probe",
		WGSLSource: &wgpu.ShaderSourceWGSL{Code: code},
	})
	if err != nil {
		return nil, fmt.Errorf("compile: %w", err)
	}
	defer sh.Release()
	pl, err := c.device.TryCreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "tiled-variant-probe",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		return nil, fmt.Errorf("pipeline: %w", err)
	}
	defer pl.Release()
	layout := pl.GetBindGroupLayout(0)
	defer layout.Release()

	K, N := rm.cols, rm.rows
	aBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(packInt8(aq, M, K)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer aBuf.Release()
	asBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(aScales[:M]), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer asBuf.Release()
	dstSize := uint64(M * N * 4)
	dstBuf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: dstSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	defer dstBuf.Release()
	dimsBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return nil, err
	}
	defer dimsBuf.Release()
	stage, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: dstSize, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	defer stage.Release()
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
		{Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()},
		{Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()},
		{Binding: 5, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return nil, err
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((uint32(N)+15)/16, (uint32(M)+15)/16, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return nil, err
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(dstBuf, 0, stage, 0, dstSize); err != nil {
		return nil, err
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return nil, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.MapAsyncStatus(0)
	if err := stage.TryMapAsync(wgpu.MapModeRead, 0, dstSize, func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("map failed: %v", status)
	}
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](stage.GetMappedRange(0, uint(dstSize))))
	if err := stage.TryUnmap(); err != nil {
		return nil, err
	}
	return out, nil
}

// TestTiledDP4A_parity is the Increment-1 gate (docs/task-gpu-batched-prefill.md's
// prerequisite): the dot4I8Packed tiled-GEMM kernel must match BOTH the int32 CPU
// reference AND the scalar-unpack kernel, at odd dims that exercise every tile
// tail. Runs both variants explicitly via runTiledKernel regardless of what this
// machine's Context.hasDP4A actually probed, so the DP4A path stays covered on a
// non-DP4A CI/test box and the fallback path stays covered on a DP4A box (today
// both dev machines probe DP4A-capable, so relying on ensureTiled's live selection
// alone would silently stop exercising the fallback kernel).
func TestTiledDP4A_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	if !ctx.hasDP4A {
		t.Skip("adapter does not accept dot4I8Packed; nothing DP4A-specific to gate here (fallback-only, covered by TestTiled_parity)")
	}

	const M, K, N = 37, 517, 131
	weight := randMat(N*K, 1)
	act := randMat(M*K, 2)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	aq, aScales := linalg.QuantizeRowsInt8(act, M, K)

	ref := make([]float32, M*N)
	for m := range M {
		for n := range N {
			var acc int32
			for k := range K {
				acc += int32(aq[m*K+k]) * int32(bq[n*K+k])
			}
			ref[m*N+n] = float32(acc) * aScales[m] * bScales[n]
		}
	}
	rm, err := ctx.UploadW8A8(bq, bScales, N, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()

	unpacked, err := runTiledKernel(ctx, matmulTiledW8A8ShaderWGSL, aq, aScales, rm, M)
	if err != nil {
		t.Fatalf("unpacked variant: %v", err)
	}
	packed, err := runTiledKernel(ctx, matmulTiledW8A8DP4AShaderWGSL, aq, aScales, rm, M)
	if err != nil {
		t.Fatalf("dot4I8Packed variant: %v", err)
	}

	cosU, maxU := cosine(unpacked, ref)
	cosP, maxP := cosine(packed, ref)
	t.Logf("unpacked vs CPU: cosine=%.8f maxAbs=%.3e", cosU, maxU)
	t.Logf("dot4I8Packed vs CPU: cosine=%.8f maxAbs=%.3e", cosP, maxP)
	if cosU < 0.99999 || maxU > 1e-3 {
		t.Errorf("unpacked variant diverges from CPU oracle: cosine=%.8f maxAbs=%.3e", cosU, maxU)
	}
	if cosP < 0.99999 || maxP > 1e-3 {
		t.Errorf("dot4I8Packed variant diverges from CPU oracle: cosine=%.8f maxAbs=%.3e", cosP, maxP)
	}

	// The two GPU variants should be bit-identical (same int32 accumulation order
	// within each 4-wide dot, same tiling/scheduling) — not just both "close to" the
	// CPU float reference.
	for i := range unpacked {
		if unpacked[i] != packed[i] {
			t.Fatalf("dot4I8Packed diverges from unpacked at [%d]: packed=%v unpacked=%v", i, packed[i], unpacked[i])
		}
	}
}

// TestTiledDP4A_microbench compares the DP4A kernel's throughput against the
// scalar-unpack kernel and the M=1 GEMV, at realistic prefill M — this is the
// number that proves or disproves task-gpu-batched-prefill.md's whole premise
// (the tiled GEMM must clear the bandwidth-bound M=1 GEMV to make batching worth
// it at all). Skips (not fails) when this adapter doesn't accept dot4I8Packed —
// informational, not a correctness gate.
func TestTiledDP4A_microbench(t *testing.T) {
	if testing.Short() {
		t.Skip("microbench")
	}
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if !ctx.hasDP4A {
		t.Skip("adapter does not accept dot4I8Packed")
	}
	t.Logf("backend: %s", ctx.Backend())

	const K, N = 4096, 4096
	weight := randMat(N*K, 1)
	bq, bScales := linalg.QuantizeRowsInt8(weight, N, K)
	rm, err := ctx.UploadW8A8(bq, bScales, N, K)
	if err != nil {
		t.Fatalf("UploadW8A8: %v", err)
	}
	defer rm.Release()

	// sequentialGEMV runs M *real* single-token GEMV dispatches (MatmulW8A8GEMV,
	// the actual decode-path kernel) — this, not the naive M-row matmul, is what
	// today's option-(a) O(prompt-len) prefill loop actually costs per layer
	// projection. That's the baseline the tiled GEMM must beat for batching to
	// pay off at all (docs/task-gpu-batched-prefill.md).
	sequentialGEMV := func(aq []int8, aScales []float32, M int) error {
		for m := range M {
			if _, err := ctx.MatmulW8A8GEMV(aq[m*K:(m+1)*K], aScales[m], rm); err != nil {
				return err
			}
		}
		return nil
	}

	bench := func(M, iters int) {
		act := randMat(M*K, uint64(M)+7)
		aq, aScales := linalg.QuantizeRowsInt8(act, M, K)
		if err := sequentialGEMV(aq, aScales, M); err != nil {
			t.Fatalf("gemv warmup: %v", err)
		}
		if _, err := runTiledKernel(ctx, matmulTiledW8A8ShaderWGSL, aq, aScales, rm, M); err != nil {
			t.Fatalf("unpacked warmup: %v", err)
		}
		if _, err := runTiledKernel(ctx, matmulTiledW8A8DP4AShaderWGSL, aq, aScales, rm, M); err != nil {
			t.Fatalf("dp4a warmup: %v", err)
		}

		t0 := time.Now()
		for range iters {
			if err := sequentialGEMV(aq, aScales, M); err != nil {
				t.Fatalf("gemv: %v", err)
			}
		}
		gemv := time.Since(t0) / time.Duration(iters)
		t1 := time.Now()
		for range iters {
			if _, err := runTiledKernel(ctx, matmulTiledW8A8ShaderWGSL, aq, aScales, rm, M); err != nil {
				t.Fatalf("unpacked: %v", err)
			}
		}
		unpacked := time.Since(t1) / time.Duration(iters)
		t2 := time.Now()
		for range iters {
			if _, err := runTiledKernel(ctx, matmulTiledW8A8DP4AShaderWGSL, aq, aScales, rm, M); err != nil {
				t.Fatalf("dp4a: %v", err)
			}
		}
		dp4a := time.Since(t2) / time.Duration(iters)

		gf := func(d time.Duration) float64 { return float64(2*M*N*K) / d.Seconds() / 1e9 }
		t.Logf("M=%-4d  M-sequential-GEMV %7.2f GFLOP/s-equiv (%v total)  |  unpacked-tiled %7.2f GFLOP/s  |  dot4I8Packed-tiled %7.2f GFLOP/s  (dp4a vs sequential-gemv %.2f×, dp4a vs unpacked %.2f×)",
			M, gf(gemv), gemv, gf(unpacked), gf(dp4a), gf(dp4a)/gf(gemv), gf(dp4a)/gf(unpacked))
	}
	bench(256, 5)
	bench(1024, 2)
}

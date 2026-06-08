//go:build gpu

package gpu

import (
	"testing"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

// TestW4A8_parity_and_bandwidth is the W4A8 probe: it validates the int4
// group-wise GEMV kernel against a CPU reference (the exact same grouped
// int8×int4 math) and measures its standalone bandwidth, then projects the
// decode token. The question: does int4 actually cut the 4.3 ms gemv floor, and
// by how much once the per-group f32 scales (~⅛ extra bytes) are counted?
func TestW4A8_parity_and_bandwidth(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()
	if err := ctx.ensureGEMVW4(); err != nil {
		t.Fatal(err)
	}

	const N, K = 8960, 1536 // the gate/up shape — the biggest decode GEMV
	const group = w4a8GroupSize
	kp := padK32(K)
	nGroups := kp / group

	// deterministic data: nibbles 0..15 (value nib-8), per-group f32 scales,
	// int8 activation + its scale.
	lcg := uint64(0x9e3779b97f4a7c15)
	next := func() uint64 { lcg = lcg*6364136223846793005 + 1442695040888963407; return lcg }
	nib := make([]uint8, N*K)
	for i := range nib {
		nib[i] = uint8(next() & 0xF)
	}
	scales := make([]float32, N*nGroups)
	for i := range scales {
		scales[i] = 0.02 + float32(next()%100)/5000 // ~0.02..0.04
	}
	act := make([]int8, kp)
	for i := range K {
		act[i] = int8(int(next()%255) - 127)
	}
	aScale := float32(0.013)

	// CPU reference: dst[n] = aScale * Σ_g scale[n,g] · Σ_{k in g} (nib-8)·act[k]
	ref := make([]float32, N)
	for n := range N {
		var total float64
		for g := range nGroups {
			var idot int
			for e := range group {
				k := g*group + e
				if k >= K {
					break
				}
				idot += (int(nib[n*K+k]) - 8) * int(act[k])
			}
			total += float64(idot) * float64(f16to32(f32to16(scales[n*nGroups+g]))) // f16 as the GPU reads them
		}
		ref[n] = float32(total * float64(aScale))
	}

	// upload weights; build activation + scale + dst + dims buffers + bind group.
	rm, err := ctx.UploadW4A8(nib, scales, N, K)
	if err != nil {
		t.Fatal(err)
	}
	defer rm.Release()
	dev := ctx.device
	aBuf, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "w4-act", Contents: wgpu.ToBytes(packInt8(act, 1, kp)), Usage: wgpu.BufferUsageStorage})
	defer aBuf.Release()
	asBuf, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "w4-ascale", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage})
	defer asBuf.Release()
	dstBuf, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Label: "w4-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer dstBuf.Release()
	dimsBuf, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "w4-dims", Contents: wgpu.ToBytes([]uint32{1, uint32(kp), uint32(N), 0}), Usage: wgpu.BufferUsageUniform})
	defer dimsBuf.Release()
	stag, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Label: "w4-stage", Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stag.Release()
	bg, _ := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ctx.gemvW4Layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
		{Binding: 1, Buffer: rm.bq, Size: rm.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()},
		{Binding: 3, Buffer: rm.bScales, Size: rm.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()},
		{Binding: 5, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
	}})
	defer bg.Release()
	gx, gy := gemvGrid(N)

	// --- parity: one dispatch, read back, cosine vs ref ---
	enc, _ := dev.CreateCommandEncoder(nil)
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(ctx.gemvW4Pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.End()
	pass.Release()
	enc.CopyBufferToBuffer(dstBuf, 0, stag, 0, uint64(N*4))
	cmd, _ := enc.Finish(nil)
	ctx.queue.Submit(cmd)
	cmd.Release()
	enc.Release()
	st := wgpu.BufferMapAsyncStatusUnknown
	stag.MapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.BufferMapAsyncStatus) { st = s })
	ctx.device.Poll(true, nil)
	if st != wgpu.BufferMapAsyncStatusSuccess {
		t.Fatalf("map failed: %v", st)
	}
	got := make([]float32, N)
	copy(got, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(N*4))))
	stag.Unmap()
	cos, maxAbs := cosine(got, ref)
	t.Logf("W4A8 GEMV parity [%d×%d, group %d]: cosine=%.6f maxAbs=%.3e", N, K, group, cos, maxAbs)
	if cos < 0.9999 {
		t.Errorf("W4A8 GEMV diverges: cosine=%.6f maxAbs=%.3e", cos, maxAbs)
	}

	// --- bandwidth: K dispatches in one pass, slope removes fixed submit/poll ---
	timeKd := func(Kd, reps int) time.Duration {
		best := time.Hour
		for range reps {
			e, _ := dev.CreateCommandEncoder(nil)
			p := e.BeginComputePass(nil)
			p.SetPipeline(ctx.gemvW4Pipeline)
			p.SetBindGroup(0, bg, nil)
			for range Kd {
				p.DispatchWorkgroups(gx, gy, 1)
			}
			p.End()
			p.Release()
			c2, _ := e.Finish(nil)
			t0 := time.Now()
			ctx.queue.Submit(c2)
			ctx.device.Poll(true, nil)
			d := time.Since(t0)
			c2.Release()
			e.Release()
			if d < best {
				best = d
			}
		}
		return best
	}
	tlo, thi := timeKd(1, 20), timeKd(200, 20)
	perUs := float64((thi - tlo).Microseconds()) / 199.0
	// bytes this GEMV streams: nibbles (kp/2 per row) + f16 group scales (nGroups*2 per row)
	wBytes := float64(N*kp/2 + N*nGroups*2)
	gbs := wBytes / (perUs * 1e3)
	int8Bytes := float64(N*kp + N*4) // same matrix at int8, for the ratio
	t.Logf("W4A8 GEMV (f16 scales): %.1f µs/dispatch | %.2f MB (%.0f%% of int8's %.2f MB) | %.1f GB/s",
		perUs, wBytes/1e6, wBytes/int8Bytes*100, int8Bytes/1e6, gbs)

	// --- project the decode token vs the measured int8 baseline ---
	// int8 token = 11.15 ms = gemv 4.3 (1.55 GB @ ~360 GB/s) + (glue+attn+
	// barriers+host) 6.85 (fixed — W4A8 doesn't touch it). The W4A8 token gemv
	// streams the whole model at the measured per-matrix byte ratio AND the
	// measured W4A8 GB/s (int4 unpack is more ALU-bound than int8).
	const int8TokenGemvGB, fixedMs = 1.55, 6.85
	w4GemvMs := int8TokenGemvGB * (wBytes / int8Bytes) / gbs * 1e3
	tokMs := w4GemvMs + fixedMs
	t.Logf("projected token: gemv %.2f ms (was 4.3) + fixed %.2f → %.2f ms = %.0f tok/s (int8 was 89.7)",
		w4GemvMs, fixedMs, tokMs, 1000.0/tokMs)
}

//go:build gpu

package gpu

import (
	"math/rand"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// TestRMSNormBatched_parity gates the dispatch-count fix in prefillrunner.go
// (ensurePrefillBatched's doc comment has the measured before/after): does ONE
// grid-(1,M) dispatch of rmsnormBatchedShaderWGSL match M separate grid-(1,1)
// dispatches of the existing rmsnormShaderWGSL (the kernel PrefillLastW8A8 used to
// call, once per row, before this fix)? No checkpoint needed — synthetic weights
// exercise the same reduction math.
//
//	go test -tags gpu ./gpu/ -run TestRMSNormBatched_parity -v
func TestRMSNormBatched_parity(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	for _, f := range []func() error{c.ensureLayer, c.ensurePrefillBatched} {
		if err := f(); err != nil {
			t.Fatalf("ensure: %v", err)
		}
	}

	const M, H = 37, 133 // an M not a multiple of a tidy power of 2, an H that isn't either
	rng := rand.New(rand.NewSource(1))
	src := make([]float32, M*H)
	for i := range src {
		src[i] = float32(rng.NormFloat64()) * 0.7
	}
	weight := make([]float32, H)
	for i := range weight {
		weight[i] = float32(rng.NormFloat64())*0.1 + 1.0
	}
	const eps = float32(1e-5)
	const addOne = false

	srcBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(src), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("upload src: %v", err)
	}
	defer srcBuf.Release()
	wBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(weight), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		t.Fatalf("upload weight: %v", err)
	}
	defer wBuf.Release()

	// REFERENCE: M separate (1,1) dispatches of rmsnormPipeline — reading row r of
	// src via a per-row COPY (the WGSL src buffer size must exactly match what the
	// M=1 kernel expects; a view isn't needed here since this is only the reference).
	refBuf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(M * H * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("alloc refBuf: %v", err)
	}
	defer refBuf.Release()
	{
		enc, err := c.device.TryCreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("encoder: %v", err)
		}
		defer enc.Release()
		for r := 0; r < M; r++ {
			rowSrc, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(H * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
			if err != nil {
				t.Fatalf("alloc rowSrc: %v", err)
			}
			defer rowSrc.Release()
			if err := enc.TryCopyBufferToBuffer(srcBuf, uint64(r*H*4), rowSrc, 0, uint64(H*4)); err != nil {
				t.Fatalf("copy row %d: %v", r, err)
			}
			rowOut, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(H * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
			if err != nil {
				t.Fatalf("alloc rowOut: %v", err)
			}
			defer rowOut.Release()
			p, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(H), f32bits(eps), boolU32(addOne), 0}), Usage: wgpu.BufferUsageUniform})
			if err != nil {
				t.Fatalf("alloc p: %v", err)
			}
			defer p.Release()
			bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.rmsnormLayout, Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: rowSrc, Size: rowSrc.GetSize()}, {Binding: 1, Buffer: wBuf, Size: wBuf.GetSize()},
				{Binding: 2, Buffer: rowOut, Size: rowOut.GetSize()}, {Binding: 3, Buffer: p, Size: p.GetSize()},
			}})
			if err != nil {
				t.Fatalf("bindgroup row %d: %v", r, err)
			}
			defer bg.Release()
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(c.rmsnormPipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(1, 1, 1)
			pass.TryEnd()
			pass.Release()
			if err := enc.TryCopyBufferToBuffer(rowOut, 0, refBuf, uint64(r*H*4), uint64(H*4)); err != nil {
				t.Fatalf("copy out row %d: %v", r, err)
			}
		}
		cmd, err := enc.TryFinish(nil)
		if err != nil {
			t.Fatalf("finish: %v", err)
		}
		defer cmd.Release()
		c.queue.Submit(cmd)
	}

	// UNDER TEST: one grid-(1,M) dispatch of rmsnormBatchedPipeline.
	gotBuf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(M * H * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("alloc gotBuf: %v", err)
	}
	defer gotBuf.Release()
	pB, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(H), f32bits(eps), boolU32(addOne), 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		t.Fatalf("alloc pB: %v", err)
	}
	defer pB.Release()
	bgB, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.rmsnormBatchedLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: srcBuf, Size: srcBuf.GetSize()}, {Binding: 1, Buffer: wBuf, Size: wBuf.GetSize()},
		{Binding: 2, Buffer: gotBuf, Size: gotBuf.GetSize()}, {Binding: 3, Buffer: pB, Size: pB.GetSize()},
	}})
	if err != nil {
		t.Fatalf("bindgroup batched: %v", err)
	}
	defer bgB.Release()
	{
		enc, err := c.device.TryCreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("encoder B: %v", err)
		}
		defer enc.Release()
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(c.rmsnormBatchedPipeline)
		pass.SetBindGroup(0, bgB, nil)
		pass.DispatchWorkgroups(1, uint32(M), 1)
		pass.TryEnd()
		pass.Release()
		cmd, err := enc.TryFinish(nil)
		if err != nil {
			t.Fatalf("finish B: %v", err)
		}
		defer cmd.Release()
		c.queue.Submit(cmd)
	}

	ref := readF32(t, c, refBuf, M*H)
	got := readF32(t, c, gotBuf, M*H)
	cos, maxAbs := cosine(ref, got)
	nDiff := 0
	for i := range ref {
		if ref[i] != got[i] {
			nDiff++
		}
	}
	t.Logf("M=%d H=%d: cosine=%.9f maxAbs=%.3e, %d/%d elements differ", M, H, cos, maxAbs, nDiff, len(ref))
	if nDiff != 0 {
		t.Fatalf("batched RMSNorm is not bit-identical to M sequential (1,1) dispatches: %d/%d elements differ, maxAbs=%.3e", nDiff, len(ref), maxAbs)
	}
}

// TestRoPEBatched_parity gates ropeBatchedShaderWGSL the same way: ONE
// grid-(heads*half,M) dispatch (with start, recovering row r's position as
// start+r) vs M separate ropeShaderWGSL dispatches each with its own scalar pos —
// the shape PrefillLastW8A8 used to call, once per row, before this fix.
//
//	go test -tags gpu ./gpu/ -run TestRoPEBatched_parity -v
func TestRoPEBatched_parity(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	if err := c.ensureAttn(); err != nil {
		t.Fatalf("ensureAttn: %v", err)
	}
	if err := c.ensurePrefillBatched(); err != nil {
		t.Fatalf("ensurePrefillBatched: %v", err)
	}

	const M, heads, hd, start = 23, 5, 16, 7 // hd even (RoPE needs pairs); start != 0 to catch a start-vs-0 mixup
	half := hd / 2
	rng := rand.New(rand.NewSource(2))
	vec := make([]float32, M*heads*hd)
	for i := range vec {
		vec[i] = float32(rng.NormFloat64()) * 0.9
	}
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(1.0 / (10000.0 * (float64(i) + 1) / float64(half)))
	}

	invFreqBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(invFreq), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		t.Fatalf("upload invFreq: %v", err)
	}
	defer invFreqBuf.Release()

	// REFERENCE: M separate ropePipeline dispatches, each rotating row r in place
	// with pos = start+r (mirroring the old per-row call site's positions[r]).
	refBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(vec), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("upload ref vec: %v", err)
	}
	defer refBuf.Release()
	{
		enc, err := c.device.TryCreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("encoder: %v", err)
		}
		defer enc.Release()
		for r := 0; r < M; r++ {
			rowBuf, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(heads * hd * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
			if err != nil {
				t.Fatalf("alloc rowBuf: %v", err)
			}
			defer rowBuf.Release()
			if err := enc.TryCopyBufferToBuffer(refBuf, uint64(r*heads*hd*4), rowBuf, 0, uint64(heads*hd*4)); err != nil {
				t.Fatalf("copy row %d in: %v", r, err)
			}
			p, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(heads), uint32(hd), uint32(half), uint32(start + r), f32bits(1), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
			if err != nil {
				t.Fatalf("alloc p row %d: %v", r, err)
			}
			defer p.Release()
			bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.ropeLayout, Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: rowBuf, Size: rowBuf.GetSize()}, {Binding: 1, Buffer: invFreqBuf, Size: invFreqBuf.GetSize()}, {Binding: 2, Buffer: p, Size: p.GetSize()},
			}})
			if err != nil {
				t.Fatalf("bindgroup row %d: %v", r, err)
			}
			defer bg.Release()
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(c.ropePipeline)
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(uint32(heads*half+63)/64, 1, 1)
			pass.TryEnd()
			pass.Release()
			if err := enc.TryCopyBufferToBuffer(rowBuf, 0, refBuf, uint64(r*heads*hd*4), uint64(heads*hd*4)); err != nil {
				t.Fatalf("copy row %d out: %v", r, err)
			}
		}
		cmd, err := enc.TryFinish(nil)
		if err != nil {
			t.Fatalf("finish: %v", err)
		}
		defer cmd.Release()
		c.queue.Submit(cmd)
	}

	// UNDER TEST: one grid-(ceil(heads*half/64),M) dispatch of ropeBatchedPipeline.
	gotBuf, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(vec), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("upload got vec: %v", err)
	}
	defer gotBuf.Release()
	pB, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes([]uint32{uint32(heads), uint32(hd), uint32(half), uint32(start), f32bits(1), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		t.Fatalf("alloc pB: %v", err)
	}
	defer pB.Release()
	bgB, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.ropeBatchedLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: gotBuf, Size: gotBuf.GetSize()}, {Binding: 1, Buffer: invFreqBuf, Size: invFreqBuf.GetSize()}, {Binding: 2, Buffer: pB, Size: pB.GetSize()},
	}})
	if err != nil {
		t.Fatalf("bindgroup batched: %v", err)
	}
	defer bgB.Release()
	{
		enc, err := c.device.TryCreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("encoder B: %v", err)
		}
		defer enc.Release()
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(c.ropeBatchedPipeline)
		pass.SetBindGroup(0, bgB, nil)
		pass.DispatchWorkgroups(uint32(heads*half+63)/64, uint32(M), 1)
		pass.TryEnd()
		pass.Release()
		cmd, err := enc.TryFinish(nil)
		if err != nil {
			t.Fatalf("finish B: %v", err)
		}
		defer cmd.Release()
		c.queue.Submit(cmd)
	}

	ref := readF32(t, c, refBuf, M*heads*hd)
	got := readF32(t, c, gotBuf, M*heads*hd)
	cos, maxAbs := cosine(ref, got)
	nDiff := 0
	for i := range ref {
		if ref[i] != got[i] {
			nDiff++
		}
	}
	t.Logf("M=%d heads=%d hd=%d start=%d: cosine=%.9f maxAbs=%.3e, %d/%d elements differ", M, heads, hd, start, cos, maxAbs, nDiff, len(ref))
	if nDiff != 0 {
		t.Fatalf("batched RoPE is not bit-identical to M sequential per-row dispatches: %d/%d elements differ, maxAbs=%.3e", nDiff, len(ref), maxAbs)
	}
}

// readF32 maps and reads back n float32s from a storage buffer, then unmaps it.
func readF32(t *testing.T, c *Context, buf *wgpu.Buffer, n int) []float32 {
	t.Helper()
	sz := uint64(n * 4)
	stag, err := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: sz, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("alloc staging: %v", err)
	}
	defer stag.Release()
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("encoder: %v", err)
	}
	defer enc.Release()
	if err := enc.TryCopyBufferToBuffer(buf, 0, stag, 0, sz); err != nil {
		t.Fatalf("copy: %v", err)
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	status := wgpu.MapAsyncStatus(0)
	if err := stag.TryMapAsync(wgpu.MapModeRead, 0, sz, func(s wgpu.MapAsyncStatus) { status = s }); err != nil {
		t.Fatalf("map: %v", err)
	}
	c.device.Poll(true, nil)
	if status != wgpu.MapAsyncStatusSuccess {
		t.Fatalf("map failed: %v", status)
	}
	out := make([]float32, n)
	copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(sz))))
	if err := stag.TryUnmap(); err != nil {
		t.Fatalf("unmap: %v", err)
	}
	return out
}

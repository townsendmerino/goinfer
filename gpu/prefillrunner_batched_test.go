//go:build gpu

package gpu

import (
	"math/rand"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// TestRoPEBatched_parity gates ropeBatchedShaderWGSL: ONE
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
		for r := range M {
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

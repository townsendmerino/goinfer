//go:build goinfer_testhooks

// Code relocated by the B-08 build-tag pass: these are test-only hooks, compiled
// only under -tags goinfer_testhooks so they are NOT part of the public API
// (audit B-08). See RELEASING.md. Imports are added to satisfy the moved bodies.

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// RouteExpertsForTest runs the router top-k selection kernel standalone (upload logits,
// dispatch, read back) — the C3a unit-test seam mirroring decoder.routeExperts for
// nGroup==1. bias may be nil. Returns the k chosen expert indices and their weights.
func (c *Context) RouteExpertsForTest(logits, bias []float32, k int, sigmoid, norm bool, scale float32) ([]int, []float32, error) {
	if err := c.ensureMoERoute(); err != nil {
		return nil, nil, err
	}
	nE := len(logits)
	lb, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-logits", Contents: wgpu.ToBytes(logits), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, nil, err
	}
	defer lb.Release()
	hasBias := uint32(0)
	bb := bias
	if bb == nil {
		bb = make([]float32, nE) // a bound (unused) buffer; hasBias=0 keeps it out of the math
	} else {
		hasBias = 1
	}
	bbuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-bias", Contents: wgpu.ToBytes(bb), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, nil, err
	}
	defer bbuf.Release()
	idxBuf, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "moe-idx", Size: uint64(k * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer idxBuf.Release()
	wgtBuf, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "moe-wgt", Size: uint64(k * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer wgtBuf.Release()
	sig, nrm := uint32(0), uint32(0)
	if sigmoid {
		sig = 1
	}
	if norm {
		nrm = 1
	}
	pbuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "moe-p", Contents: wgpu.ToBytes([]uint32{uint32(nE), uint32(k), sig, nrm, f32bits(scale), hasBias, 0, 0}), Usage: wgpu.BufferUsageUniform})
	defer pbuf.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeRouteLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: lb, Size: lb.GetSize()}, {Binding: 1, Buffer: bbuf, Size: bbuf.GetSize()},
		{Binding: 2, Buffer: idxBuf, Size: idxBuf.GetSize()}, {Binding: 3, Buffer: wgtBuf, Size: wgtBuf.GetSize()},
		{Binding: 4, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return nil, nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeRoutePipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(1, 1, 1)
	pass.End()
	pass.Release()
	idxStg, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(k * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer idxStg.Release()
	wgtStg, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(k * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer wgtStg.Release()
	enc.CopyBufferToBuffer(idxBuf, 0, idxStg, 0, uint64(k*4))
	enc.CopyBufferToBuffer(wgtBuf, 0, wgtStg, 0, uint64(k*4))
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var si, sw wgpu.BufferMapAsyncStatus
	if err := idxStg.MapAsync(wgpu.MapModeRead, 0, uint64(k*4), func(s wgpu.BufferMapAsyncStatus) { si = s }); err != nil {
		return nil, nil, err
	}
	if err := wgtStg.MapAsync(wgpu.MapModeRead, 0, uint64(k*4), func(s wgpu.BufferMapAsyncStatus) { sw = s }); err != nil {
		return nil, nil, err
	}
	c.device.Poll(true, nil)
	if si != wgpu.BufferMapAsyncStatusSuccess || sw != wgpu.BufferMapAsyncStatusSuccess {
		return nil, nil, fmt.Errorf("gpu: moe route map failed: idx=%v wgt=%v", si, sw)
	}
	idx := make([]int, k)
	for i, v := range wgpu.FromBytes[uint32](idxStg.GetMappedRange(0, uint(k*4))) {
		idx[i] = int(v)
	}
	idxStg.Unmap()
	wts := make([]float32, k)
	copy(wts, wgpu.FromBytes[float32](wgtStg.GetMappedRange(0, uint(k*4))))
	wgtStg.Unmap()
	return idx, wts, nil
}

// IndexedGEMVForTest runs ONE indexed expert GEMV standalone (overwrite mode): dst[N] =
// (aq·expert[idx[slot]]ᵀ). The C3b unit-test seam — isolates the stacked-index addressing
// from the full MoE block (C3c). aq is the quantized activation [K], aScale its scale.
func (c *Context) IndexedGEMVForTest(s *ResidentStackedW8A8, aq []int8, aScale float32, idx []int, slot int) ([]float32, error) {
	if err := c.ensureMoEExpert(); err != nil {
		return nil, err
	}
	N := s.rows
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-act", Contents: wgpu.ToBytes(packInt8(aq, 1, s.cols)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer aBuf.Release()
	asBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-as", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage})
	defer asBuf.Release()
	idxU := make([]uint32, len(idx))
	for i, v := range idx {
		idxU[i] = uint32(v)
	}
	idxBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-idx", Contents: wgpu.ToBytes(idxU), Usage: wgpu.BufferUsageStorage})
	defer idxBuf.Release()
	wgtBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-wgt", Contents: wgpu.ToBytes(make([]float32, len(idx))), Usage: wgpu.BufferUsageStorage})
	defer wgtBuf.Release()
	dstBuf, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "ig-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer dstBuf.Release()
	dims, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig-dims", Contents: wgpu.ToBytes([]uint32{uint32(s.kp), uint32(N), uint32(slot), 0}), Usage: wgpu.BufferUsageUniform})
	defer dims.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeExpertLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: s.bq, Size: s.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: s.bScales, Size: s.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()}, {Binding: 5, Buffer: idxBuf, Size: idxBuf.GetSize()},
		{Binding: 6, Buffer: wgtBuf, Size: wgtBuf.GetSize()}, {Binding: 7, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeExpertPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := gemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.End()
	pass.Release()
	stg, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stg.Release()
	enc.CopyBufferToBuffer(dstBuf, 0, stg, 0, uint64(N*4))
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var st wgpu.BufferMapAsyncStatus
	if err := stg.MapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.BufferMapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: indexed gemv map failed: %v", st)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stg.GetMappedRange(0, uint(N*4))))
	stg.Unmap()
	return out, nil
}

// IndexedGEMVForTestInt4 runs ONE indexed int4 expert GEMV standalone (overwrite
// mode): dst[N] = aScale · (aq · dequant(expert[idx[slot]])ᵀ). The W4A8 analog of
// IndexedGEMVForTest — the parity seam for the stacked-int4 path.
func (c *Context) IndexedGEMVForTestInt4(s *ResidentStackedW8A8, aq []int8, aScale float32, idx []int, slot int) ([]float32, error) {
	if !s.w4 {
		return nil, fmt.Errorf("gpu: IndexedGEMVForTestInt4 on a non-w4 stack")
	}
	if err := c.ensureMoEExpertW4(); err != nil {
		return nil, err
	}
	N := s.rows
	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig4-act", Contents: wgpu.ToBytes(packInt8(aq, 1, s.cols)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer aBuf.Release()
	asBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig4-as", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage})
	defer asBuf.Release()
	idxU := make([]uint32, len(idx))
	for i, v := range idx {
		idxU[i] = uint32(v)
	}
	idxBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig4-idx", Contents: wgpu.ToBytes(idxU), Usage: wgpu.BufferUsageStorage})
	defer idxBuf.Release()
	wgtBuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig4-wgt", Contents: wgpu.ToBytes(make([]float32, len(idx))), Usage: wgpu.BufferUsageStorage})
	defer wgtBuf.Release()
	dstBuf, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "ig4-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer dstBuf.Release()
	dims, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "ig4-dims", Contents: wgpu.ToBytes([]uint32{uint32(s.kp), uint32(N), uint32(slot), 0}), Usage: wgpu.BufferUsageUniform})
	defer dims.Release()
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeExpertW4Layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: s.bq, Size: s.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: s.bScales, Size: s.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()}, {Binding: 5, Buffer: idxBuf, Size: idxBuf.GetSize()},
		{Binding: 6, Buffer: wgtBuf, Size: wgtBuf.GetSize()}, {Binding: 7, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeExpertW4Pipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := gemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.End()
	pass.Release()
	stg, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stg.Release()
	enc.CopyBufferToBuffer(dstBuf, 0, stg, 0, uint64(N*4))
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var st wgpu.BufferMapAsyncStatus
	if err := stg.MapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.BufferMapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: indexed w4 gemv map failed: %v", st)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stg.GetMappedRange(0, uint(N*4))))
	stg.Unmap()
	return out, nil
}

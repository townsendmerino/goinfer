//go:build gpu && goinfer_testhooks

// Test-only hooks, compiled only under -tags goinfer_testhooks so they are NOT part of the public API (see RELEASING.md).
//
// HAND-MAINTAINED, NOT MACHINE-GENERATED: there is no //go:generate directive and nothing regenerates this file; edit it
// directly like any other file. Mirrors decoder/testhooks.go and cuda/testhooks.go.

package kernels

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
	"sort"
	"time"
	"weak"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// RouteExpertsForTest runs the router top-k selection kernel standalone (upload logits,
// dispatch, read back) — the C3a unit-test seam mirroring decoder.routeExperts for
// nGroup==1. bias may be nil. Returns the k chosen expert indices and their weights.
func (c *Context) RouteExpertsForTest(logits, bias []float32, k int, sigmoid, norm bool, scale float32) ([]int, []float32, error) {
	if err := c.ensureMoERoute(); err != nil {
		return nil, nil, err
	}
	nE := len(logits)
	lb, err := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "moe-logits", Contents: wgpu.ToBytes(logits), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, nil, err
	}
	defer releaseBuf(lb)
	hasBias := uint32(0)
	bb := bias
	if bb == nil {
		bb = make([]float32, nE) // a bound (unused) buffer; hasBias=0 keeps it out of the math
	} else {
		hasBias = 1
	}
	bbuf, err := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "moe-bias", Contents: wgpu.ToBytes(bb), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, nil, err
	}
	defer releaseBuf(bbuf)
	idxBuf, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Label: "moe-idx", Size: uint64(k * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer releaseBuf(idxBuf)
	wgtBuf, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Label: "moe-wgt", Size: uint64(k * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer releaseBuf(wgtBuf)
	sig, nrm := uint32(0), uint32(0)
	if sigmoid {
		sig = 1
	}
	if norm {
		nrm = 1
	}
	pbuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "moe-p", Contents: wgpu.ToBytes([]uint32{uint32(nE), uint32(k), sig, nrm, F32bits(scale), hasBias, 0, 0}), Usage: wgpu.BufferUsageUniform})
	defer releaseBuf(pbuf)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeRouteLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: lb, Size: lb.GetSize()}, {Binding: 1, Buffer: bbuf, Size: bbuf.GetSize()},
		{Binding: 2, Buffer: idxBuf, Size: idxBuf.GetSize()}, {Binding: 3, Buffer: wgtBuf, Size: wgtBuf.GetSize()},
		{Binding: 4, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		return nil, nil, err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeRoutePipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(1, 1, 1)
	pass.TryEnd()
	pass.Release()
	idxStg, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Size: uint64(k * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer releaseBuf(idxStg)
	wgtStg, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Size: uint64(k * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer releaseBuf(wgtStg)
	enc.TryCopyBufferToBuffer(idxBuf, 0, idxStg, 0, uint64(k*4))
	enc.TryCopyBufferToBuffer(wgtBuf, 0, wgtStg, 0, uint64(k*4))
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var si, sw wgpu.MapAsyncStatus
	if err := idxStg.TryMapAsync(wgpu.MapModeRead, 0, uint64(k*4), func(s wgpu.MapAsyncStatus) { si = s }); err != nil {
		return nil, nil, err
	}
	if err := wgtStg.TryMapAsync(wgpu.MapModeRead, 0, uint64(k*4), func(s wgpu.MapAsyncStatus) { sw = s }); err != nil {
		return nil, nil, err
	}
	c.device.Poll(true, nil)
	if si != wgpu.MapAsyncStatusSuccess || sw != wgpu.MapAsyncStatusSuccess {
		return nil, nil, fmt.Errorf("gpu: moe route map failed: idx=%v wgt=%v", si, sw)
	}
	idx := make([]int, k)
	for i, v := range wgpu.FromBytes[uint32](idxStg.GetMappedRange(0, uint(k*4))) {
		idx[i] = int(v)
	}
	idxStg.TryUnmap()
	wts := make([]float32, k)
	copy(wts, wgpu.FromBytes[float32](wgtStg.GetMappedRange(0, uint(k*4))))
	wgtStg.TryUnmap()
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
	aBuf, err := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig-act", Contents: wgpu.ToBytes(packInt8(aq, 1, s.cols)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(aBuf)
	asBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig-as", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(asBuf)
	idxU := make([]uint32, len(idx))
	for i, v := range idx {
		idxU[i] = uint32(v)
	}
	idxBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig-idx", Contents: wgpu.ToBytes(idxU), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(idxBuf)
	wgtBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig-wgt", Contents: wgpu.ToBytes(make([]float32, len(idx))), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(wgtBuf)
	dstBuf, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Label: "ig-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer releaseBuf(dstBuf)
	dims, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig-dims", Contents: wgpu.ToBytes([]uint32{uint32(s.kp), uint32(N), uint32(slot), 0}), Usage: wgpu.BufferUsageUniform})
	defer releaseBuf(dims)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeExpertLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: s.bq, Size: s.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: s.bScales, Size: s.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()}, {Binding: 5, Buffer: idxBuf, Size: idxBuf.GetSize()},
		{Binding: 6, Buffer: wgtBuf, Size: wgtBuf.GetSize()}, {Binding: 7, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeExpertPipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := GemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.TryEnd()
	pass.Release()
	stg, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer releaseBuf(stg)
	enc.TryCopyBufferToBuffer(dstBuf, 0, stg, 0, uint64(N*4))
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var st wgpu.MapAsyncStatus
	if err := stg.TryMapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.MapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: indexed gemv map failed: %v", st)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stg.GetMappedRange(0, uint(N*4))))
	stg.TryUnmap()
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
	aBuf, err := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig4-act", Contents: wgpu.ToBytes(packInt8(aq, 1, s.cols)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(aBuf)
	asBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig4-as", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(asBuf)
	idxU := make([]uint32, len(idx))
	for i, v := range idx {
		idxU[i] = uint32(v)
	}
	idxBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig4-idx", Contents: wgpu.ToBytes(idxU), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(idxBuf)
	wgtBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig4-wgt", Contents: wgpu.ToBytes(make([]float32, len(idx))), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(wgtBuf)
	dstBuf, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Label: "ig4-dst", Size: uint64(N * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer releaseBuf(dstBuf)
	dims, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "ig4-dims", Contents: wgpu.ToBytes([]uint32{uint32(s.kp), uint32(N), uint32(slot), 0}), Usage: wgpu.BufferUsageUniform})
	defer releaseBuf(dims)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.moeExpertW4Layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: s.bq, Size: s.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: s.bScales, Size: s.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()}, {Binding: 5, Buffer: idxBuf, Size: idxBuf.GetSize()},
		{Binding: 6, Buffer: wgtBuf, Size: wgtBuf.GetSize()}, {Binding: 7, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.moeExpertW4Pipeline)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := GemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.TryEnd()
	pass.Release()
	stg, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer releaseBuf(stg)
	enc.TryCopyBufferToBuffer(dstBuf, 0, stg, 0, uint64(N*4))
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var st wgpu.MapAsyncStatus
	if err := stg.TryMapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.MapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: indexed w4 gemv map failed: %v", st)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stg.GetMappedRange(0, uint(N*4))))
	stg.TryUnmap()
	return out, nil
}

// GptOssDownForTest runs gpt-oss's down-projection combine standalone for one router slot:
// dst[n] = dstInit[n] + wgt[slot]·(aScale·(aq·dequant(expert[idx[slot]])ᵀ)[n] + dbias[idx[slot]·N+n]).
// It dispatches whatever gptOssDownPipelineFor selects for s, which is the resident builder's own choice, so a stack routed to
// the wrong kernel fails here exactly as it does in decode.
func (c *Context) GptOssDownForTest(s *ResidentStackedW8A8, aq []int8, aScale float32, idx []int, wgt, dbias, dstInit []float32, slot int) ([]float32, error) {
	if err := c.ensureMoEExpertGptOssDown(); err != nil {
		return nil, err
	}
	if err := c.ensureMoEExpertGptOssDownW4(); err != nil {
		return nil, err
	}
	N := s.rows
	pl, ly := c.gptOssDownPipelineFor(s)
	aBuf, err := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "gd-act", Contents: wgpu.ToBytes(packInt8(aq, 1, s.cols)), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return nil, err
	}
	defer releaseBuf(aBuf)
	asBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "gd-as", Contents: wgpu.ToBytes([]float32{aScale}), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(asBuf)
	idxU := make([]uint32, len(idx))
	for i, v := range idx {
		idxU[i] = uint32(v)
	}
	idxBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "gd-idx", Contents: wgpu.ToBytes(idxU), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(idxBuf)
	wgtBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "gd-wgt", Contents: wgpu.ToBytes(wgt), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(wgtBuf)
	dbBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "gd-dbias", Contents: wgpu.ToBytes(dbias), Usage: wgpu.BufferUsageStorage})
	defer releaseBuf(dbBuf)
	dstBuf, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "gd-dst", Contents: wgpu.ToBytes(dstInit), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer releaseBuf(dstBuf)
	dims, _ := c.tryCreateBufferInit(mem.LineTest, &wgpu.BufferInitDescriptor{Label: "gd-dims", Contents: wgpu.ToBytes([]uint32{uint32(s.kp), uint32(N), uint32(slot), 0}), Usage: wgpu.BufferUsageUniform})
	defer releaseBuf(dims)
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: ly, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()}, {Binding: 1, Buffer: s.bq, Size: s.bq.GetSize()},
		{Binding: 2, Buffer: asBuf, Size: asBuf.GetSize()}, {Binding: 3, Buffer: s.bScales, Size: s.bScales.GetSize()},
		{Binding: 4, Buffer: dstBuf, Size: dstBuf.GetSize()}, {Binding: 5, Buffer: idxBuf, Size: idxBuf.GetSize()},
		{Binding: 6, Buffer: wgtBuf, Size: wgtBuf.GetSize()}, {Binding: 7, Buffer: dbBuf, Size: dbBuf.GetSize()},
		{Binding: 8, Buffer: dims, Size: dims.GetSize()},
	}})
	if err != nil {
		return nil, err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	gx, gy := GemvGrid(N)
	pass.DispatchWorkgroups(gx, gy, 1)
	pass.TryEnd()
	pass.Release()
	stg, _ := c.tryCreateBuffer(mem.LineTest, &wgpu.BufferDescriptor{Size: uint64(N * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer releaseBuf(stg)
	enc.TryCopyBufferToBuffer(dstBuf, 0, stg, 0, uint64(N*4))
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	var st wgpu.MapAsyncStatus
	if err := stg.TryMapAsync(wgpu.MapModeRead, 0, uint64(N*4), func(s wgpu.MapAsyncStatus) { st = s }); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.MapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: gpt-oss down map failed: %v", st)
	}
	out := make([]float32, N)
	copy(out, wgpu.FromBytes[float32](stg.GetMappedRange(0, uint(N*4))))
	stg.TryUnmap()
	return out, nil
}

// LMHeadForTest returns the runner's LM-head weight (decodeWeight), for tests that need to call the same GEMV kernel
// directly against the model's own resident buffer; see gemv_lmhead_isolation_test.go.
func (r *DecodeRunner) LMHeadForTest() DecodeWeight { return r.lmHead }

// SetPrefillProfForTest enables (on=true) or disables the R10 prefill decomposition profiler
// (gpu/prefill_prof.go) and resets its accumulators. See PrefillProfForTest.
func (c *Context) SetPrefillProfForTest(on bool) {
	if on {
		c.prefillProf = &PrefillProf{}
	} else {
		c.prefillProf = nil
	}
}

// PrefillProfForTest reports the accumulated per-category wall time since the profiler was last
// enabled/reset. Zero values if profiling was never turned on.
func (c *Context) PrefillProfForTest() (gemm, attn, normsRope, kvWrite time.Duration) {
	if c.prefillProf == nil {
		return 0, 0, 0, 0
	}
	return c.prefillProf.Gemm, c.prefillProf.Attn, c.prefillProf.NormsRope, c.prefillProf.KvWrite
}

// LeasedBufferForTest is one live buffer leased on a Context: its bytes, its wgpu usage flags, its plan line and its place in
// the process's creation order.
type LeasedBufferForTest struct {
	Bytes int64
	Usage uint64
	Line  string
	Seq   int64
}

// LeasedBuffersForTest lists c's live leased buffers in creation order: the buffer list a model of the driver's
// allocations predicts from (SB2-vk). A buffer the GC has already collected is left out.
func (c *Context) LeasedBuffersForTest() []LeasedBufferForTest {
	var out []LeasedBufferForTest
	bufLeases.Range(func(k, v any) bool {
		e := v.(leaseEntry)
		if e.c != c {
			return true
		}
		b := k.(weak.Pointer[wgpu.Buffer]).Value()
		if b == nil {
			return true
		}
		out = append(out, LeasedBufferForTest{Bytes: e.n, Usage: uint64(b.GetUsage()), Line: e.line, Seq: e.seq})
		return true
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out
}

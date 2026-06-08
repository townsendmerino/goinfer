//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// Device (submit-no-poll) variants of the layer ops + the attention sub-block, so
// a decode token's attention half records into the running command stream with no
// CPU interleave (the MLP half is FusedMLP). The KV cache lives in device buffers
// (KCache/VCache per layer); each token appends its k/v and attention reads the
// whole window. Validated against the staged composition of the already-CPU-parity
// ops (TestAttnBlock_parity).

// rmsnormDevice: RMSNorm(x)·weight → new device buffer (submit, no poll).
func (c *Context) rmsnormDevice(x, weight *DeviceBuffer, H int, eps float32, addOne bool) (*DeviceBuffer, []func(), error) {
	if err := c.ensureLayer(); err != nil {
		return nil, nil, err
	}
	out, err := c.newF32("rmsn", H)
	if err != nil {
		return nil, nil, err
	}
	pbuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "rmsn-p", Contents: wgpu.ToBytes([]uint32{uint32(H), f32bits(eps), boolU32(addOne), 0}), Usage: wgpu.BufferUsageUniform})
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.rmsnormLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: x.buf, Size: x.buf.GetSize()}, {Binding: 1, Buffer: weight.buf, Size: weight.buf.GetSize()},
		{Binding: 2, Buffer: out, Size: out.GetSize()}, {Binding: 3, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		out.Release()
		pbuf.Release()
		return nil, nil, err
	}
	if err := c.submitUnary(c.rmsnormPipeline, bg, 64); err != nil {
		return nil, nil, err
	}
	free := []func(){func() { out.Release() }, pbuf.Release, bg.Release}
	return &DeviceBuffer{buf: out, n: H}, free, nil
}

// residualInPlace: x += y (submit, no poll).
func (c *Context) residualInPlace(x, y *DeviceBuffer, H int) ([]func(), error) {
	if err := c.ensureLayer(); err != nil {
		return nil, err
	}
	pbuf, _ := c.dims4("res-p", uint32(H), 0)
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.residualLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: x.buf, Size: x.buf.GetSize()}, {Binding: 1, Buffer: y.buf, Size: y.buf.GetSize()},
		{Binding: 2, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		pbuf.Release()
		return nil, err
	}
	if err := c.submitUnary(c.residualPipeline, bg, H); err != nil {
		return nil, err
	}
	return []func(){pbuf.Release, bg.Release}, nil
}

// ropeInPlace rotates a device vec [heads*headDim] at pos (submit, no poll).
func (c *Context) ropeInPlace(vec, invFreq *DeviceBuffer, heads, headDim, pos int, scale float32) ([]func(), error) {
	if err := c.ensureAttn(); err != nil {
		return nil, err
	}
	half := headDim / 2
	pbuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "rope-p", Contents: wgpu.ToBytes([]uint32{uint32(heads), uint32(headDim), uint32(half), uint32(pos), f32bits(scale), 0, 0, 0}), Usage: wgpu.BufferUsageUniform})
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.ropeLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: vec.buf, Size: vec.buf.GetSize()}, {Binding: 1, Buffer: invFreq.buf, Size: invFreq.buf.GetSize()},
		{Binding: 2, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		pbuf.Release()
		return nil, err
	}
	if err := c.submitUnary(c.ropePipeline, bg, heads*half); err != nil {
		return nil, err
	}
	return []func(){pbuf.Release, bg.Release}, nil
}

// kvAppend copies a device k or v [kvDim] into the cache at position pos (a copy
// command, submitted; no poll).
func (c *Context) kvAppend(src, cache *DeviceBuffer, pos, kvDim int) error {
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	if err := enc.CopyBufferToBuffer(src.buf, 0, cache.buf, uint64(pos*kvDim*4), uint64(kvDim*4)); err != nil {
		return err
	}
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// attnDevice: single-query attention over the resident KV cache (submit, no poll).
func (c *Context) attnDevice(q, kCache, vCache *DeviceBuffer, nH, nKV, hd, nKeys, start int, scale float32) (*DeviceBuffer, []func(), error) {
	if err := c.ensureAttn(); err != nil {
		return nil, nil, err
	}
	group := nH / nKV
	ctxBuf, err := c.newF32("attn-ctx", nH*hd)
	if err != nil {
		return nil, nil, err
	}
	pbuf, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "attn-p", Contents: wgpu.ToBytes([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(nKeys), uint32(start), uint32(group), f32bits(scale), 0}), Usage: wgpu.BufferUsageUniform})
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.attnLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: q.buf, Size: q.buf.GetSize()}, {Binding: 1, Buffer: kCache.buf, Size: kCache.buf.GetSize()},
		{Binding: 2, Buffer: vCache.buf, Size: vCache.buf.GetSize()}, {Binding: 3, Buffer: ctxBuf, Size: ctxBuf.GetSize()},
		{Binding: 4, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		ctxBuf.Release()
		pbuf.Release()
		return nil, nil, err
	}
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.attnPipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nH), 1, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		ctxBuf.Release()
		pbuf.Release()
		bg.Release()
		return nil, nil, err
	}
	pass.Release()
	cmd, _ := enc.Finish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	free := []func(){func() { ctxBuf.Release() }, pbuf.Release, bg.Release}
	return &DeviceBuffer{buf: ctxBuf, n: nH * hd}, free, nil
}

// AttnWeights bundles a layer's attention weights + norm (already resident).
type AttnWeights struct {
	Norm                *DeviceBuffer // RMSNorm weight [hidden]
	QProj, KProj, VProj *ResidentW8A8 // int8 projections
	OProj               *ResidentW8A8 //
	InvFreq             *DeviceBuffer // [headDim/2]
	KCache, VCache      *DeviceBuffer // [maxLen*kvDim] resident
}

// AttnBlock runs the attention sub-block for one decode token entirely on device,
// syncing once: x += OProj(attn(rope(QKV(rmsnorm(x))))). x is updated in place;
// returns the (read-back) result. pos is the new token's absolute position;
// nKeys = pos+1; start = the window start (0 for full attention).
func (c *Context) AttnBlock(x []float32, w AttnWeights, hidden, nH, nKV, hd, pos, start int, eps, scale float32, addOne bool) ([]float32, error) {
	var frees []func()
	defer func() {
		for _, f := range frees {
			f()
		}
	}()
	keep := func(fs []func(), err error) error {
		frees = append(frees, fs...)
		return err
	}
	kvDim := nKV * hd

	xd, err := c.UploadF32(x)
	if err != nil {
		return nil, err
	}
	frees = append(frees, func() { xd.Release() })

	xn, fs, err := c.rmsnormDevice(xd, w.Norm, hidden, eps, addOne)
	if err := keep(fs, err); err != nil {
		return nil, err
	}
	qb, sb, err := c.quantizeDevice(xn, 1, hidden)
	if err != nil {
		return nil, err
	}
	frees = append(frees, func() { qb.Release() }, func() { sb.Release() })
	q, err := c.matmulW8A8Device(qb, sb, w.QProj, 1)
	if err != nil {
		return nil, err
	}
	k, err := c.matmulW8A8Device(qb, sb, w.KProj, 1)
	if err != nil {
		return nil, err
	}
	v, err := c.matmulW8A8Device(qb, sb, w.VProj, 1)
	if err != nil {
		return nil, err
	}
	frees = append(frees, func() { q.Release() }, func() { k.Release() }, func() { v.Release() })

	if err := keep(c.ropeInPlace(q, w.InvFreq, nH, hd, pos, 1)); err != nil {
		return nil, err
	}
	if err := keep(c.ropeInPlace(k, w.InvFreq, nKV, hd, pos, 1)); err != nil {
		return nil, err
	}
	if err := c.kvAppend(k, w.KCache, pos, kvDim); err != nil {
		return nil, err
	}
	if err := c.kvAppend(v, w.VCache, pos, kvDim); err != nil {
		return nil, err
	}

	ctxv, fs2, err := c.attnDevice(q, w.KCache, w.VCache, nH, nKV, hd, pos+1, start, scale)
	if err := keep(fs2, err); err != nil {
		return nil, err
	}
	cq, cs, err := c.quantizeDevice(ctxv, 1, nH*hd)
	if err != nil {
		return nil, err
	}
	frees = append(frees, func() { cq.Release() }, func() { cs.Release() })
	attnOut, err := c.matmulW8A8Device(cq, cs, w.OProj, 1)
	if err != nil {
		return nil, err
	}
	frees = append(frees, func() { attnOut.Release() })
	if err := keep(c.residualInPlace(xd, attnOut, hidden)); err != nil {
		return nil, err
	}
	return c.Readback(xd) // the single sync
}

// NewKVCache creates a resident KV cache buffer of capElems f32 with the first
// len(initial) prefilled (prior positions). CopyDst lets kvAppend write into it.
func (c *Context) NewKVCache(initial []float32, capElems int) (*DeviceBuffer, error) {
	buf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "kvcache", Size: uint64(capElems * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	if len(initial) > 0 {
		if err := c.queue.WriteBuffer(buf, 0, wgpu.ToBytes(initial)); err != nil {
			buf.Release()
			return nil, err
		}
	}
	return &DeviceBuffer{buf: buf, n: capElems}, nil
}

var _ = fmt.Sprint

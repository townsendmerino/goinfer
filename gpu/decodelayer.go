//go:build gpu

package gpu

import (
	"fmt"
	"math"

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
	return newDeviceBuffer(out, H), free, nil
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
	return newDeviceBuffer(ctxBuf, nH*hd), free, nil
}

// AttnWeights bundles a layer's attention weights + norm (already resident).
type AttnWeights struct {
	Norm                *DeviceBuffer // RMSNorm weight [hidden]
	QProj, KProj, VProj *ResidentW8A8 // int8 projections
	OProj               *ResidentW8A8 //
	InvFreq             *DeviceBuffer // [headDim/2]
	KCache, VCache      *DeviceBuffer // [maxLen*kvDim] resident
}

// attnBlockInto runs the attention sub-block on the device buffer xd IN PLACE
// (xd += OProj(attn(rope(QKV(rmsnorm(xd)))))), submitting with no Poll. It returns
// the cleanup funcs for its scratch — the caller must NOT run them until after the
// final fence (the buffers are read by queued-but-unexecuted dispatches).
func (c *Context) attnBlockInto(xd *DeviceBuffer, w AttnWeights, hidden, nH, nKV, hd, pos, start int, eps, scale float32, addOne bool) ([]func(), error) {
	var frees []func()
	keep := func(fs []func(), err error) error {
		frees = append(frees, fs...)
		return err
	}
	kvDim := nKV * hd

	xn, fs, err := c.rmsnormDevice(xd, w.Norm, hidden, eps, addOne)
	if err := keep(fs, err); err != nil {
		return nil, err
	}
	qb, sb, err := c.quantizeDevice(xn, 1, hidden)
	if err != nil {
		return nil, err
	}
	frees = append(frees, func() { qb.Close() }, func() { sb.Close() })
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
	frees = append(frees, func() { q.Close() }, func() { k.Close() }, func() { v.Close() })

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
	frees = append(frees, func() { cq.Close() }, func() { cs.Close() })
	attnOut, err := c.matmulW8A8Device(cq, cs, w.OProj, 1)
	if err != nil {
		return nil, err
	}
	frees = append(frees, func() { attnOut.Close() })
	if err := keep(c.residualInPlace(xd, attnOut, hidden)); err != nil {
		return nil, err
	}
	return frees, nil
}

// AttnBlock is the standalone (host in/out) wrapper around attnBlockInto: upload,
// run, read back (the single sync), free. Used by the parity test.
func (c *Context) AttnBlock(x []float32, w AttnWeights, hidden, nH, nKV, hd, pos, start int, eps, scale float32, addOne bool) ([]float32, error) {
	xd, err := c.UploadF32(x)
	if err != nil {
		return nil, err
	}
	defer xd.Close()
	frees, err := c.attnBlockInto(xd, w, hidden, nH, nKV, hd, pos, start, eps, scale, addOne)
	if err != nil {
		return nil, err
	}
	out, err := c.Readback(xd) // the single sync
	for _, f := range frees {
		f()
	}
	return out, err
}

// swigluDevice: silu(gate)·up → new device buffer (submit, no poll).
func (c *Context) swigluDevice(gate, up *DeviceBuffer, inter int) (*DeviceBuffer, []func(), error) {
	if err := c.ensureLayer(); err != nil {
		return nil, nil, err
	}
	mid, err := c.newF32("swiglu-mid", inter)
	if err != nil {
		return nil, nil, err
	}
	pbuf, _ := c.dims4("swiglu-p", uint32(inter), 0)
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: c.swigluLayout, Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: gate.buf, Size: gate.buf.GetSize()}, {Binding: 1, Buffer: up.buf, Size: up.buf.GetSize()},
		{Binding: 2, Buffer: mid, Size: mid.GetSize()}, {Binding: 3, Buffer: pbuf, Size: pbuf.GetSize()},
	}})
	if err != nil {
		mid.Release()
		pbuf.Release()
		return nil, nil, err
	}
	if err := c.submitUnary(c.swigluPipeline, bg, inter); err != nil {
		return nil, nil, err
	}
	return newDeviceBuffer(mid, inter), []func(){func() { mid.Release() }, pbuf.Release, bg.Release}, nil
}

// mlpInto runs the gated-MLP sub-block on xd in place (xd += Down(swiglu(Gate/Up
// (rmsnorm(xd))))), submit-no-poll. Returns scratch frees (free after the fence).
func (c *Context) mlpInto(xd, rmsW *DeviceBuffer, gate, up, down *ResidentW8A8, hidden, inter int, eps float32, addOne bool) ([]func(), error) {
	var frees []func()
	keep := func(fs []func(), err error) error { frees = append(frees, fs...); return err }
	xn, fs, err := c.rmsnormDevice(xd, rmsW, hidden, eps, addOne)
	if err := keep(fs, err); err != nil {
		return frees, err
	}
	qb, sb, err := c.quantizeDevice(xn, 1, hidden)
	if err != nil {
		return frees, err
	}
	frees = append(frees, func() { qb.Close() }, func() { sb.Close() })
	gd, err := c.matmulW8A8Device(qb, sb, gate, 1)
	if err != nil {
		return frees, err
	}
	ud, err := c.matmulW8A8Device(qb, sb, up, 1)
	if err != nil {
		return frees, err
	}
	frees = append(frees, func() { gd.Close() }, func() { ud.Close() })
	mid, fs2, err := c.swigluDevice(gd, ud, inter)
	if err := keep(fs2, err); err != nil {
		return frees, err
	}
	qb2, sb2, err := c.quantizeDevice(mid, 1, inter)
	if err != nil {
		return frees, err
	}
	frees = append(frees, func() { qb2.Close() }, func() { sb2.Close() })
	dd, err := c.matmulW8A8Device(qb2, sb2, down, 1)
	if err != nil {
		return frees, err
	}
	frees = append(frees, func() { dd.Close() })
	if err := keep(c.residualInPlace(xd, dd, hidden)); err != nil {
		return frees, err
	}
	return frees, nil
}

// LayerW is one transformer layer's resident GPU weights (attention + gated MLP).
type LayerW struct {
	Attn           AttnWeights
	MLPNorm        *DeviceBuffer
	Gate, Up, Down *ResidentW8A8
}

// ModelW is a decode-resident model: layers + final norm + LM head.
type ModelW struct {
	Layers    []LayerW
	FinalNorm *DeviceBuffer
	LMHead    *ResidentW8A8
}

// DecodeToken runs a full decode step for one token entirely on device — every
// layer's attention + MLP chained, then final norm + LM head — as ONE command
// stream with ONE fence (the logits readback). x is the token's input embedding;
// returns the logits. This is the one-command-buffer-per-token forward.
func (c *Context) DecodeToken(x []float32, m ModelW, hidden, nH, nKV, hd, inter, pos, start int, eps, scale float32, addOne bool) ([]float32, error) {
	xd, err := c.UploadF32(x)
	if err != nil {
		return nil, err
	}
	var frees []func()
	relAll := func() {
		xd.Close()
		for _, f := range frees {
			f()
		}
	}
	for i := range m.Layers {
		lw := &m.Layers[i]
		fa, err := c.attnBlockInto(xd, lw.Attn, hidden, nH, nKV, hd, pos, start, eps, scale, addOne)
		frees = append(frees, fa...)
		if err != nil {
			relAll()
			return nil, err
		}
		fm, err := c.mlpInto(xd, lw.MLPNorm, lw.Gate, lw.Up, lw.Down, hidden, inter, eps, addOne)
		frees = append(frees, fm...)
		if err != nil {
			relAll()
			return nil, err
		}
	}
	xn, fs, err := c.rmsnormDevice(xd, m.FinalNorm, hidden, eps, addOne)
	frees = append(frees, fs...)
	if err != nil {
		relAll()
		return nil, err
	}
	qb, sb, err := c.quantizeDevice(xn, 1, hidden)
	if err != nil {
		relAll()
		return nil, err
	}
	frees = append(frees, func() { qb.Close() }, func() { sb.Close() })
	logits, err := c.matmulW8A8Device(qb, sb, m.LMHead, 1)
	if err != nil {
		relAll()
		return nil, err
	}
	frees = append(frees, func() { logits.Close() })
	out, err := c.Readback(logits) // the single fence for the whole token
	relAll()
	return out, err
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
	return newDeviceBuffer(buf, capElems), nil
}

// NewKVCacheF16 creates a resident f16 KV cache holding capElems logical elements
// in ceil(capElems/2) u32 words (2 f16/word) — half the bytes of NewKVCache, the
// 2×-context lever. The ropeStoreF16/kvStoreF16 kernels pack into it and attnF16
// unpacks; the buffer is opaque u32 to WGSL. n stays the logical element count.
// initial (prior positions) is f16-packed before upload (the same rounding the
// kernels apply), so a prefilled f16 cache matches what a decode would have written.
func (c *Context) NewKVCacheF16(initial []float32, capElems int) (*DeviceBuffer, error) {
	words := (capElems + 1) / 2
	buf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "kvcache-f16", Size: uint64(words * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	if len(initial) > 0 {
		if err := c.queue.WriteBuffer(buf, 0, wgpu.ToBytes(packF16Pairs(initial))); err != nil {
			buf.Release()
			return nil, err
		}
	}
	return newDeviceBuffer(buf, capElems), nil
}

// NewKVCacheI8 creates a resident int8 KV cache holding capElems logical elements
// packed 4/word into ceil(capElems/4) u32 words (a quarter of NewKVCache's bytes,
// half of NewKVCacheF16's), plus a per-(position,KV-head) f32 scale side buffer of
// capElems/hd entries. The ropeStoreI8/kvStoreI8 kernels write packed int8 + scales
// during decode (and prefill, which is sequential Forward); attnI8 unpacks ×scale.
// initial (prior positions, [nPos*kvDim]) is per-head quantized (maxabs/127 — the
// same scheme the kernels and the CPU int8 cache use) so a prefilled cache matches
// what a decode would have written. nKV/hd give the per-head grouping for scales.
// Returns (packed data, scales); the caller Releases both.
func (c *Context) NewKVCacheI8(initial []float32, capElems, nKV, hd int) (*DeviceBuffer, *DeviceBuffer, error) {
	words := (capElems + 3) / 4
	scaleElems := capElems / hd // = ctxCap*nKV (one scale per position×KV-head)
	dataBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "kvcache-i8", Size: uint64(words * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, nil, err
	}
	scaleBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "kvcache-i8-scale", Size: uint64(scaleElems * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
	if err != nil {
		dataBuf.Release()
		return nil, nil, err
	}
	if len(initial) > 0 {
		w, s := packKVInt8(initial, nKV, hd)
		if err := c.queue.WriteBuffer(dataBuf, 0, wgpu.ToBytes(w)); err != nil {
			dataBuf.Release()
			scaleBuf.Release()
			return nil, nil, err
		}
		if err := c.queue.WriteBuffer(scaleBuf, 0, wgpu.ToBytes(s)); err != nil {
			dataBuf.Release()
			scaleBuf.Release()
			return nil, nil, err
		}
	}
	return newDeviceBuffer(dataBuf, capElems), newDeviceBuffer(scaleBuf, scaleElems), nil
}

// packKVInt8 per-head quantizes a [nPos*nKV*hd] f32 KV slab to int8 (4/word) plus a
// [nPos*nKV] f32 scale (maxabs/127 per (position,KV-head)) — the CPU mirror of the
// ropeStoreI8/kvStoreI8 WRITE kernels, used to prefill a resident int8 cache.
func packKVInt8(vals []float32, nKV, hd int) ([]uint32, []float32) {
	kvDim := nKV * hd
	nPos := len(vals) / kvDim
	words := make([]uint32, (len(vals)+3)/4)
	scales := make([]float32, nPos*nKV)
	for p := range nPos {
		for h := range nKV {
			base := p*kvDim + h*hd
			var amax float32
			for d := range hd {
				if a := float32(math.Abs(float64(vals[base+d]))); a > amax {
					amax = a
				}
			}
			sc := amax / 127
			if sc == 0 {
				sc = 1
			}
			scales[p*nKV+h] = sc
			inv := 1 / sc
			for d := range hd {
				q := int32(math.Round(float64(vals[base+d] * inv)))
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				e := base + d
				words[e>>2] |= uint32(uint8(int8(q))) << ((e & 3) * 8)
			}
		}
	}
	return words, scales
}

var _ = fmt.Sprint

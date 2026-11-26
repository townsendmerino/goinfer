//go:build gpu

package kernels

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// fusedEnc is one DecodeTokenFused or DecodeTokenFusedBatched call: its command encoder, the buffers and bind groups it
// releases at the end, and the FIRST device-allocation/bind failure. The helpers short-circuit once buildErr is set and
// the entry point returns it before Submit, so VRAM exhaustion is an error the caller falls back on rather than a panic
// (bind) or a downstream nil-buffer deref.
type fusedEnc struct {
	c         *Context
	name      string           // the entry point, as its errors name it
	storUsage wgpu.BufferUsage // the usage storF creates a buffer with
	// flushEvery > 0 commits the recorded passes every flushEvery dispatches (DecodeTokenFusedBatched's flushPasses);
	// 0 records every pass into the one encoder.
	flushEvery int

	hidden, nH, nKV, hd, inter, start, kvDim int
	eps, scale                               float32
	addOne                                   bool

	enc       *wgpu.CommandEncoder
	keep      []func()
	buildErr  error
	dispCount int
	// noSinks and noHasSink are FeatAttnSink's bindings — always bound (WGSL bind groups can't
	// bind a null storage buffer); these test-only paths never carry a real sink, so one shared
	// harmless dummy + hasSink=0 for every layer/row, matching attnShaderWGSL's convention.
	noSinks, noHasSink *wgpu.Buffer
}

// release runs at the end of the call: the open encoder, then every buffer and bind group kept.
func (fe *fusedEnc) release() {
	fe.enc.Release()
	for _, f := range fe.keep {
		f()
	}
}

func (fe *fusedEnc) keepBuf(b *wgpu.Buffer) { fe.keep = append(fe.keep, releaser(b)) }

func (fe *fusedEnc) keepBG(b *wgpu.BindGroup) { fe.keep = append(fe.keep, b.Release) }

func (fe *fusedEnc) storF(n int) *wgpu.Buffer {
	if fe.buildErr != nil {
		return nil
	}
	b, e := fe.c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: fe.storUsage})
	if e != nil {
		fe.buildErr = e
		return nil
	}
	fe.keepBuf(b)
	return b
}

func (fe *fusedEnc) uni(v []uint32) *wgpu.Buffer {
	if fe.buildErr != nil {
		return nil
	}
	b, e := fe.c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
	if e != nil {
		fe.buildErr = e
		return nil
	}
	fe.keepBuf(b)
	return b
}

func (fe *fusedEnc) bind(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
	if fe.buildErr != nil {
		return nil
	}
	es := make([]wgpu.BindGroupEntry, len(bufs))
	for i, b := range bufs {
		if b == nil { // an upstream storF/uni failed; don't deref a nil buffer
			fe.buildErr = fmt.Errorf("gpu: %s: nil buffer for binding %d (allocation failed)", fe.name, i)
			return nil
		}
		es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
	}
	bg, e := fe.c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
	if e != nil {
		fe.buildErr = e
		return nil
	}
	fe.keepBG(bg)
	return bg
}

// flushPasses commits the passes recorded so far and opens a fresh encoder, WITHOUT waiting
// for the GPU (no Poll) — same-queue submissions are ordered by WebGPU/Metal automatically, so
// a later row's KV-cache read still happens-after an earlier row's KV-cache write even across
// this boundary. Its only job is to keep the number of UNCOMMITTED native command buffers
// under Metal's cap (see DecodeTokenFusedBatched's doc comment); it is not a synchronization point.
func (fe *fusedEnc) flushPasses() {
	if fe.buildErr != nil {
		return
	}
	cmd, e := fe.enc.TryFinish(nil)
	if e != nil {
		fe.buildErr = e
		return
	}
	fe.c.queue.Submit(cmd)
	cmd.Release()
	fe.enc.Release()
	fe.enc, e = fe.c.device.TryCreateCommandEncoder(nil)
	if e != nil {
		fe.buildErr = e
	}
}

func (fe *fusedEnc) disp(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
	if fe.buildErr != nil || bg == nil {
		return
	}
	pass := fe.enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(gx, gy, 1)
	if e := pass.TryEnd(); e != nil {
		fe.buildErr = fmt.Errorf("gpu: %s: pass: %w", fe.name, e)
	}
	pass.Release()
	if fe.flushEvery == 0 {
		return
	}
	fe.dispCount++
	if fe.dispCount >= fe.flushEvery {
		fe.flushPasses()
		fe.dispCount = 0
	}
}

// cpy is a buildErr/nil-guarded CopyBufferToBuffer — a nil src/dst here means an upstream
// storF/tiledProj already failed, so skip rather than deref.
func (fe *fusedEnc) cpy(src *wgpu.Buffer, so uint64, dst *wgpu.Buffer, do, sz uint64) {
	if fe.buildErr != nil || src == nil || dst == nil {
		return
	}
	if e := fe.enc.TryCopyBufferToBuffer(src, so, dst, do, sz); e != nil {
		fe.buildErr = fmt.Errorf("gpu: %s: copy: %w", fe.name, e)
	}
}

// copyKV appends one position's K or V row to its cache.
func (fe *fusedEnc) copyKV(src, cache *wgpu.Buffer, pos int) {
	if fe.buildErr != nil || src == nil || cache == nil {
		return
	}
	if e := fe.enc.TryCopyBufferToBuffer(src, 0, cache, uint64(pos*fe.kvDim*4), uint64(fe.kvDim*4)); e != nil {
		fe.buildErr = fmt.Errorf("gpu: %s: KV copy: %w", fe.name, e)
	}
}

// --- per-row ops (one row's [hidden]/[dim] buffer) ---

func (fe *fusedEnc) rms(in, w *wgpu.Buffer) *wgpu.Buffer {
	c := fe.c
	out := fe.storF(fe.hidden)
	p := fe.uni([]uint32{uint32(fe.hidden), F32bits(fe.eps), BoolU32(fe.addOne), 0})
	// The rmsnorm kernel is single-workgroup (reduces one row); one workgroup, not 64. 64 was 64×
	// redundant work + 64 unsynchronised writes to the same addresses.
	fe.disp(c.rmsnormPipeline, fe.bind(c.rmsnormLayout, in, w, out, p), 1, 1)
	return out
}

func (fe *fusedEnc) quant(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
	c := fe.c
	kp := padK(K)
	q := fe.storF(kp / 4)
	s := fe.storF(1)
	p := fe.uni([]uint32{1, uint32(K), uint32(kp), 0})
	fe.disp(c.quantizePipeline, fe.bind(c.quantizeLayout, in, q, s, p), 1, 1)
	return q, s
}

func (fe *fusedEnc) gemv(aq, as *wgpu.Buffer, rm *ResidentW8A8) *wgpu.Buffer {
	c := fe.c
	out := fe.storF(rm.rows)
	p := fe.uni([]uint32{1, uint32(rm.kp), uint32(rm.rows), 0})
	bg := fe.bind(c.gemvLayout, aq, rm.bq, as, rm.bScales, out, p)
	gx, gy := GemvGrid(rm.rows)
	fe.disp(c.gemvPipeline, bg, gx, gy)
	return out
}

func (fe *fusedEnc) rope(vec, invFreq *wgpu.Buffer, heads, pos int) {
	c, hd := fe.c, fe.hd
	half := hd / 2
	p := fe.uni([]uint32{uint32(heads), uint32(hd), uint32(half), uint32(pos), F32bits(1), 0, 0, 0})
	fe.disp(c.ropePipeline, fe.bind(c.ropeLayout, vec, invFreq, p), uint32(heads*half+63)/64, 1)
}

func (fe *fusedEnc) residual(x, y *wgpu.Buffer) {
	c := fe.c
	p := fe.uni([]uint32{uint32(fe.hidden), 0, 0, 0})
	fe.disp(c.residualPipeline, fe.bind(c.residualLayout, x, y, p), uint32(fe.hidden+63)/64, 1)
}

// attn runs one row's attention over the cache up to pos into a fresh [nH*hd] context.
func (fe *fusedEnc) attn(q *wgpu.Buffer, lw *LayerW, pos int) *wgpu.Buffer {
	c, nH, nKV, hd := fe.c, fe.nH, fe.nKV, fe.hd
	cv := fe.storF(nH * hd)
	ap := fe.uni([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(pos + 1), uint32(fe.start), uint32(nH / nKV), F32bits(fe.scale), 0})
	fe.disp(c.attnPipeline, fe.bind(c.attnLayout, q, lw.Attn.KCache.buf, lw.Attn.VCache.buf, cv, fe.noSinks, ap, fe.noHasSink), uint32(nH), 1)
	return cv
}

// swiglu is one row's gated product into a fresh [inter] buffer.
func (fe *fusedEnc) swiglu(gate, up *wgpu.Buffer) *wgpu.Buffer {
	c, inter := fe.c, fe.inter
	md := fe.storF(inter)
	sp := fe.uni([]uint32{uint32(inter), 0, 0, 0})
	fe.disp(c.swigluPipeline, fe.bind(c.swigluLayout, gate, up, md, sp), uint32(inter+63)/64, 1)
	return md
}

// tiledProj batches a projection over all M rows: gather the M quantized rows
// into one packed buffer, run ONE tiled GEMM (weights streamed once), scatter the
// M output rows back to per-row buffers. This is the Stage-B weight-stream win.
func (fe *fusedEnc) tiledProj(xnRows []*wgpu.Buffer, rm *ResidentW8A8) []*wgpu.Buffer {
	c, M := fe.c, len(xnRows)
	K, N := rm.cols, rm.rows
	kw := rm.kp / 4
	aqC := fe.storF(M * kw)
	asC := fe.storF(M)
	for r, xn := range xnRows {
		q, s := fe.quant(xn, K)
		fe.cpy(q, 0, aqC, uint64(r*kw*4), uint64(kw*4))
		fe.cpy(s, 0, asC, uint64(r*4), 4)
	}
	dstC := fe.storF(M * N)
	p := fe.uni([]uint32{uint32(M), uint32(rm.kp), uint32(N), 0})
	gx, gy := GemvGrid(N) // thin-M kernel: one workgroup per output column
	fe.disp(c.gemmRowPipeline, fe.bind(c.gemmRowLayout, aqC, rm.bq, asC, rm.bScales, dstC, p), gx, gy)
	outs := make([]*wgpu.Buffer, M)
	for r := range outs {
		o := fe.storF(N)
		fe.cpy(dstC, uint64(r*N*4), o, 0, uint64(N*4))
		outs[r] = o
	}
	return outs
}

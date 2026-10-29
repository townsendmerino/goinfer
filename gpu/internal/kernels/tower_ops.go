//go:build gpu

package kernels

import (
	"github.com/townsendmerino/goinfer/internal/mem"
	"math"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// What the float32 device towers share (Gemma 4's vision tower, the grid towers): a Context of their own, a way to run
// a kernel over any number of buffers, a scope that releases a stretch of the forward's buffers together, the row
// RMSNorm, and attention over a row segment as matmuls.

// towerRMSRowsWGSL: RMSNorm over each of `rows` rows of width h, dst = src / sqrt(mean(src²) + eps) * weight. One
// workgroup per row, rows laid out gx to a line of workgroups so a row count past the per-dimension limit still fits.
const towerRMSRowsWGSL = `
struct P { rows: u32, h: u32, eps: f32, gx: u32 };
@group(0) @binding(0) var<storage, read>       src:    array<f32>; // [rows, h]
@group(0) @binding(1) var<storage, read>       weight: array<f32>; // [h]
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>; // [rows, h]
@group(0) @binding(3) var<uniform>             p:      P;
var<workgroup> sh: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.y * p.gx + wid.x;
    if (row >= p.rows) { return; }
    let base = row * p.h;
    let t = lid.x;
    var s2: f32 = 0.0;
    for (var i: u32 = t; i < p.h; i = i + 64u) { let v = src[base + i]; s2 = s2 + v * v; }
    sh[t] = s2;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { sh[t] = sh[t] + sh[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = 1.0 / sqrt(sh[0] / f32(p.h) + p.eps);
    for (var i: u32 = t; i < p.h; i = i + 64u) {
        dst[base + i] = src[base + i] * inv * weight[i];
    }
}
`

type towerPipe struct {
	pl  *wgpu.ComputePipeline
	lay *wgpu.BindGroupLayout
}

// towerOps is a tower's Context and shared kernels. One forward at a time: the tower holds its own lock.
type towerOps struct {
	c       *Context
	rms     towerPipe
	eps     float32 // RMSNorm's
	pending int     // score-matrix bytes submitted since the device was last waited for
	err     error   // the first pipeline that failed to build (pipe)
}

// towerBlockForTest, when positive, is the number of query rows one attention block scores at a time, in place of the
// number that fits towerScoreBytes. 0 outside the towers' gates.
var towerBlockForTest int

// towerScoreBytes bounds one attention block's score matrix [rows, keys]. The device is waited for whenever four
// times that has been submitted, so released score matrices do not pile up behind the queue.
const towerScoreBytes = 64 << 20

// newTowerOps opens a Context and builds the kernels every tower uses. rmsEps is the tower's RMSNorm epsilon.
func newTowerOps(rmsEps float32) (*towerOps, error) {
	c, err := New()
	if err != nil {
		return nil, err
	}
	o := &towerOps{c: c, eps: rmsEps}
	if err := c.ensureVision(); err != nil {
		c.Close()
		return nil, err
	}
	if o.pipe(&o.rms, "towerRMSRows", towerRMSRowsWGSL); o.err != nil {
		c.Close()
		return nil, o.err
	}
	return o, nil
}

// pipe builds one kernel into dst; the first failure is kept in o.err and later calls do nothing.
func (o *towerOps) pipe(dst *towerPipe, label, code string) {
	if o.err != nil {
		return
	}
	_, dst.pl, dst.lay, o.err = o.c.mkPipeline(label, code)
}

// close releases bufs (nil entries skipped) and the Context. A second call does nothing.
func (o *towerOps) close(bufs []*DeviceBuffer) error {
	if o == nil || o.c == nil {
		return nil
	}
	c := o.c
	o.c = nil
	c.releaseOwned(func() {
		for _, d := range bufs {
			if d != nil {
				_ = d.Close()
			}
		}
	})
	return c.Close()
}

// kernelN runs k over bufs (bindings 0 to len(bufs)-1) and a uniform holding p (the next binding), gx by gy
// workgroups. No Poll.
func (c *Context) kernelN(k towerPipe, bufs []*wgpu.Buffer, p []uint32, gx, gy int) error {
	pbuf, err := c.tryCreateBufferInit(mem.LineCompanion, &wgpu.BufferInitDescriptor{Label: "kn-p", Contents: wgpu.ToBytes(p), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return err
	}
	defer releaseBuf(pbuf)
	entries := make([]wgpu.BindGroupEntry, 0, len(bufs)+1)
	for i, b := range bufs {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()})
	}
	entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(len(bufs)), Buffer: pbuf, Size: pbuf.GetSize()})
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: k.lay, Entries: entries})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.TryCreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(k.pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(gx), uint32(gy), 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return err
	}
	pass.Release()
	cmd, _ := enc.TryFinish(nil)
	defer cmd.Release()
	c.queue.Submit(cmd)
	return nil
}

// towerScope holds the device buffers one stretch of a forward allocates, released together. A buffer may be released
// once its work is submitted: the device keeps it until that work is done.
type towerScope struct{ bufs []*DeviceBuffer }

func (s *towerScope) add(d *DeviceBuffer, err error) (*DeviceBuffer, error) {
	if err == nil {
		s.bufs = append(s.bufs, d)
	}
	return d, err
}

func (s *towerScope) release() {
	for _, d := range s.bufs {
		_ = d.Close()
	}
	s.bufs = nil
}

// rmsRows is RMSNorm over rows of width h into a new buffer, at the tower's epsilon.
func (o *towerOps) rmsRows(s *towerScope, src, weight *DeviceBuffer, rows, h int) (*DeviceBuffer, error) {
	return o.rmsRowsEps(s, src, weight, rows, h, o.eps)
}

// rmsRowsEps is rmsRows at epsilon eps.
func (o *towerOps) rmsRowsEps(s *towerScope, src, weight *DeviceBuffer, rows, h int, eps float32) (*DeviceBuffer, error) {
	dst, err := s.add(o.c.newDevBuf(rows * h))
	if err != nil {
		return nil, err
	}
	gx := min(rows, 32768)
	p := []uint32{uint32(rows), uint32(h), math.Float32bits(eps), uint32(gx)}
	return dst, o.c.kernelN(o.rms, []*wgpu.Buffer{src.buf, weight.buf, dst.buf}, p, gx, (rows+gx-1)/gx)
}

// attention is full attention among the T rows from row0 of q, k, v [rows, nH, hd] (H = nH*hd wide), one head at a
// time, written to the same rows of ctx. The scores are multiplied by scale before the softmax. A head's query rows are
// scored in blocks, so the score matrix a block holds stays under towerScoreBytes.
func (o *towerOps) attention(q, k, v, ctx *DeviceBuffer, row0, T, H, nH, hd int, scale float32) error {
	c := o.c
	block := max(64, towerScoreBytes/(4*T)/64*64)
	if towerBlockForTest > 0 {
		block = towerBlockForTest
	}
	block = min(block, T)
	// rows scores B query rows from r0 of the head at off against the segment's keys and writes their context rows.
	rows := func(kh, vt *DeviceBuffer, r0, B, off int) error {
		var bs towerScope
		defer bs.release()
		rowOff := (row0+r0)*H + off // the block's first row, this head, in a [rows, H] buffer
		qh, err := bs.add(c.newDevBuf(B * hd))
		if err != nil {
			return err
		}
		if err := c.copyHeadDevice(q.buf, qh.buf, B, hd, H, rowOff, 0); err != nil {
			return err
		}
		scores, err := bs.add(c.matmulF32Device(qh.buf, kh.buf, B, hd, T)) // [B, T]
		if err != nil {
			return err
		}
		if err := c.inplaceKernel(c.softmaxPipeline, c.softmaxLayout, scores.buf, []uint32{uint32(B), uint32(T), math.Float32bits(scale), 0}, B); err != nil {
			return err
		}
		oh, err := bs.add(c.matmulF32Device(scores.buf, vt.buf, B, T, hd)) // [B, hd]
		if err != nil {
			return err
		}
		return c.copyHeadDevice(oh.buf, ctx.buf, B, hd, H, rowOff, 1)
	}
	// head runs one head over every block of query rows.
	head := func(off int) error {
		var hs towerScope
		defer hs.release()
		kh, err := hs.add(c.newDevBuf(T * hd))
		if err != nil {
			return err
		}
		if err := c.copyHeadDevice(k.buf, kh.buf, T, hd, H, row0*H+off, 0); err != nil {
			return err
		}
		vt, err := hs.add(c.newDevBuf(T * hd)) // [hd, T]
		if err != nil {
			return err
		}
		if err := c.copyHeadDevice(v.buf, vt.buf, T, hd, H, row0*H+off, 2); err != nil {
			return err
		}
		for r0 := 0; r0 < T; r0 += block {
			B := min(block, T-r0)
			if err := rows(kh, vt, r0, B, off); err != nil {
				return err
			}
			if o.pending += 4 * B * T; o.pending > 4*towerScoreBytes {
				c.device.Poll(true, nil) // lets the device free the score matrices already released
				o.pending = 0
			}
		}
		return nil
	}
	for h := range nH {
		if err := head(h * hd); err != nil {
			return err
		}
	}
	return nil
}

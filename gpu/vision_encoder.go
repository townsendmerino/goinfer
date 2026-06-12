//go:build gpu

package gpu

import (
	"fmt"
	"math"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/aikit/vision"
)

// VisionEncoder is the resident GPU SigLIP forward. It uploads the tower once
// (int8 matmul weights as ResidentW8A8, f32 norms/biases as device buffers) and
// runs ForwardPatches entirely on the device: the [np, hidden] activation stays
// resident, every op chains as a Submit, and there is ONE Poll per layer (which
// just waits for that layer's compute and bounds memory before the next layer's
// scratch is allocated — 8 GB can't hold 27 layers of intermediates at once).
// This pays WebGPU's submit/sync cost ~27× instead of the per-op-offload's ~162×
// (a measured dead end). vision.Encoder delegates here when a resident backend is
// attached (-tags gpu); the default build is pure-Go CPU.
type VisionEncoder struct {
	c                                        *Context
	w                                        vision.GPUWeights
	patchW, patchB, posEmb, postLNw, postLNb *DeviceBuffer
	layers                                   []visLayer
}

type visLayer struct {
	ln1w, ln1b, ln2w, ln2b     *DeviceBuffer
	qb, kb, vb, ob, fc1b, fc2b *DeviceBuffer
	qw, kw, vw, ow, fc1w, fc2w *ResidentW8A8
}

// NewVisionEncoder uploads the tower to the device.
func (c *Context) NewVisionEncoder(w vision.GPUWeights) (*VisionEncoder, error) {
	if err := c.ensureVision(); err != nil {
		return nil, err
	}
	if err := c.ensureQuant(); err != nil {
		return nil, err
	}
	ve := &VisionEncoder{c: c, w: w}
	var err error
	up := func(x []float32) *DeviceBuffer {
		if err != nil {
			return nil
		}
		var d *DeviceBuffer
		d, err = c.UploadF32(x)
		return d
	}
	upW := func(m vision.GPUMat) *ResidentW8A8 {
		if err != nil {
			return nil
		}
		var r *ResidentW8A8
		r, err = c.UploadW8A8(m.Q, m.Scales, m.Rows, m.Cols)
		return r
	}
	ve.patchW, ve.patchB, ve.posEmb = up(w.PatchW), up(w.PatchB), up(w.PosEmb)
	ve.postLNw, ve.postLNb = up(w.PostLNw), up(w.PostLNb)
	ve.layers = make([]visLayer, len(w.Layers))
	for i := range w.Layers {
		l := &w.Layers[i]
		ve.layers[i] = visLayer{
			ln1w: up(l.LN1w), ln1b: up(l.LN1b), ln2w: up(l.LN2w), ln2b: up(l.LN2b),
			qb: up(l.Qb), kb: up(l.Kb), vb: up(l.Vb), ob: up(l.Ob), fc1b: up(l.FC1b), fc2b: up(l.FC2b),
			qw: upW(l.Qw), kw: upW(l.Kw), vw: upW(l.Vw), ow: upW(l.Ow), fc1w: upW(l.FC1w), fc2w: upW(l.FC2w),
		}
	}
	if err != nil {
		ve.Close()
		return nil, fmt.Errorf("gpu: upload vision tower: %w", err)
	}
	return ve, nil
}

func (c *Context) newDevBuf(n int) (*DeviceBuffer, error) {
	b, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "vbuf", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return nil, err
	}
	return &DeviceBuffer{buf: b, n: n}, nil
}

// ln runs LayerNorm(src) → a new device buffer [rows, h].
func (ve *VisionEncoder) ln(src *DeviceBuffer, w, b *DeviceBuffer, rows, h int) (*DeviceBuffer, error) {
	dst, err := ve.c.newDevBuf(rows * h)
	if err != nil {
		return nil, err
	}
	if err := ve.c.layerNormRowsDevice(src.buf, w.buf, b.buf, dst.buf, rows, h, ve.w.Eps); err != nil {
		dst.Release()
		return nil, err
	}
	return dst, nil
}

// proj runs an int8 projection: quantize(src[M,K]) → W8A8 matmul against rm[N,K]
// → +bias[N] → out[M,N]. Releases the transient quantized activation.
func (ve *VisionEncoder) proj(src *DeviceBuffer, rm *ResidentW8A8, bias *DeviceBuffer, M, K int) (*DeviceBuffer, error) {
	c := ve.c
	aq, as, err := c.quantizeDevice(src, M, K)
	if err != nil {
		return nil, err
	}
	defer aq.Release()
	defer as.Release()
	out, err := c.matmulW8A8Device(aq, as, rm, M)
	if err != nil {
		return nil, err
	}
	if err := c.addRowsDevice(out.buf, bias.buf, M*rm.rows, rm.rows, 0); err != nil {
		out.Release()
		return nil, err
	}
	return out, nil
}

// gatherHead extracts head `off:off+hd` of src[rows, stride] into a contiguous
// buffer: [rows,hd] (dir 0) or transposed [hd,rows] (dir 2).
func (ve *VisionEncoder) gatherHead(src *DeviceBuffer, rows, hd, stride, off, dir int) (*DeviceBuffer, error) {
	dst, err := ve.c.newDevBuf(rows * hd)
	if err != nil {
		return nil, err
	}
	if err := ve.c.copyHeadDevice(src.buf, dst.buf, rows, hd, stride, off, dir); err != nil {
		dst.Release()
		return nil, err
	}
	return dst, nil
}

// ForwardPatches runs the resident SigLIP forward on im2col patches
// [np * (C*P*P)] (vision.Encoder.GridPatches) and returns last_hidden_state
// [np * hidden].
func (ve *VisionEncoder) ForwardPatches(patches []float32) ([]float32, error) {
	c, w := ve.c, ve.w
	np, hidden, inter, hd, nH := w.NumPatches, w.Hidden, w.Inter, w.HeadDim, w.NumHeads
	cpp := len(patches) / np
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	// Patch embed: h = patches·patchWᵀ + patchB + posEmb. h is the resident
	// residual stream for the whole forward.
	pd, err := c.UploadF32(patches)
	if err != nil {
		return nil, err
	}
	h, err := c.matmulF32Device(pd.buf, ve.patchW.buf, np, cpp, hidden)
	pd.Release()
	if err != nil {
		return nil, err
	}
	defer h.Release()
	if err := c.addRowsDevice(h.buf, ve.patchB.buf, np*hidden, hidden, 0); err != nil {
		return nil, err
	}
	if err := c.addRowsDevice(h.buf, ve.posEmb.buf, np*hidden, hidden, 1); err != nil {
		return nil, err
	}
	c.device.Poll(true, nil)

	for li := range ve.layers {
		L := &ve.layers[li]
		var keep []*DeviceBuffer
		fail := func(e error) error {
			for _, b := range keep {
				b.Release()
			}
			return e
		}
		add := func(d *DeviceBuffer, e error) (*DeviceBuffer, error) {
			if e != nil {
				return nil, e
			}
			keep = append(keep, d)
			return d, nil
		}

		// --- attention block (pre-LN, residual) ---
		n1, err := add(ve.ln(h, L.ln1w, L.ln1b, np, hidden))
		if err != nil {
			return nil, fail(err)
		}
		q, err := add(ve.proj(n1, L.qw, L.qb, np, hidden))
		if err != nil {
			return nil, fail(err)
		}
		k, err := add(ve.proj(n1, L.kw, L.kb, np, hidden))
		if err != nil {
			return nil, fail(err)
		}
		v, err := add(ve.proj(n1, L.vw, L.vb, np, hidden))
		if err != nil {
			return nil, fail(err)
		}
		att, err := add(c.newDevBuf(np * hidden))
		if err != nil {
			return nil, fail(err)
		}
		for head := 0; head < nH; head++ {
			off := head * hd
			qh, e1 := ve.gatherHead(q, np, hd, hidden, off, 0)
			kh, e2 := ve.gatherHead(k, np, hd, hidden, off, 0)
			vt, e3 := ve.gatherHead(v, np, hd, hidden, off, 2) // [hd, np]
			if e1 != nil || e2 != nil || e3 != nil {
				return nil, fail(fmt.Errorf("gather: %v %v %v", e1, e2, e3))
			}
			scores, e := c.matmulF32Device(qh.buf, kh.buf, np, hd, np) // [np,np]
			if e != nil {
				return nil, fail(e)
			}
			if e := c.inplaceKernel(c.softmaxPipeline, c.softmaxLayout, scores.buf, []uint32{uint32(np), uint32(np), math.Float32bits(scale), 0}, np); e != nil {
				return nil, fail(e)
			}
			oh, e := c.matmulF32Device(scores.buf, vt.buf, np, np, hd) // [np,hd]
			if e != nil {
				return nil, fail(e)
			}
			if e := c.copyHeadDevice(oh.buf, att.buf, np, hd, hidden, off, 1); e != nil { // scatter
				return nil, fail(e)
			}
			qh.Release()
			kh.Release()
			vt.Release()
			scores.Release()
			oh.Release()
		}
		o, err := add(ve.proj(att, L.ow, L.ob, np, hidden))
		if err != nil {
			return nil, fail(err)
		}
		if err := c.addRowsDevice(h.buf, o.buf, np*hidden, hidden, 1); err != nil { // h += o
			return nil, fail(err)
		}

		// --- MLP block (pre-LN, residual): fc2(gelu(fc1(x))) ---
		n2, err := add(ve.ln(h, L.ln2w, L.ln2b, np, hidden))
		if err != nil {
			return nil, fail(err)
		}
		mid, err := add(ve.proj(n2, L.fc1w, L.fc1b, np, hidden)) // [np, inter]
		if err != nil {
			return nil, fail(err)
		}
		if err := c.inplaceKernel(c.geluPipeline, c.geluLayout, mid.buf, []uint32{uint32(np * inter), 0, 0, 0}, maxWG((np*inter+63)/64)); err != nil {
			return nil, fail(err)
		}
		mlp, err := add(ve.proj(mid, L.fc2w, L.fc2b, np, inter)) // [np, hidden]
		if err != nil {
			return nil, fail(err)
		}
		if err := c.addRowsDevice(h.buf, mlp.buf, np*hidden, hidden, 1); err != nil { // h += mlp
			return nil, fail(err)
		}

		c.device.Poll(true, nil) // ONE sync per layer; release this layer's scratch
		for _, b := range keep {
			b.Release()
		}
	}

	out, err := ve.ln(h, ve.postLNw, ve.postLNb, np, hidden)
	if err != nil {
		return nil, err
	}
	defer out.Release()
	return c.Readback(out)
}

// Close releases all device buffers.
func (ve *VisionEncoder) Close() {
	for _, d := range []*DeviceBuffer{ve.patchW, ve.patchB, ve.posEmb, ve.postLNw, ve.postLNb} {
		if d != nil {
			d.Release()
		}
	}
	for i := range ve.layers {
		L := &ve.layers[i]
		for _, d := range []*DeviceBuffer{L.ln1w, L.ln1b, L.ln2w, L.ln2b, L.qb, L.kb, L.vb, L.ob, L.fc1b, L.fc2b} {
			if d != nil {
				d.Release()
			}
		}
		for _, r := range []*ResidentW8A8{L.qw, L.kw, L.vw, L.ow, L.fc1w, L.fc2w} {
			if r != nil {
				r.Release()
			}
		}
	}
}

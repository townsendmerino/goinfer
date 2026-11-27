//go:build gpu

package towers

import (
	"fmt"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/mem"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// VisionEncoder is the resident GPU SigLIP forward. It uploads the tower once (the matmul weights as int8 ResidentW8A8,
// or as float32 device buffers in the float32 form; f32 norms/biases as device buffers) and runs ForwardPatches entirely
// on the device: the [np, hidden] activation stays resident, every op chains as a Submit, and there is ONE Poll per layer
// (which just waits for that layer's compute and bounds memory before the next layer's scratch is allocated).
// vision.Encoder delegates here when a resident backend is attached (-tags gpu); the default build is pure-Go CPU.
// History: docs/code-notes/gpu.md#VisionEncoder.
type VisionEncoder struct {
	c *Context
	// w carries the tower's dimensions and eps in both forms; its Layers are set in the int8 form only.
	w                                        vision.GPUWeights
	patchW, patchB, posEmb, postLNw, postLNb *DeviceBuffer
	layers                                   []visLayer
	// f32: the projections are float32 (NewVisionEncoderF32). ForwardPatches then returns the last block's output and the
	// caller applies the post-layernorm on the host (vision.Encoder.FinishHidden), which the float32 export leaves there.
	f32     bool
	profile map[string]time.Duration // test seam: per-step wall time
	defect  int                      // test seam: planted defect
}

type visLayer struct {
	ln1w, ln1b, ln2w, ln2b     *DeviceBuffer
	qb, kb, vb, ob, fc1b, fc2b *DeviceBuffer
	qw, kw, vw, ow, fc1w, fc2w *ResidentW8A8 // the int8 form's projections
	qf, kf, vf, of, fc1f, fc2f *DeviceBuffer // the float32 form's, each [out, in]
}

// NewVisionEncoderF32 uploads the tower in float32 from the encoder's own weights (vision.Encoder.Weights; the tower must
// be loaded with quant=false).
func NewVisionEncoderF32(c *Context, w vision.SiglipWeights) (*VisionEncoder, error) {
	if err := c.EnsureVision(); err != nil {
		return nil, err
	}
	cfg := w.Cfg
	if cfg.NumAttentionHeads <= 0 || cfg.HiddenSize%cfg.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("gpu: vision tower hidden %d does not divide into %d heads", cfg.HiddenSize, cfg.NumAttentionHeads)
	}
	ve := &VisionEncoder{c: c, f32: true, w: vision.GPUWeights{
		Hidden: cfg.HiddenSize, Inter: cfg.IntermediateSize, NumLayers: len(w.Blocks), NumHeads: cfg.NumAttentionHeads,
		HeadDim: cfg.HiddenSize / cfg.NumAttentionHeads, NumPatches: w.NumPatches, PatchSize: cfg.PatchSize,
		NumChannels: cfg.NumChannels, Eps: float32(cfg.LayerNormEps),
	}}
	var err error
	up := func(x []float32) *DeviceBuffer {
		if err != nil {
			return nil
		}
		var d *DeviceBuffer
		d, err = c.UploadF32On(mem.LineCompanion, x)
		return d
	}
	upP := func(name string, p vision.VisionProj, out, in int) (*DeviceBuffer, *DeviceBuffer) {
		if err == nil && (p.Out != out || p.In != in || len(p.W) != out*in || len(p.B) != out) {
			err = fmt.Errorf("%s is [%d, %d] with %d weights and %d biases, want [%d, %d]", name, p.Out, p.In, len(p.W), len(p.B), out, in)
		}
		return up(p.W), up(p.B)
	}
	h, inter := cfg.HiddenSize, cfg.IntermediateSize
	ve.patchW, ve.patchB, ve.posEmb = up(w.PatchW), up(w.PatchB), up(w.PosEmb)
	ve.layers = make([]visLayer, len(w.Blocks))
	for i := range w.Blocks {
		b := &w.Blocks[i]
		L := &ve.layers[i]
		L.ln1w, L.ln1b, L.ln2w, L.ln2b = up(b.LN1W), up(b.LN1B), up(b.LN2W), up(b.LN2B)
		L.qf, L.qb = upP("q", b.Q, h, h)
		L.kf, L.kb = upP("k", b.K, h, h)
		L.vf, L.vb = upP("v", b.V, h, h)
		L.of, L.ob = upP("o", b.O, h, h)
		L.fc1f, L.fc1b = upP("fc1", b.FC1, inter, h)
		L.fc2f, L.fc2b = upP("fc2", b.FC2, h, inter)
	}
	if err != nil {
		ve.Close()
		return nil, fmt.Errorf("gpu: upload float32 vision tower: %w", err)
	}
	return ve, nil
}

// projF32 runs a float32 projection: src[M,K] · w[N,K]ᵀ + bias[N] → out[M,N].
func (ve *VisionEncoder) projF32(src, w, bias *DeviceBuffer, M, K, N int) (*DeviceBuffer, error) {
	out, err := ve.c.MatmulF32Device(src.Raw(), w.Raw(), M, K, N)
	if err != nil {
		return nil, err
	}
	if bias != nil {
		if err := ve.c.AddRowsDevice(out.Raw(), bias.Raw(), M*N, N, 0); err != nil {
			_ = out.Close()
			return nil, err
		}
	}
	return out, nil
}

// NewVisionEncoder uploads the tower to the device.
func NewVisionEncoder(c *Context, w vision.GPUWeights) (*VisionEncoder, error) {
	if err := c.EnsureVision(); err != nil {
		return nil, err
	}
	if err := c.EnsureQuant(); err != nil {
		return nil, err
	}
	ve := &VisionEncoder{c: c, w: w}
	var err error
	up := func(x []float32) *DeviceBuffer {
		if err != nil {
			return nil
		}
		var d *DeviceBuffer
		d, err = c.UploadF32On(mem.LineCompanion, x)
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

// ln runs LayerNorm(src) → a new device buffer [rows, h].
func (ve *VisionEncoder) ln(src *DeviceBuffer, w, b *DeviceBuffer, rows, h int) (*DeviceBuffer, error) {
	dst, err := ve.c.NewDevBuf(rows * h)
	if err != nil {
		return nil, err
	}
	if err := ve.c.LayerNormRowsDevice(src.Raw(), w.Raw(), b.Raw(), dst.Raw(), rows, h, ve.w.Eps); err != nil {
		_ = dst.Close()
		return nil, err
	}
	return dst, nil
}

// proj runs an int8 projection: quantize(src[M,K]) → W8A8 matmul against rm[N,K]
// → +bias[N] → out[M,N]. Releases the transient quantized activation.
func (ve *VisionEncoder) proj(src *DeviceBuffer, rm *ResidentW8A8, bias *DeviceBuffer, M, K int) (*DeviceBuffer, error) {
	c := ve.c
	aq, as, err := c.QuantizeDevice(src, M, K)
	if err != nil {
		return nil, err
	}
	defer aq.Close()
	defer as.Close()
	out, err := c.MatmulW8A8Device(aq, as, rm, M)
	if err != nil {
		return nil, err
	}
	if err := c.AddRowsDevice(out.Raw(), bias.Raw(), M*rm.Rows(), rm.Rows(), 0); err != nil {
		_ = out.Close()
		return nil, err
	}
	return out, nil
}

// gatherHead extracts head `off:off+hd` of src[rows, stride] into a contiguous
// buffer: [rows,hd] (dir 0) or transposed [hd,rows] (dir 2).
func (ve *VisionEncoder) gatherHead(src *DeviceBuffer, rows, hd, stride, off, dir int) (*DeviceBuffer, error) {
	dst, err := ve.c.NewDevBuf(rows * hd)
	if err != nil {
		return nil, err
	}
	if err := ve.c.CopyHeadDevice(src.Raw(), dst.Raw(), rows, hd, stride, off, dir); err != nil {
		_ = dst.Close()
		return nil, err
	}
	return dst, nil
}

// ForwardPatches runs the resident SigLIP forward on im2col patches
// [np * (C*P*P)] (vision.Encoder.GridPatches) and returns last_hidden_state
// [np * hidden].
func (ve *VisionEncoder) ForwardPatches(patches []float32) ([]float32, error) {
	c, w := ve.c, ve.w
	np, hidden, inter, nH := w.NumPatches, w.Hidden, w.Inter, w.NumHeads
	cpp := len(patches) / np
	last := time.Now()
	mark := func(step string) {
		if ve.profile == nil {
			return
		}
		c.PollWait()
		now := time.Now()
		ve.profile[step] += now.Sub(last)
		last = now
	}

	// Patch embed: h = patches·patchWᵀ + patchB + posEmb. h is the resident
	// residual stream for the whole forward.
	pd, err := c.UploadF32On(mem.LineCompanion, patches)
	if err != nil {
		return nil, err
	}
	h, err := c.MatmulF32Device(pd.Raw(), ve.patchW.Raw(), np, cpp, hidden)
	_ = pd.Close()
	if err != nil {
		return nil, err
	}
	defer h.Close()
	if err := c.AddRowsDevice(h.Raw(), ve.patchB.Raw(), np*hidden, hidden, 0); err != nil {
		return nil, err
	}
	if !(ve.f32 && ve.defect == 1) {
		if err := c.AddRowsDevice(h.Raw(), ve.posEmb.Raw(), np*hidden, hidden, 1); err != nil {
			return nil, err
		}
	}
	c.PollWait()
	mark("patch embed")
	// proj is one projection in whichever form the tower was uploaded.
	proj := func(src *DeviceBuffer, w8 *ResidentW8A8, wf, bias *DeviceBuffer, M, K, N int) (*DeviceBuffer, error) {
		if ve.f32 {
			return ve.projF32(src, wf, bias, M, K, N)
		}
		return ve.proj(src, w8, bias, M, K)
	}

	cfg := vision.EncoderConfig{HiddenSize: hidden, IntermediateSize: inter, NumAttentionHeads: nH, LayerNormEps: float64(w.Eps)}
	prog := vision.SiglipProgram(cfg)
	for li := range ve.layers {
		defect := 0
		if ve.f32 && li == 0 {
			defect = ve.defect
		}
		keep, err := ve.walkBlock(prog, &ve.layers[li], h, defect, proj, mark)
		if err == nil {
			c.PollWait() // ONE sync per layer; release this layer's scratch
			mark("mlp residual and the layer's sync")
		}
		for _, b := range keep {
			_ = b.Close()
		}
		if err != nil {
			return nil, fmt.Errorf("gpu: vision block %d: %w", li, err)
		}
	}

	if ve.f32 { // the post-layernorm is the host's (vision.Encoder.FinishHidden)
		return c.Readback(h)
	}
	out, err := ve.ln(h, ve.postLNw, ve.postLNb, np, hidden)
	if err != nil {
		return nil, err
	}
	defer out.Close()
	return c.Readback(out)
}

// Close releases every device buffer and resident matrix this encoder allocated. It returns an error to satisfy io.Closer and
// match the rest of the gpu resource types; teardown is best-effort (the underlying releases cannot meaningfully fail), so it
// always returns nil. Calls the standard Close() (not the deprecated Release alias) on each sub-resource.
func (ve *VisionEncoder) Close() error {
	ve.c.ReleaseOwned(ve.release)
	return nil
}

// release is Close's body (the wrappers it closes see they are already on the serializing goroutine and run in place).
func (ve *VisionEncoder) release() {
	for _, d := range []*DeviceBuffer{ve.patchW, ve.patchB, ve.posEmb, ve.postLNw, ve.postLNb} {
		if d != nil {
			_ = d.Close()
		}
	}
	for i := range ve.layers {
		L := &ve.layers[i]
		for _, d := range []*DeviceBuffer{L.ln1w, L.ln1b, L.ln2w, L.ln2b, L.qb, L.kb, L.vb, L.ob, L.fc1b, L.fc2b, L.qf, L.kf, L.vf, L.of, L.fc1f, L.fc2f} {
			if d != nil {
				_ = d.Close()
			}
		}
		for _, r := range []*ResidentW8A8{L.qw, L.kw, L.vw, L.ow, L.fc1w, L.fc2w} {
			if r != nil {
				_ = r.Close()
			}
		}
	}
}

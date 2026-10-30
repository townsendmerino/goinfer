//go:build cuda

package cuda

import (
	"fmt"
	"sync"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"

	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The Whisper-style audio encoder on CUDA (docs/tasks/task-multimodal-support-2026-10.md, S14.6 lever B): aikit's audio.WhisperEncoder (Whisper's, and Voxtral's tower) run in
// float32 on the SigLIP tower's machinery: the two Conv1d stems as im2col + GEMM (audio_encoder.cu), the learned positions added, pre-LayerNorm layers (biased q, v and out; no bias
// on k) with the fused attention kernel, an exact-GELU MLP, a final LayerNorm. Nothing is quantized: the result is held to the CPU encoder by cosine, not by bits.

func init() {
	multimodal.RegisterWhisperDevice("cuda", func(enc *audio.WhisperEncoder) (audio.WhisperDevice, error) {
		t, err := newAudioTower(enc)
		if err == nil {
			return t, nil
		}
		// The layers do not fit next to what the device already holds (a resident decoder takes most of an 8 GB card): stream them,
		// one layer's weights at a time over the bus, rather than hand the window back to the CPU.
		t2, err2 := newAudioTowerMode(enc, true)
		if err2 != nil {
			return nil, fmt.Errorf("%w (and streamed: %v)", err, err2)
		}
		return t2, nil
	})
}

// audioDefect numbers the planted defects the same way audio.WhisperEncoder's are (SetDefectForTest), so one list drives both encoders.
type audioDefect int

const (
	audioDefectNone audioDefect = iota
	audioDefectNoPos
	audioDefectPosShifted
	audioDefectNoScale
	audioDefectTanhGELU
	audioDefectNoFinalNorm
	audioDefectWindowed // not injectable on the device: the fused attention kernel takes whole buffers, with no offset view to give a window
	audioDefectPostNorm
)

type audioLayer struct {
	ln1W, ln1B, ln2W, ln2B Buffer
	q, k, v, o, fc1, fc2   gridProj
}

type audioTower struct {
	g               *gridTower
	cfg             audio.WhisperEncoderConfig
	pos             []float32 // kept for the shifted-positions defect
	posB            Buffer
	conv1, conv2    gridProj
	layers          []audioLayer // every layer's weights on the device; streamed: one slot, refilled per layer from host
	host            []audio.WhisperLayerWeights
	streamed        bool
	lnW, lnB        Buffer
	geluErf, im2col Pipeline
	planted         audioDefect
	mu              sync.Mutex
}

// Placement says where the weights live, for the one line serve prints about the encoder.
func (t *audioTower) Placement() string {
	if t.streamed {
		return "layers streamed from host, one at a time"
	}
	return "weights resident"
}

// newAudioTower holds every layer's weights on the device.
func newAudioTower(enc *audio.WhisperEncoder) (*audioTower, error) {
	return newAudioTowerMode(enc, false)
}

// newAudioTowerMode is newAudioTower, or with streamed true a tower that holds one layer's buffers and refills them from the host's
// copy before each layer of every window: the same kernels on the same bytes, so the result is bit-identical to the resident tower's,
// for the price of the weights crossing the bus once per window.
func newAudioTowerMode(enc *audio.WhisperEncoder, streamed bool) (*audioTower, error) {
	w := enc.Weights()
	c := w.Cfg
	if len(w.Layers) != c.Layers || len(w.Pos) != c.MaxPositions*c.DModel {
		return nil, fmt.Errorf("cuda: Whisper encoder export: %d layers for %d, %d position values for %d x %d", len(w.Layers), c.Layers, len(w.Pos), c.MaxPositions, c.DModel)
	}
	g, err := newGridTower(gridSiglip, c.DModel, c.FFN, c.Heads, 0, float64(w.LayerNormEps))
	if err != nil {
		return nil, err
	}
	t := &audioTower{g: g, cfg: c, pos: w.Pos, streamed: streamed, host: w.Layers}
	o := g.ops
	if _, err := g.build(func() error {
		lib, e := o.dev.CompileLibrary(audioEncoderPTX)
		if e != nil {
			return fmt.Errorf("compile audio_encoder: %w", e)
		}
		if t.geluErf, e = o.dev.NewComputePipeline(lib, "audio_gelu_erf"); e != nil {
			return e
		}
		if t.im2col, e = o.dev.NewComputePipeline(lib, "audio_im2col_k3"); e != nil {
			return e
		}
		D := c.DModel
		p := func(wt, b []float32, out, in int) gridProj {
			return gridProj{w: o.up(wt), b: o.up(b), out: out, in: in}
		}
		t.conv1, t.conv2 = p(w.Conv1W, w.Conv1B, D, c.MelBins*3), p(w.Conv2W, w.Conv2B, D, D*3)
		t.posB = o.up(w.Pos)
		if streamed {
			t.layers = []audioLayer{t.emptyLayer()}
		} else {
			for _, l := range w.Layers {
				t.layers = append(t.layers, audioLayer{ln1W: o.up(l.LN1W), ln1B: o.up(l.LN1B), ln2W: o.up(l.LN2W), ln2B: o.up(l.LN2B),
					q: p(l.QW, l.QB, D, D), k: p(l.KW, nil, D, D), v: p(l.VW, l.VB, D, D), o: p(l.OW, l.OB, D, D), fc1: p(l.FC1W, l.FC1B, c.FFN, D), fc2: p(l.FC2W, l.FC2B, D, c.FFN)})
			}
		}
		t.lnW, t.lnB = o.up(w.FinalLNW), o.up(w.FinalLNB)
		return nil
	}); err != nil {
		return nil, err
	}
	return t, nil
}

// block is the layer as the tower walker's block.
func (l *audioLayer) block() gridBlock {
	return gridBlock{norm1w: l.ln1W, norm1b: l.ln1B, norm2w: l.ln2W, norm2b: l.ln2B, q: l.q, k: l.k, v: l.v, proj: l.o, fc1: l.fc1, fc2: l.fc2}
}

// emptyLayer allocates the device buffers one layer's tensors fill, for streaming. Runs on the executor.
func (t *audioTower) emptyLayer() audioLayer {
	c, o := t.cfg, t.g.ops
	D, F := c.DModel, c.FFN
	buf := func(n int) Buffer { return gpu.NewBufferLenOfOn[float32](o.dev, mem.LineCompanion, n) }
	p := func(out, in int, bias bool) gridProj {
		r := gridProj{w: buf(out * in), out: out, in: in}
		if bias {
			r.b = buf(out)
		}
		return r
	}
	return audioLayer{ln1W: buf(D), ln1B: buf(D), ln2W: buf(D), ln2B: buf(D),
		q: p(D, D, true), k: p(D, D, false), v: p(D, D, true), o: p(D, D, true), fc1: p(F, D, true), fc2: p(D, F, true)}
}

// fillLayer copies one layer's tensors from the host into the streaming slot. gpu.Upload waits for the kernels already queued before it
// copies, so the layer being replaced has finished when its bytes are overwritten.
func (t *audioTower) fillLayer(dst *audioLayer, h *audio.WhisperLayerWeights) error {
	for _, u := range []struct {
		b Buffer
		x []float32
	}{{dst.ln1W, h.LN1W}, {dst.ln1B, h.LN1B}, {dst.ln2W, h.LN2W}, {dst.ln2B, h.LN2B},
		{dst.q.w, h.QW}, {dst.q.b, h.QB}, {dst.k.w, h.KW}, {dst.v.w, h.VW}, {dst.v.b, h.VB}, {dst.o.w, h.OW}, {dst.o.b, h.OB},
		{dst.fc1.w, h.FC1W}, {dst.fc1.b, h.FC1B}, {dst.fc2.w, h.FC2W}, {dst.fc2.b, h.FC2B}} {
		if err := gpu.Upload(u.b, u.x); err != nil {
			return err
		}
	}
	return nil
}

// gelu is the encoder's activation: the exact kernel, or the tanh form under the planted defect.
func (t *audioTower) gelu(x Buffer, n int) {
	if t.planted == audioDefectTanhGELU {
		t.g.ops.geluTanh(x, n)
		return
	}
	t.g.ops.launch(t.geluErf, gpu.Grid1D(n, 256), Arg(x), i64(int64(n)))
}

// conv runs one k=3 pad=1 Conv1d: im2col from src (channel c at chStride, frame t at tStride) into col, then col · Wᵀ + bias into out.
func (t *audioTower) conv(src, col, out Buffer, p *gridProj, Tin, Tout, stride, chStride, tStride int) {
	C := p.in / 3
	n := Tout * C * 3
	t.g.ops.launch(t.im2col, gpu.Grid1D(n, 256), Arg(src), Arg(col), i32(int32(Tin)), i32(int32(stride)), i32(int32(C)), i64(int64(chStride)), i64(int64(tStride)), i64(int64(n)))
	t.g.linear(col, p, out, Tout)
}

// Forward is audio.WhisperDevice: feats [mels][3000] to the [1500][DModel] last hidden state.
func (t *audioTower) Forward(feats []float32) ([]float32, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	c := t.cfg
	const T = audio.WhisperFrames
	if len(feats) != c.MelBins*T {
		return nil, fmt.Errorf("cuda: %d feature values, want %d mels x %d frames", len(feats), c.MelBins, T)
	}
	if t.planted == audioDefectWindowed {
		return nil, fmt.Errorf("cuda: the windowed-attention defect cannot be injected on the device")
	}
	D, N, F := c.DModel, c.MaxPositions, c.FFN
	var out []float32
	err := t.g.ops.do(func() error {
		defer t.g.ops.releaseScratch()
		o, a := t.g.ops, t.g
		fb, col1, x1 := o.af(c.MelBins*T), o.af(T*c.MelBins*3), o.af(T*D)
		if e := gpu.Upload(fb, feats); e != nil {
			return e
		}
		t.conv(fb, col1, x1, &t.conv1, T, T, 1, T, 1)
		t.gelu(x1, T*D)
		col2, x, xn := o.af(N*D*3), o.af(N*D), o.af(N*D)
		t.conv(x1, col2, x, &t.conv2, T, N, 2, 1, D)
		t.gelu(x, N*D)
		switch t.planted {
		case audioDefectNoPos:
		case audioDefectPosShifted:
			sh := make([]float32, len(t.pos))
			for i := range N {
				copy(sh[i*D:(i+1)*D], t.pos[((i+1)%N)*D:((i+1)%N+1)*D])
			}
			pb := o.af(N * D)
			if e := gpu.Upload(pb, sh); e != nil {
				return e
			}
			o.addVec(x, pb, N*D)
		default:
			o.addVec(x, t.posB, N*D)
		}
		q, k, v, ctx, tmp, f1 := o.af(N*D), o.af(N*D), o.af(N*D), o.af(N*D), o.af(N*D), o.af(N*F)
		env := &towerWalkEnv{geluErf: t.gelu}
		if t.planted == audioDefectNoScale {
			env.scale = 1
		}
		prog := audio.WhisperProgram(c)
		if t.planted == audioDefectPostNorm {
			prog = audio.WhisperPostNormProgramForTest(c)
		}
		regs := &towerRegs{x: x, xn: xn, q: q, k: k, v: v, ctx: ctx, mid: f1, tmp: tmp}
		for li := range t.host {
			l := &t.layers[min(li, len(t.layers)-1)]
			if t.streamed {
				if e := t.fillLayer(l, &t.host[li]); e != nil {
					return e
				}
			}
			b := l.block()
			if e := a.walkBlock(prog, &b, regs, N, env); e != nil {
				return fmt.Errorf("layer %d: %w", li, e)
			}
		}
		res := x
		if t.planted != audioDefectNoFinalNorm {
			o.layerNorm(x, t.lnW, t.lnB, xn, N, D, a.eps)
			res = xn
		}
		if e := o.finish(); e != nil {
			return e
		}
		out = make([]float32, N*D)
		return gpu.Download(res, out)
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: Whisper encoder: %w", err)
	}
	return out, nil
}

// Close releases the device.
func (t *audioTower) Close() error { return t.g.Close() }

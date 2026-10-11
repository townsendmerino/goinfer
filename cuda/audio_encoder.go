//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"sync"

	gpu "github.com/townsendmerino/aikit/gpu"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The Whisper-style audio encoder on CUDA (docs/tasks/task-multimodal-support-2026-10.md, S14.6 lever B): aikit's audio.WhisperEncoder (Whisper's, and Voxtral's tower) run in
// float32 on the SigLIP tower's machinery: the two Conv1d stems as im2col + GEMM (audio_encoder.cu), the learned positions added, pre-LayerNorm layers (biased q, v and out; no bias
// on k) with the fused attention kernel, an exact-GELU MLP, a final LayerNorm. Nothing is quantized: the result is held to the CPU encoder by cosine, not by bits.

func init() {
	multimodal.RegisterWhisperDevice("cuda", func(enc *audio.WhisperEncoder) (audio.WhisperDevice, error) {
		return newAudioTower(enc)
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
	layers          []audioLayer
	lnW, lnB        Buffer
	geluErf, im2col Pipeline
	planted         audioDefect
	mu              sync.Mutex
}

func newAudioTower(enc *audio.WhisperEncoder) (*audioTower, error) {
	w := enc.Weights()
	c := w.Cfg
	if len(w.Layers) != c.Layers || len(w.Pos) != c.MaxPositions*c.DModel {
		return nil, fmt.Errorf("cuda: Whisper encoder export: %d layers for %d, %d position values for %d x %d", len(w.Layers), c.Layers, len(w.Pos), c.MaxPositions, c.DModel)
	}
	g, err := newGridTower(gridSiglip, c.DModel, c.FFN, c.Heads, 0, float64(w.LayerNormEps))
	if err != nil {
		return nil, err
	}
	t := &audioTower{g: g, cfg: c, pos: w.Pos}
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
		for _, l := range w.Layers {
			t.layers = append(t.layers, audioLayer{ln1W: o.up(l.LN1W), ln1B: o.up(l.LN1B), ln2W: o.up(l.LN2W), ln2B: o.up(l.LN2B),
				q: p(l.QW, l.QB, D, D), k: p(l.KW, nil, D, D), v: p(l.VW, l.VB, D, D), o: p(l.OW, l.OB, D, D), fc1: p(l.FC1W, l.FC1B, c.FFN, D), fc2: p(l.FC2W, l.FC2B, D, c.FFN)})
		}
		t.lnW, t.lnB = o.up(w.FinalLNW), o.up(w.FinalLNB)
		return nil
	}); err != nil {
		return nil, err
	}
	return t, nil
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
	D, N, F, nH := c.DModel, c.MaxPositions, c.FFN, c.Heads
	hd := D / nH
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
		scale := float32(1 / math.Sqrt(float64(hd)))
		if t.planted == audioDefectNoScale {
			scale = 1
		}
		post := t.planted == audioDefectPostNorm
		for li := range t.layers {
			l := &t.layers[li]
			src := xn
			if post {
				src = x
			} else {
				o.layerNorm(x, l.ln1W, l.ln1B, xn, N, D, a.eps)
			}
			a.linear(src, &l.q, q, N)
			a.linear(src, &l.k, k, N)
			a.linear(src, &l.v, v, N)
			if e := o.attention(q, k, v, ctx, N, nH, hd, scale); e != nil {
				return e
			}
			a.linearAdd(ctx, &l.o, x, tmp, N)
			if post {
				o.layerNorm(x, l.ln1W, l.ln1B, xn, N, D, a.eps)
				x, xn = xn, x
			} else {
				o.layerNorm(x, l.ln2W, l.ln2B, xn, N, D, a.eps)
			}
			mlpIn := xn
			if post {
				mlpIn = x
			}
			a.linear(mlpIn, &l.fc1, f1, N)
			t.gelu(f1, N*F)
			a.linearAdd(f1, &l.fc2, x, tmp, N)
			if post {
				o.layerNorm(x, l.ln2W, l.ln2B, xn, N, D, a.eps)
				x, xn = xn, x
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

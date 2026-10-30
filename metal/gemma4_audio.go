//go:build darwin

package metal

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
	"math"
	"sync"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// The Gemma 4 audio tower's conformer blocks on Metal (Phase A's AM, docs/tasks/task-embeddinggemma2.md): aikit's
// audio.Gemma4AudioEncoder blocks in float32, from its exported Layers. The GEMMs and RMSNorm are aikit's ViT kernels
// and the ClippableLinear clamps the vision tower's (gemma4_vision.go); this file adds the SiLU, the weighted residual,
// the GLU, the causal depthwise conv, the query/key scales and a sliding-window attention kernel (one threadgroup per
// query and head over the Window keys behind it, relative-position keys added to each, softcap, softmax). The
// subsampler and the projections after the blocks stay aikit's, on the host. Registered as the "metal" audio
// accelerator.

func init() {
	embeddinggemma2.RegisterAudioAccelerator("metal", func(enc *audio.Gemma4AudioEncoder) (embeddinggemma2.AudioAccelerator, error) {
		return newG4AAccel(enc)
	})
}

type g4aPipes struct{ silu, axpy, glu, dwconv, scaleQK, attn Pipeline }

type g4aFF struct {
	pre, post Buffer
	up, down  g4vProj
}

type g4aLayer struct {
	ff1, ff2                          g4aFF
	normPreAttn, normPostAttn, out    Buffer
	q, k, v, post, convStart, convEnd g4vProj
	relk, qs, convW                   Buffer
	convPre, convNorm                 Buffer
}

// g4aAccel is the audio blocks on Metal. One clip at a time (mu).
type g4aAccel struct {
	mu sync.Mutex
	eg2Ops
	gp                                          g4vPipes
	ap                                          g4aPipes
	cfg                                         audio.Gemma4AudioConfig
	layers                                      []g4aLayer
	cap                                         int
	x, xn, cp, u, f, q, k, v, ctx, o, g, gl, cv Buffer
}

func newG4AAccel(enc *audio.Gemma4AudioEncoder) (*g4aAccel, error) {
	c := enc.Cfg
	H, nH := c.HiddenSize, c.NumAttentionHeads
	hd := H / nH
	if hd > 1024 || c.Window() > 64 || c.Window() < 1 {
		return nil, fmt.Errorf("metal: Gemma 4 audio head dim %d, window %d: the attention kernel takes head dim <= 1024 and a window of 1-64", hd, c.Window())
	}
	a := &g4aAccel{cfg: c}
	var err error
	a.eg2Ops, err = newEG2Ops(g4vMSL+g4aMSL, map[string]*Pipeline{"g4v_pos_add": &a.gp.posAdd, "g4v_clamp": &a.gp.clamp,
		"g4v_clamp_copy": &a.gp.clampCopy, "g4v_rope": &a.gp.rope, "g4a_silu": &a.ap.silu, "g4a_axpy": &a.ap.axpy,
		"g4a_glu": &a.ap.glu, "g4a_dwconv": &a.ap.dwconv, "g4a_scale_qk": &a.ap.scaleQK, "g4a_attn": &a.ap.attn})
	if err != nil {
		return nil, err
	}
	a.eps = float32(c.RMSNormEps)
	d := a.d
	up := func(v []float32) Buffer { return NewBufferFloatsOn(mem.LineCompanion, d, v) }
	proj := func(p audio.Gemma4Proj) g4vProj {
		return g4vProj{w: up(p.W), out: p.Out, in: p.In, inMin: p.InMin, inMax: p.InMax, outMin: p.OutMin, outMax: p.OutMax}
	}
	ff := func(f audio.Gemma4AudioFF) g4aFF {
		return g4aFF{pre: up(f.PreNorm), post: up(f.PostNorm), up: proj(f.Up), down: proj(f.Down)}
	}
	base := 1 / math.Ln2 / math.Sqrt(float64(hd))
	for _, l := range enc.Layers {
		qs := make([]float32, hd)
		for i := range qs {
			qs[i] = float32(base) * float32(math.Log1p(math.Exp(float64(l.PerDimScale[i]))))
		}
		a.layers = append(a.layers, g4aLayer{ff1: ff(l.FF1), ff2: ff(l.FF2), normPreAttn: up(l.NormPreAttn),
			normPostAttn: up(l.NormPostAttn), out: up(l.NormOut), q: proj(l.Q), k: proj(l.K), v: proj(l.V),
			post: proj(l.Post), convStart: proj(l.ConvStart), convEnd: proj(l.ConvEnd), relk: up(l.RelK), qs: up(qs),
			convW: up(l.ConvW), convPre: up(l.ConvPreNorm), convNorm: up(l.ConvNorm)})
	}
	return a, nil
}

func (a *g4aAccel) Name() string { return "metal" }

// Close frees the tower's device memory; a forward after it is an error.
func (a *g4aAccel) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closeDevice()
	a.layers, a.cap = nil, 0
	return nil
}

// grow sizes the scratch for n rows, releasing what it replaces.
func (a *g4aAccel) grow(n int) error {
	if err := a.open(); err != nil {
		return err
	}
	if n <= a.cap {
		return nil
	}
	a.release(&a.x, &a.xn, &a.cp, &a.u, &a.f, &a.q, &a.k, &a.v, &a.ctx, &a.o, &a.g, &a.gl, &a.cv)
	H := a.cfg.HiddenSize
	b := func(cols int) Buffer { return a.d.NewBufferLenOn(mem.LineCompanion, n*cols) }
	a.x, a.xn, a.cp, a.u, a.f = b(H), b(H), b(4*H), b(4*H), b(H)
	a.q, a.k, a.v, a.ctx, a.o = b(H), b(H), b(H), b(H), b(H)
	a.g, a.gl, a.cv = b(2*H), b(H), b(H)
	a.cap = n
	return nil
}

// proj is a ClippableLinear over rows of x into out (as g4vAccel.proj).
func (a *g4aAccel) proj(e *Encoder, x Buffer, p *g4vProj, out Buffer, rows int) {
	in := x
	if finite(p.inMin) || finite(p.inMax) {
		n := rows * p.in
		e.Dispatch(a.gp.clampCopy, n, 256, x, a.cp, a.u32(uint32(n)), a.f32(p.inMin), a.f32(p.inMax))
		in = a.cp
	}
	a.gemm(e, in, p.w, out, rows, p.out, p.in)
	if finite(p.outMin) || finite(p.outMax) {
		n := rows * p.out
		e.Dispatch(a.gp.clamp, n, 256, out, a.u32(uint32(n)), a.f32(p.outMin), a.f32(p.outMax))
	}
}

// Blocks runs every conformer block over h [n, hidden] (see embeddinggemma2.AudioAccelerator), a command buffer per
// block so the scalar arena is reused.
func (a *g4aAccel) Blocks(h []float32, n int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	c := a.cfg
	H := c.HiddenSize
	if n <= 0 || len(h) != n*H {
		return nil, fmt.Errorf("metal: %d values for %d rows of %d", len(h), n, H)
	}
	if err := a.grow(n); err != nil {
		return nil, err
	}
	copy(a.x.Floats()[:n*H], h)
	prog := audio.Gemma4AudioProgram(c)
	for li := range a.layers {
		a.slot = 0
		e := a.eg2Ops.q.Begin()
		if err := a.walkBlock(e, prog, &a.layers[li], n); err != nil {
			e.End()
			return nil, fmt.Errorf("metal: audio block %d: %w", li, err)
		}
		e.End()
		if err := e.Err(); err != nil {
			return nil, err
		}
	}
	return append([]float32(nil), a.x.Floats()[:n*H]...), nil
}

// reg is the tower's activation buffer for a program register (vision.TowerReg): x is the residual stream, mid is u
// ([rows, 4·hidden]), wide is g ([rows, 2·hidden]), the rest are [rows, hidden].
func (a *g4aAccel) reg(n vision.TowerReg) (Buffer, error) {
	switch n {
	case vision.TowerX:
		return a.x, nil
	case vision.TowerXN:
		return a.xn, nil
	case vision.TowerQ:
		return a.q, nil
	case vision.TowerK:
		return a.k, nil
	case vision.TowerV:
		return a.v, nil
	case vision.TowerCtx:
		return a.ctx, nil
	case vision.TowerMid:
		return a.u, nil
	case vision.TowerWide:
		return a.g, nil
	}
	return Buffer{}, fmt.Errorf("register %d unknown", n)
}

// walkBlock queues one block's program over n rows in e, each op through the dispatch the hand-ordered block used: a
// norm through rmsEps at the op's epsilon, a linear through proj with its clamps, SiLU through g4a_silu, the GLU through
// g4a_glu, the depthwise conv through g4a_dwconv, local attention as g4a_scale_qk then g4a_attn, the scaled add through
// g4a_axpy and the add through add. An op whose registers do not fit its kind (vision.TowerOp.CheckRegs), whose weight
// the layer lacks or does not fit its registers (CheckShape), or that does not fit this tower (an RMSNorm only, a window
// of 1 to 64 keys) is an error before it is queued.
func (a *g4aAccel) walkBlock(e *Encoder, prog vision.TowerBlock, ly *g4aLayer, n int) error {
	c := a.cfg
	H, nH := c.HiddenSize, c.NumAttentionHeads
	hd, I := H/nH, 4*H
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		var in, out Buffer
		var err error
		if op.ReadsIn() {
			if in, err = a.reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		if out, err = a.reg(op.Out); err != nil {
			return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
		}
		nH4 := n * H
		switch op.Kind {
		case vision.TowerNorm:
			w, ok := ly.towerNorm(op.W)
			if !ok || op.Form != vision.TowerRMSNorm {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			a.rmsEps(e, in, w, out, n, H, float32(op.Eps))
		case vision.TowerLinear:
			p, ok := ly.towerProj(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if err := op.CheckShape(p.out, p.in, H, I); err != nil {
				return fmt.Errorf("op %d: %w", i, err)
			}
			a.proj(e, in, p, out, n)
		case vision.TowerSiLU:
			m := n * vision.TowerRegWidth(op.Out, H, I)
			e.Dispatch(a.ap.silu, m, 256, out, a.u32(uint32(m)))
		case vision.TowerGLU:
			e.Dispatch(a.ap.glu, nH4, 256, in, out, a.u32(uint32(H)), a.u32(uint32(nH4)))
		case vision.TowerDWConv:
			e.Dispatch(a.ap.dwconv, nH4, 256, in, ly.convW, out, a.u32(uint32(H)), a.u32(uint32(c.ConvKernelSize)), a.u32(uint32(nH4)))
		case vision.TowerLocalAttention:
			if op.Window < 1 || op.Window > 64 || op.Window > c.Window() {
				return fmt.Errorf("op %d: local attention over %d keys; the kernel takes 1-64, the relative keys cover %d", i,
					op.Window, c.Window())
			}
			e.Dispatch(a.ap.scaleQK, nH4, 256, a.q, a.k, ly.qs, a.f32(float32(math.Log(1+math.E)/math.Ln2)), a.u32(uint32(hd)), a.u32(uint32(nH4)))
			e.Dispatch(a.ap.attn, n*nH*hd, hd, a.q, a.k, a.v, ly.relk, out, a.u32(uint32(H)), a.u32(uint32(nH)),
				a.u32(uint32(op.Window)), a.f32(float32(op.Scale)))
		case vision.TowerAddScaled:
			e.Dispatch(a.ap.axpy, nH4, 256, out, in, a.u32(uint32(nH4)), a.f32(float32(op.Scale)))
		case vision.TowerAdd:
			e.Dispatch(a.p.add, nH4, 256, out, in, a.u32(uint32(nH4)))
		default:
			return fmt.Errorf("op %d: kind %s not implemented on the Metal Gemma 4 audio tower", i, op.Kind)
		}
	}
	return nil
}

func (ly *g4aLayer) towerNorm(w vision.TowerWeight) (Buffer, bool) {
	switch w {
	case vision.TowerNormFF1Pre:
		return ly.ff1.pre, true
	case vision.TowerNormFF1Post:
		return ly.ff1.post, true
	case vision.TowerNormFF2Pre:
		return ly.ff2.pre, true
	case vision.TowerNormFF2Post:
		return ly.ff2.post, true
	case vision.TowerNorm1:
		return ly.normPreAttn, true
	case vision.TowerNormPostAttn:
		return ly.normPostAttn, true
	case vision.TowerNormConvPre:
		return ly.convPre, true
	case vision.TowerNormConv:
		return ly.convNorm, true
	case vision.TowerNormOut:
		return ly.out, true
	}
	return Buffer{}, false
}

func (ly *g4aLayer) towerProj(w vision.TowerWeight) (*g4vProj, bool) {
	switch w {
	case vision.TowerWFF1Up:
		return &ly.ff1.up, true
	case vision.TowerWFF1Down:
		return &ly.ff1.down, true
	case vision.TowerWFF2Up:
		return &ly.ff2.up, true
	case vision.TowerWFF2Down:
		return &ly.ff2.down, true
	case vision.TowerWQ:
		return &ly.q, true
	case vision.TowerWK:
		return &ly.k, true
	case vision.TowerWV:
		return &ly.v, true
	case vision.TowerWO:
		return &ly.post, true
	case vision.TowerWConvStart:
		return &ly.convStart, true
	case vision.TowerWConvEnd:
		return &ly.convEnd, true
	}
	return nil, false
}

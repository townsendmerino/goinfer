//go:build darwin

package metal

import (
	"fmt"
	"github.com/townsendmerino/goinfer/internal/mem"
	"math"
	"sync"

	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The Gemma 4 vision tower on Metal (Phase VM, docs/tasks/task-embeddinggemma2.md): aikit's Gemma4Encoder.Forward up
// to its pool, in float32, from aikit's export (Gemma4Encoder.Weights). The patch embed and the 16 layers run here,
// with the text encoder's GEMMs, RMSNorm and attention blocks (eg2Ops, embeddinggemma2.go: full attention, scale 1.0,
// one head per KV head), plus the position-table add, the ClippableLinear clamps and the axial 2-D RoPE below. The
// pool and the projection after them stay aikit's (FinishHidden), on the host. Registered as multimodal's "metal" Gemma 4
// tower, so Gemma 4's own image input (serve, --backend metal) and an EmbeddingGemma 2 encoder on Metal both run it.

func init() {
	multimodal.RegisterGemma4Tower("metal", func(enc *vision.Gemma4Encoder) (multimodal.Gemma4TowerAccelerator, error) {
		return newG4VAccel(enc)
	})
}

type g4vPipes struct{ posAdd, clamp, clampCopy, rope Pipeline }

// g4vProj is one ClippableLinear on the device: its [Out, In] weight and its bounds.
type g4vProj struct {
	w                            Buffer
	out, in                      int
	inMin, inMax, outMin, outMax float32
}

type g4vLayer struct {
	inNorm, postAttnNorm, preFFNorm, postFFNorm, qNorm, kNorm Buffer
	q, k, v, o, gate, up, down                                g4vProj
}

// g4vAccel is the tower on Metal. One image at a time (mu): its scratch and scalar arena are shared.
type g4vAccel struct {
	mu sync.Mutex
	eg2Ops
	gp                    g4vPipes
	cfg                   vision.Gemma4EncoderConfig
	theta                 float64
	patchW, posX, posY    Buffer
	ones                  Buffer // v's unweighted norm
	layers                []g4vLayer
	clampOff              bool // test seam: skip every clamp (a planted defect)
	cap                   int
	x, xn, cp, pt, posBuf Buffer
	q, k, v, ctx, o, g, u Buffer
	f, qh, kh, vh, sc, vt Buffer
	ob                    Buffer
	ropeCos, ropeSin      Buffer
}

func newG4VAccel(enc *vision.Gemma4Encoder) (*g4vAccel, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}
	c := w.Cfg
	if c.HeadDim%4 != 0 || c.NumAttentionHeads*c.HeadDim != c.HiddenSize {
		return nil, fmt.Errorf("metal: Gemma 4 tower %d heads of %d over hidden %d: the kernels need heads*hd = hidden, hd a multiple of 4",
			c.NumAttentionHeads, c.HeadDim, c.HiddenSize)
	}
	a := &g4vAccel{cfg: c, theta: w.RopeTheta}
	a.eg2Ops, err = newEG2Ops(g4vMSL, map[string]*Pipeline{"g4v_pos_add": &a.gp.posAdd, "g4v_clamp": &a.gp.clamp,
		"g4v_clamp_copy": &a.gp.clampCopy, "g4v_rope": &a.gp.rope})
	if err != nil {
		return nil, err
	}
	a.eps = float32(c.RMSNormEps)
	d := a.d
	up := func(v []float32) Buffer { return NewBufferFloatsOn(mem.LineCompanion, d, v) }
	proj := func(p vision.Gemma4Proj) g4vProj {
		return g4vProj{w: up(p.W), out: p.Out, in: p.In, inMin: p.InMin, inMax: p.InMax, outMin: p.OutMin, outMax: p.OutMax}
	}
	a.patchW, a.posX, a.posY = up(w.PatchEmbed), up(w.PosEmbX), up(w.PosEmbY)
	ones := make([]float32, c.HeadDim)
	for i := range ones {
		ones[i] = 1
	}
	a.ones = up(ones)
	for _, l := range w.Layers {
		a.layers = append(a.layers, g4vLayer{inNorm: up(l.InputNorm), postAttnNorm: up(l.PostAttnNorm), preFFNorm: up(l.PreFFNNorm),
			postFFNorm: up(l.PostFFNNorm), qNorm: up(l.QNorm), kNorm: up(l.KNorm), q: proj(l.Q), k: proj(l.K), v: proj(l.V),
			o: proj(l.O), gate: proj(l.Gate), up: proj(l.Up), down: proj(l.Down)})
	}
	return a, nil
}

func (a *g4vAccel) Name() string { return "metal" }

// Close frees the tower's device memory; a forward after it is an error.
func (a *g4vAccel) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closeDevice()
	a.layers, a.cap = nil, 0
	return nil
}

// grow sizes the scratch for np patches, releasing what it replaces.
func (a *g4vAccel) grow(np int) error {
	if err := a.open(); err != nil {
		return err
	}
	if np <= a.cap {
		return nil
	}
	a.release(&a.x, &a.xn, &a.cp, &a.pt, &a.posBuf, &a.q, &a.k, &a.v, &a.ctx, &a.o, &a.f, &a.g, &a.u, &a.qh, &a.kh, &a.vh,
		&a.sc, &a.vt, &a.ob, &a.ropeCos, &a.ropeSin)
	c := a.cfg
	H, I, hd := c.HiddenSize, c.IntermediateSize, c.HeadDim
	n := func(cols int) Buffer { return a.d.NewBufferLenOn(mem.LineCompanion, np*cols) }
	B := min(eg2Block, np)
	a.x, a.xn, a.cp, a.pt = n(H), n(H), n(max(H, I)), n(3*c.PatchSize*c.PatchSize)
	a.posBuf = a.d.NewBufferBytesOn(mem.LineCompanion, np*8)
	a.q, a.k, a.v, a.ctx, a.o, a.f = n(H), n(H), n(H), n(H), n(H), n(H)
	a.g, a.u = n(I), n(I)
	a.qh, a.kh, a.vh = n(H), n(H), n(H)
	a.sc, a.vt, a.ob = a.d.NewBufferLenOn(mem.LineCompanion, B*np), a.d.NewBufferLenOn(mem.LineCompanion, hd*np), a.d.NewBufferLenOn(mem.LineCompanion, B*hd)
	a.ropeCos, a.ropeSin = n(hd), n(hd)
	a.cap = np
	return nil
}

func finite(v float32) bool { return !math.IsInf(float64(v), 0) }

// proj runs one ClippableLinear over rows of x into out: the input clamp into a copy when its bounds are finite, the
// GEMM, then the output clamp.
func (a *g4vAccel) proj(e *Encoder, x Buffer, p *g4vProj, out Buffer, rows int) {
	in := x
	if !a.clampOff && (finite(p.inMin) || finite(p.inMax)) {
		n := rows * p.in
		e.Dispatch(a.gp.clampCopy, n, 256, x, a.cp, a.u32(uint32(n)), a.f32(p.inMin), a.f32(p.inMax))
		in = a.cp
	}
	a.gemm(e, in, p.w, out, rows, p.out, p.in)
	if !a.clampOff && (finite(p.outMin) || finite(p.outMax)) {
		n := rows * p.out
		e.Dispatch(a.gp.clamp, n, 256, out, a.u32(uint32(n)), a.f32(p.outMin), a.f32(p.outMax))
	}
}

// Hidden runs the patch embed and every layer (see embeddinggemma2.VisionAccelerator). A command buffer per layer, so
// the scalar arena is reused from layer to layer.
func (a *g4vAccel) Hidden(patches []float32, pos [][2]int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	c := a.cfg
	H, hd := c.HiddenSize, c.HeadDim
	np, pd := len(pos), 3*c.PatchSize*c.PatchSize
	if np == 0 || len(patches) != np*pd {
		return nil, fmt.Errorf("metal: %d patch values for %d patches of %d", len(patches), np, pd)
	}
	if err := a.grow(np); err != nil {
		return nil, err
	}
	pt := a.pt.Floats()[:np*pd]
	for i, v := range patches {
		pt[i] = 2 * (v - 0.5) // the patch embedder's own remap, as Forward does it
	}
	pb := a.posBuf.U32s()[:2*np]
	for i, p := range pos {
		if p[0] < 0 || p[0] >= c.PosEmbTableSize || p[1] < 0 || p[1] >= c.PosEmbTableSize {
			return nil, fmt.Errorf("metal: position id (%d,%d) out of range [0,%d)", p[0], p[1], c.PosEmbTableSize)
		}
		pb[2*i], pb[2*i+1] = uint32(p[0]), uint32(p[1])
	}
	cs, sn := vision.Gemma4RopeTables(pos, hd, a.theta)
	copy(a.ropeCos.Floats(), cs)
	copy(a.ropeSin.Floats(), sn)
	a.slot = 0
	e := a.eg2Ops.q.Begin()
	a.gemm(e, a.pt, a.patchW, a.x, np, H, pd)
	e.Dispatch(a.gp.posAdd, np*H, 256, a.x, a.posX, a.posY, a.posBuf, a.u32(uint32(H)), a.u32(uint32(np*H)))
	if len(a.layers) == 0 {
		e.End()
		if err := e.Err(); err != nil {
			return nil, err
		}
	}
	prog := vision.Gemma4Program(c)
	for li := range a.layers {
		if err := a.walkBlock(e, prog, &a.layers[li], np); err != nil {
			e.End()
			return nil, fmt.Errorf("metal: Gemma 4 tower block %d: %w", li, err)
		}
		e.End()
		if err := e.Err(); err != nil {
			return nil, err
		}
		a.slot = 0
		if li+1 < len(a.layers) {
			e = a.eg2Ops.q.Begin()
		}
	}
	return append([]float32(nil), a.x.Floats()[:np*H]...), nil
}

// reg is the tower's activation buffer for a program register (vision.TowerReg): x is the residual stream, mid and mid2
// are the gate and up products g and u ([rows, intermediate]), the rest are [rows, hidden].
func (a *g4vAccel) reg(n vision.TowerReg) (Buffer, error) {
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
		return a.g, nil
	case vision.TowerMid2:
		return a.u, nil
	}
	return Buffer{}, fmt.Errorf("register %d unknown", n)
}

// walkBlock queues one block's program over rows in e, each op through the tower's existing dispatch: a norm and a
// per-head norm through rmsEps at the op's epsilon (the ones buffer for the unscaled form), a linear through proj with
// its clamps, axial RoPE through g4v_rope on q or k, attention through the eg2 attention at scale 1.0, the GELU product
// through geluMul, a residual add through add. An op whose registers do not fit its kind (vision.TowerOp.CheckRegs), or
// that does not fit this tower (a per-head norm and RoPE on q or k, v's unscaled norm only, a linear's registers matching
// its weight's shape (vision.TowerOp.CheckShape), attention at scale 1.0), is an error before it is queued.
func (a *g4vAccel) walkBlock(e *Encoder, prog vision.TowerBlock, ly *g4vLayer, rows int) error {
	c := a.cfg
	H, I, nH, hd := c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads, c.HeadDim
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
		switch op.Kind {
		case vision.TowerNorm:
			w, ok := ly.towerNorm(op.W)
			if !ok || op.Form != vision.TowerRMSNorm {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			a.rmsEps(e, in, w, out, rows, H, float32(op.Eps))
		case vision.TowerHeadNorm:
			if op.Out != vision.TowerQ && op.Out != vision.TowerK && op.Out != vision.TowerV {
				return fmt.Errorf("op %d: the per-head norm runs on q, k or v, not register %d", i, op.Out)
			}
			w := a.ones
			switch op.Form {
			case vision.TowerRMSNormUnscaled:
				if op.W != 0 {
					return fmt.Errorf("op %d: the unscaled norm reads no weight, not weight %d", i, op.W)
				}
			case vision.TowerRMSNorm:
				var ok bool
				if w, ok = ly.towerNorm(op.W); !ok {
					return fmt.Errorf("op %d: per-head norm of weight %d", i, op.W)
				}
			default:
				return fmt.Errorf("op %d: per-head norm form %d", i, op.Form)
			}
			a.rmsEps(e, out, w, out, rows*nH, hd, float32(op.Eps))
		case vision.TowerLinear:
			p, ok := ly.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if err := op.CheckShape(p.out, p.in, H, I); err != nil {
				return fmt.Errorf("op %d: %w", i, err)
			}
			a.proj(e, in, p, out, rows)
		case vision.TowerRoPEAxial:
			if op.Out != vision.TowerQ && op.Out != vision.TowerK {
				return fmt.Errorf("op %d: RoPE runs on q or k, not register %d", i, op.Out)
			}
			nr := rows * nH * hd / 2
			e.Dispatch(a.gp.rope, nr, 64, out, a.ropeCos, a.ropeSin, a.u32(uint32(nH)), a.u32(uint32(hd)), a.u32(uint32(nr)))
		case vision.TowerAttention:
			if op.Scale != 1 {
				return fmt.Errorf("op %d: the tower's attention runs at scale 1, not %g", i, op.Scale)
			}
			a.attention(e, eg2AttnBufs{q: a.q, k: a.k, v: a.v, qh: a.qh, kh: a.kh, vh: a.vh, sc: a.sc, vt: a.vt, ob: a.ob, ctx: out},
				rows, nH, hd, nH, 0)
		case vision.TowerGELUMul:
			n := rows * I
			e.Dispatch(a.p.geluMul, n, 256, out, in, a.u32(uint32(n)))
		case vision.TowerAdd:
			n := rows * H
			e.Dispatch(a.p.add, n, 256, out, in, a.u32(uint32(n)))
		default:
			return fmt.Errorf("op %d: kind %s not implemented on the Metal Gemma 4 tower", i, op.Kind)
		}
	}
	return nil
}

func (ly *g4vLayer) towerNorm(w vision.TowerWeight) (Buffer, bool) {
	switch w {
	case vision.TowerNorm1:
		return ly.inNorm, true
	case vision.TowerNormPostAttn:
		return ly.postAttnNorm, true
	case vision.TowerNorm2:
		return ly.preFFNorm, true
	case vision.TowerNormPostMLP:
		return ly.postFFNorm, true
	case vision.TowerQNorm:
		return ly.qNorm, true
	case vision.TowerKNorm:
		return ly.kNorm, true
	}
	return Buffer{}, false
}

func (ly *g4vLayer) towerLinear(w vision.TowerWeight) (*g4vProj, bool) {
	switch w {
	case vision.TowerWQ:
		return &ly.q, true
	case vision.TowerWK:
		return &ly.k, true
	case vision.TowerWV:
		return &ly.v, true
	case vision.TowerWO:
		return &ly.o, true
	case vision.TowerWGate:
		return &ly.gate, true
	case vision.TowerWUp:
		return &ly.up, true
	case vision.TowerWDown:
		return &ly.down, true
	}
	return nil, false
}

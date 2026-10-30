//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"sync"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The Gemma 4 vision tower on CUDA (docs/tasks/task-multimodal-support-2026-10.md): aikit's Gemma4Encoder.Forward up to
// its pool, in float32, from aikit's export (Gemma4Encoder.Weights), on the CUDA tower base (tower_base.go). The patch
// embed and the layers run here; the pool and the projection after them stay aikit's (FinishHidden), on the host. The
// port of metal/gemma4_vision.go: the same ops in the same order, with aikit's bidirectional attention kernel (scale
// 1.0, one head per KV head, no window) in place of Metal's matmul-blocked attention. Registered as multimodal's "cuda"
// Gemma 4 tower.

func init() {
	multimodal.RegisterGemma4Tower("cuda", func(enc *vision.Gemma4Encoder) (multimodal.Gemma4TowerAccelerator, error) {
		return newG4Tower(enc)
	})
}

// g4Proj is one ClippableLinear on the device: its [Out, In] weight and its bounds.
type g4Proj struct {
	w                            Buffer
	out, in                      int
	inMin, inMax, outMin, outMax float32
}

type g4Layer struct {
	inNorm, postAttnNorm, preFFNorm, postFFNorm, qNorm, kNorm Buffer
	q, k, v, o, gate, up, down                                g4Proj
}

// g4Defect is the test seam for planted defects: each bool breaks one thing the tower must get right, so a gate can show
// it notices.
type g4Defect struct {
	noClamp       bool // skip every ClippableLinear clamp
	swapPosXY     bool // add the Y table by x and the X table by y
	weightedVNorm bool // v's norm takes the layer's k weight instead of being unweighted
	scaleRoot     bool // attention scale 1/sqrt(hd) instead of 1.0
	noPosAdd      bool // skip the position tables
	noRope        bool // skip the axial RoPE
}

// g4Tower is the tower on CUDA. One image at a time (mu): its scratch is per call.
type g4Tower struct {
	mu    sync.Mutex
	ops   *towerOps
	cfg   vision.Gemma4EncoderConfig
	theta float64

	patchW, posX, posY, ones Buffer
	layers                   []g4Layer
	defect                   g4Defect
}

func newG4Tower(enc *vision.Gemma4Encoder) (tw *g4Tower, err error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("cuda: %w", err)
	}
	c := w.Cfg
	if c.HeadDim%4 != 0 || c.NumAttentionHeads*c.HeadDim != c.HiddenSize {
		return nil, fmt.Errorf("cuda: Gemma 4 tower %d heads of %d over hidden %d: the kernels need heads*hd = hidden, hd a multiple of 4",
			c.NumAttentionHeads, c.HeadDim, c.HiddenSize)
	}
	ops, err := newTowerOps(max(c.HiddenSize, c.IntermediateSize))
	if err != nil {
		return nil, err
	}
	tw = &g4Tower{ops: ops, cfg: c, theta: w.RopeTheta}
	if err := ops.do(func() error {
		up := ops.up
		proj := func(p vision.Gemma4Proj) g4Proj {
			return g4Proj{w: up(p.W), out: p.Out, in: p.In, inMin: p.InMin, inMax: p.InMax, outMin: p.OutMin, outMax: p.OutMax}
		}
		tw.patchW, tw.posX, tw.posY = up(w.PatchEmbed), up(w.PosEmbX), up(w.PosEmbY)
		ones := make([]float32, c.HeadDim)
		for i := range ones {
			ones[i] = 1
		}
		tw.ones = up(ones)
		for _, l := range w.Layers {
			tw.layers = append(tw.layers, g4Layer{inNorm: up(l.InputNorm), postAttnNorm: up(l.PostAttnNorm), preFFNorm: up(l.PreFFNNorm),
				postFFNorm: up(l.PostFFNNorm), qNorm: up(l.QNorm), kNorm: up(l.KNorm), q: proj(l.Q), k: proj(l.K), v: proj(l.V),
				o: proj(l.O), gate: proj(l.Gate), up: proj(l.Up), down: proj(l.Down)})
		}
		return ops.finish()
	}); err != nil {
		ops.close()
		return nil, fmt.Errorf("cuda: Gemma 4 tower upload: %w", err)
	}
	return tw, nil
}

func (a *g4Tower) Name() string { return "cuda" }

func (a *g4Tower) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.ops.close()
	return nil
}

func finite32(v float32) bool { return !math.IsInf(float64(v), 0) }

// proj runs one ClippableLinear over rows of x into out: the input clamp into the copy buffer cp when its bounds are finite, the GEMM, then the output clamp.
func (a *g4Tower) proj(x Buffer, p *g4Proj, out, cp Buffer, rows int) {
	in := x
	if !a.defect.noClamp && (finite32(p.inMin) || finite32(p.inMax)) {
		a.ops.clampCopyTo(x, cp, rows*p.in, p.inMin, p.inMax)
		in = cp
	}
	a.ops.gemm(in, p.w, out, rows, p.out, p.in)
	if !a.defect.noClamp && (finite32(p.outMin) || finite32(p.outMax)) {
		a.ops.clampInPlace(out, rows*p.out, p.outMin, p.outMax)
	}
}

// Hidden runs the patch embed and every layer (multimodal.Gemma4TowerAccelerator): the last layer's output, [np, hidden], for aikit's FinishHidden.
func (a *g4Tower) Hidden(patches []float32, pos [][2]int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	c := a.cfg
	H, I, hd := c.HiddenSize, c.IntermediateSize, c.HeadDim
	np, pd := len(pos), 3*c.PatchSize*c.PatchSize
	if np == 0 || len(patches) != np*pd {
		return nil, fmt.Errorf("cuda: %d patch values for %d patches of %d", len(patches), np, pd)
	}
	pt := make([]float32, len(patches))
	for i, v := range patches {
		pt[i] = 2 * (v - 0.5) // the patch embedder's own remap, as Forward does it
	}
	pb := make([]int32, 2*np)
	for i, p := range pos {
		if p[0] < 0 || p[0] >= c.PosEmbTableSize || p[1] < 0 || p[1] >= c.PosEmbTableSize {
			return nil, fmt.Errorf("cuda: position id (%d,%d) out of range [0,%d)", p[0], p[1], c.PosEmbTableSize)
		}
		pb[2*i], pb[2*i+1] = int32(p[0]), int32(p[1])
	}
	cs, sn := vision.Gemma4RopeTables(pos, hd, a.theta)
	var out []float32
	err := a.ops.do(func() error {
		defer a.ops.releaseScratch()
		o := a.ops
		af := o.af
		x, xn := af(np*H), af(np*H)
		cp := af(np * max(H, I))
		ptB := af(np * pd)
		q, k, v, ctx := af(np*H), af(np*H), af(np*H), af(np*H)
		g, u := af(np*I), af(np*I)
		csB, snB := af(np*hd), af(np*hd)
		posB := o.ai(2 * np)
		for _, up := range []struct {
			b Buffer
			v []float32
		}{{ptB, pt}, {csB, cs}, {snB, sn}} {
			if e := gpu.Upload(up.b, up.v); e != nil {
				return e
			}
		}
		if e := gpu.Upload(posB, pb); e != nil {
			return e
		}
		o.gemm(ptB, a.patchW, x, np, H, pd)
		if !a.defect.noPosAdd {
			pxT, pyT := a.posX, a.posY
			if a.defect.swapPosXY {
				pxT, pyT = a.posY, a.posX
			}
			o.posAddTo(x, pxT, pyT, posB, np, H)
		}
		regs := &towerRegs{x: x, xn: xn, q: q, k: k, v: v, ctx: ctx, mid: g, mid2: u}
		prog := vision.Gemma4Program(c)
		for li := range a.layers {
			if e := a.walkBlock(prog, &a.layers[li], regs, cp, csB, snB, np); e != nil {
				return fmt.Errorf("block %d: %w", li, e)
			}
		}
		if e := o.finish(); e != nil {
			return e
		}
		out = make([]float32, np*H)
		return gpu.Download(x, out)
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: Gemma 4 tower: %w", err)
	}
	return out, nil
}

// walkBlock queues one block's program over rows, each op through the tower's existing operation: a norm and a per-head
// norm through towerOps.rms at the op's epsilon (the ones buffer for v's unscaled form), a linear through proj with its
// clamps (cp is its input copy), axial RoPE through towerOps.ropeAxialTo on q or k, attention over every row at the op's
// scale, the GELU product as geluTanh then mulVec, a residual add through addVec. An op that does not fit its kind
// (vision.TowerOp.CheckRegs), its weight (vision.TowerOp.CheckShape) or this tower is an error before it is queued. The
// planted defects (g4Defect) act at their ops.
func (a *g4Tower) walkBlock(prog vision.TowerBlock, ly *g4Layer, r *towerRegs, cp, cs, sn Buffer, rows int) error {
	c := a.cfg
	o := a.ops
	H, I, nH, hd := c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads, c.HeadDim
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		var in, out Buffer
		var err error
		if op.ReadsIn() {
			if in, err = r.reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		if out, err = r.reg(op.Out); err != nil {
			return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
		}
		switch op.Kind {
		case vision.TowerNorm:
			w, ok := ly.towerNorm(op.W)
			if !ok || op.Form != vision.TowerRMSNorm {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			o.rms(in, w, out, rows, H, float32(op.Eps))
		case vision.TowerHeadNorm:
			if op.Out != vision.TowerQ && op.Out != vision.TowerK && op.Out != vision.TowerV {
				return fmt.Errorf("op %d: the per-head norm runs on q, k or v, not register %d", i, op.Out)
			}
			w := a.ones
			switch op.Form {
			case vision.TowerRMSNormUnscaled: // CheckRegs: it names no weight
				if a.defect.weightedVNorm {
					w = ly.kNorm
				}
			case vision.TowerRMSNorm:
				var ok bool
				if w, ok = ly.towerNorm(op.W); !ok {
					return fmt.Errorf("op %d: per-head norm of weight %d", i, op.W)
				}
			default:
				return fmt.Errorf("op %d: per-head norm form %d", i, op.Form)
			}
			o.rms(out, w, out, rows*nH, hd, float32(op.Eps))
		case vision.TowerLinear:
			p, ok := ly.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if err := op.CheckShape(p.out, p.in, H, I); err != nil {
				return fmt.Errorf("op %d: %w", i, err)
			}
			a.proj(in, p, out, cp, rows)
		case vision.TowerRoPEAxial:
			if op.Out != vision.TowerQ && op.Out != vision.TowerK {
				return fmt.Errorf("op %d: RoPE runs on q or k, not register %d", i, op.Out)
			}
			if !a.defect.noRope {
				o.ropeAxialTo(out, cs, sn, rows, nH, hd)
			}
		case vision.TowerAttention:
			scale := float32(op.Scale)
			if a.defect.scaleRoot {
				scale = float32(1 / math.Sqrt(float64(hd)))
			}
			if err := o.attention(r.q, r.k, r.v, out, rows, nH, hd, scale); err != nil {
				return err
			}
		case vision.TowerGELUMul:
			o.geluTanh(out, rows*I)
			o.mulVec(out, in, rows*I)
		case vision.TowerAdd:
			o.addVec(out, in, rows*H)
		default:
			return fmt.Errorf("op %d: kind %s not implemented on the CUDA Gemma 4 tower", i, op.Kind)
		}
	}
	return nil
}

func (ly *g4Layer) towerNorm(w vision.TowerWeight) (Buffer, bool) {
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

func (ly *g4Layer) towerLinear(w vision.TowerWeight) (*g4Proj, bool) {
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

//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"sync"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The Gemma 4 vision tower on CUDA (S4 of docs/tasks/task-multimodal-support-2026-10.md): aikit's Gemma4Encoder.Forward up to its pool, in float32, from aikit's
// export (Gemma4Encoder.Weights), on the CUDA tower base (tower_base.go). The patch embed and the 16 layers run here; the pool and the projection after them stay
// aikit's (FinishHidden), on the host. The port of metal/gemma4_vision.go: the same ops in the same order, with aikit's bidirectional attention kernel (scale 1.0,
// one head per KV head, no window) in place of Metal's matmul-blocked attention. Registered as multimodal's "cuda" Gemma 4 tower.

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

// g4Defect is the test seam for planted defects (G-S2c's CUDA list): each bool breaks one thing the tower must get right, so a gate can show it notices.
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
	H, I, nH, hd := c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads, c.HeadDim
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
	scale := float32(1)
	if a.defect.scaleRoot {
		scale = float32(1 / math.Sqrt(float64(hd)))
	}
	var out []float32
	err := a.ops.do(func() error {
		defer a.ops.releaseScratch()
		o := a.ops
		af := o.af
		x, xn := af(np*H), af(np*H)
		cp := af(np * max(H, I))
		ptB := af(np * pd)
		q, k, v, ctx, ob, f := af(np*H), af(np*H), af(np*H), af(np*H), af(np*H), af(np*H)
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
		for li := range a.layers {
			ly := &a.layers[li]
			eps := float32(c.RMSNormEps)
			o.rms(x, ly.inNorm, xn, np, H, eps)
			a.proj(xn, &ly.q, q, cp, np)
			a.proj(xn, &ly.k, k, cp, np)
			a.proj(xn, &ly.v, v, cp, np)
			o.rms(q, ly.qNorm, q, np*nH, hd, eps)
			o.rms(k, ly.kNorm, k, np*nH, hd, eps)
			vw := a.ones
			if a.defect.weightedVNorm {
				vw = ly.kNorm
			}
			o.rms(v, vw, v, np*nH, hd, eps)
			if !a.defect.noRope {
				o.ropeAxialTo(q, csB, snB, np, nH, hd)
				o.ropeAxialTo(k, csB, snB, np, nH, hd)
			}
			if e := o.attention(q, k, v, ctx, np, nH, hd, scale); e != nil {
				return e
			}
			a.proj(ctx, &ly.o, ob, cp, np)
			o.rms(ob, ly.postAttnNorm, ob, np, H, eps)
			o.addVec(x, ob, np*H)
			o.rms(x, ly.preFFNorm, xn, np, H, eps)
			a.proj(xn, &ly.gate, g, cp, np)
			a.proj(xn, &ly.up, u, cp, np)
			o.geluTanh(g, np*I)
			o.mulVec(g, u, np*I)
			a.proj(g, &ly.down, f, cp, np)
			o.rms(f, ly.postFFNorm, f, np, H, eps)
			o.addVec(x, f, np*H)
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

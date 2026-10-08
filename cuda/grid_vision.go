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

// The Qwen3.5+ and GLM-OCR vision towers on CUDA (S4 and S2's CUDA twins, docs/tasks/task-multimodal-support-2026-10.md): aikit's CPU Forward up to the merger, in
// float32, from aikit's export (Qwen3VisionEncoder.Weights, GlmOcrVisionEncoder.Weights), on the CUDA tower base. The port of metal/grid_vision.go with aikit's own
// scaled bidirectional attention kernel (one call per image frame, the frame being the attention segment) in place of Metal's matmul-blocked attention, and a
// separate q, k and v throughout (GLM-OCR norms q and k per head before RoPE, which a fused qkv buffer cannot do).
//
// Both towers' blocks: biased q, k and v (the fused [3*hidden, hidden] qkv split at upload), NeoX rotate-half RoPE from aikit's own tables, attention at scale
// 1/sqrt(head_dim) within each image frame, biased proj and residual, then the MLP and residual. Qwen3: LayerNorm with bias (eps 1e-6), MLP fc1 -> GELU-tanh -> fc2,
// and the interpolated position rows added after the patch embed. GLM-OCR: RMSNorm (eps rms_norm_eps), per-head RMSNorm of q and k before RoPE, MLP
// down(silu(gate) * up), no position table. The merger after the blocks stays aikit's (FinishHidden), on the host.

func init() {
	multimodal.RegisterQwen3Tower("cuda", func(enc *vision.Qwen3VisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newQwen3Tower(enc)
	})
	multimodal.RegisterGlmOcrTower("cuda", func(enc *vision.GlmOcrVisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newGlmOcrTower(enc)
	})
}

type gridKind uint8

const (
	gridQwen3  gridKind = iota // LayerNorm, GELU-tanh MLP, interpolated position rows, one segment per frame
	gridGlmOcr                 // RMSNorm, per-head q/k RMSNorm, SiLU-gated MLP, no position table, one segment per frame
	gridSiglip                 // Gemma 3's SigLIP: LayerNorm, GELU-tanh MLP, a fixed position table, NO RoPE, one segment over every patch (siglip_vision.go)
)

type gridProj struct {
	w, b    Buffer // [out, in]; bias [out] (zero Buffer when none)
	out, in int
}

type gridBlock struct {
	norm1w, norm1b, norm2w, norm2b Buffer // LayerNorm (Qwen3) or RMSNorm weight only (GLM-OCR: the b's are zero)
	q, k, v, proj                  gridProj
	qNorm, kNorm                   Buffer // GLM-OCR's per-head q/k RMSNorm; zero otherwise
	fc1, fc2                       gridProj
	gate, up, down                 gridProj
}

// gridDefect is the test seam for planted defects (G-S2c's list, Metal's five and CUDA's): each breaks one thing the tower must get right.
type gridDefect struct {
	noScale, swapRope, transposePos, noQKNorm, noPatchBias bool
	oneSegment                                             bool // every frame of a batch attends over every frame (the segment boundaries ignored)
	noPosEmbed                                             bool // SigLIP: the position table not added
}

type gridTower struct {
	mu                  sync.Mutex
	ops                 *towerOps
	kind                gridKind
	hidden, inter       int
	heads, hd, patchDim int
	eps                 float32
	patchW, patchB      Buffer
	blocks              []gridBlock
	posEmbeds           func([][3]int) []float32            // Qwen3's interpolated position rows; nil for GLM-OCR
	ropeTables          func([][3]int) (cos, sin []float32) // aikit's [np, head_dim] tables, rows cat(f, f)
	planted             gridDefect
}

func (a *gridTower) Name() string { return "cuda" }

func (a *gridTower) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.ops.close()
	return nil
}

func newGridTower(kind gridKind, hidden, inter, heads, patchDim int, eps float64) (*gridTower, error) {
	if heads <= 0 || hidden%heads != 0 || (hidden/heads)%2 != 0 {
		return nil, fmt.Errorf("cuda: vision tower hidden %d over %d heads: the kernels need an even head_dim dividing hidden", hidden, heads)
	}
	ops, err := newTowerOps(max(hidden, inter))
	if err != nil {
		return nil, err
	}
	return &gridTower{ops: ops, kind: kind, hidden: hidden, inter: inter, heads: heads, hd: hidden / heads, patchDim: patchDim, eps: float32(eps)}, nil
}

func (a *gridTower) proj(p vision.VisionProj) gridProj {
	return gridProj{w: a.ops.up(p.W), b: a.ops.up(p.B), out: p.Out, in: p.In}
}

// splitQKV uploads a fused [3*hidden, hidden] qkv projection (rows q, then k, then v) as three projections.
func (a *gridTower) splitQKV(p vision.VisionProj) (q, k, v gridProj, err error) {
	H := a.hidden
	if p.Out != 3*H || p.In != H || len(p.W) != 3*H*H {
		return q, k, v, fmt.Errorf("cuda: qkv is %dx%d, want %dx%d", p.Out, p.In, 3*H, H)
	}
	part := func(i int) gridProj {
		r := gridProj{w: a.ops.up(p.W[i*H*H : (i+1)*H*H]), out: H, in: H}
		if len(p.B) == 3*H {
			r.b = a.ops.up(p.B[i*H : (i+1)*H])
		}
		return r
	}
	return part(0), part(1), part(2), nil
}

// build runs upload on the executor, closing the ops and returning an error (never a panic) if the device cannot hold the weights.
func (a *gridTower) build(upload func() error) (*gridTower, error) {
	if err := a.ops.do(func() error {
		if err := upload(); err != nil {
			return err
		}
		return a.ops.finish()
	}); err != nil {
		a.ops.close()
		return nil, fmt.Errorf("cuda: vision tower upload: %w", err)
	}
	return a, nil
}

func newQwen3Tower(enc *vision.Qwen3VisionEncoder) (*gridTower, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("cuda: %w", err)
	}
	c := w.Cfg
	a, err := newGridTower(gridQwen3, c.HiddenSize, c.IntermediateSize, c.NumHeads, c.InChannels*c.TemporalPatchSize*c.PatchSize*c.PatchSize, w.LNEps)
	if err != nil {
		return nil, err
	}
	a.posEmbeds, a.ropeTables = enc.PositionEmbeds, enc.RopeTables
	return a.build(func() error {
		a.patchW, a.patchB = a.ops.up(w.PatchW), a.ops.up(w.PatchB)
		for _, b := range w.Blocks {
			q, k, v, err := a.splitQKV(b.QKV)
			if err != nil {
				return err
			}
			a.blocks = append(a.blocks, gridBlock{norm1w: a.ops.up(b.Norm1W), norm1b: a.ops.up(b.Norm1B), norm2w: a.ops.up(b.Norm2W), norm2b: a.ops.up(b.Norm2B),
				q: q, k: k, v: v, proj: a.proj(b.Proj), fc1: a.proj(b.FC1), fc2: a.proj(b.FC2)})
		}
		return nil
	})
}

func newGlmOcrTower(enc *vision.GlmOcrVisionEncoder) (*gridTower, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("cuda: %w", err)
	}
	c := w.Cfg
	a, err := newGridTower(gridGlmOcr, c.HiddenSize, c.IntermediateSize, c.NumHeads, c.InChannels*c.TemporalPatchSize*c.PatchSize*c.PatchSize, w.RMSEps)
	if err != nil {
		return nil, err
	}
	a.ropeTables = enc.RopeTables
	return a.build(func() error {
		a.patchW, a.patchB = a.ops.up(w.PatchW), a.ops.up(w.PatchB)
		for _, b := range w.Blocks {
			q, k, v, err := a.splitQKV(b.QKV)
			if err != nil {
				return err
			}
			if len(b.QNorm) != a.hd || len(b.KNorm) != a.hd {
				return fmt.Errorf("cuda: GLM-OCR q/k norm widths %d/%d, want head_dim %d", len(b.QNorm), len(b.KNorm), a.hd)
			}
			a.blocks = append(a.blocks, gridBlock{norm1w: a.ops.up(b.Norm1W), norm2w: a.ops.up(b.Norm2W), q: q, k: k, v: v, proj: a.proj(b.Proj),
				qNorm: a.ops.up(b.QNorm), kNorm: a.ops.up(b.KNorm), gate: a.proj(b.Gate), up: a.proj(b.Up), down: a.proj(b.Down)})
		}
		return nil
	})
}

// linear runs out = x * Wt (+ bias) over rows.
func (a *gridTower) linear(x Buffer, p *gridProj, out Buffer, rows int) {
	if p.b != (Buffer{}) {
		a.ops.gemmBias(x, p.w, p.b, out, rows, p.out, p.in)
		return
	}
	a.ops.gemm(x, p.w, out, rows, p.out, p.in)
}

// linearAdd runs resid += x * Wt (+ bias) over rows (tmp is [rows, out] scratch for the paths with no fused epilogue).
func (a *gridTower) linearAdd(x Buffer, p *gridProj, resid, tmp Buffer, rows int) {
	if p.b != (Buffer{}) {
		a.ops.gemmBiasAdd(x, p.w, p.b, resid, tmp, rows, p.out, p.in)
		return
	}
	a.ops.gemm(x, p.w, tmp, rows, p.out, p.in)
	a.ops.addVec(resid, tmp, rows*p.out)
}

// norm is the block's norm: LayerNorm with bias (Qwen3) or the weight-only RMSNorm (GLM-OCR).
func (a *gridTower) norm(x, w, b, out Buffer, rows int) {
	if a.kind == gridGlmOcr {
		a.ops.rms(x, w, out, rows, a.hidden, a.eps)
		return
	}
	a.ops.layerNorm(x, w, b, out, rows, a.hidden, a.eps)
}

// Hidden runs the patch embed and every block (multimodal.GridTowerAccelerator): the last block's output, [np, hidden], for aikit's FinishHidden.
func (a *gridTower) Hidden(pixels []float32, gridTHW [][3]int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	segs := vision.VisionSegments(gridTHW)
	np := segs[len(segs)-1]
	if np == 0 || len(pixels) != np*a.patchDim {
		return nil, fmt.Errorf("cuda: %d pixel values for %d patches of %d", len(pixels), np, a.patchDim)
	}
	hd := a.hd
	cos, sin := a.ropeTables(gridTHW)
	if len(cos) != np*hd || len(sin) != np*hd {
		return nil, fmt.Errorf("cuda: rope tables %d/%d for %d patches of head_dim %d", len(cos), len(sin), np, hd)
	}
	if a.planted.swapRope { // the row and column quarters swapped within each half of the cat(f, f) row
		cos, sin = append([]float32(nil), cos...), append([]float32(nil), sin...)
		half, q := hd/2, hd/4
		for i := range np {
			for base := 0; base < hd; base += half {
				for _, tb := range [][]float32{cos, sin} {
					row := tb[i*hd+base : i*hd+base+half]
					for j := range q {
						row[j], row[j+q] = row[j+q], row[j]
					}
				}
			}
		}
	}
	var pe []float32
	if a.posEmbeds != nil {
		g := gridTHW
		if a.planted.transposePos { // the position rows for the transposed grid
			g = make([][3]int, len(gridTHW))
			for i, t := range gridTHW {
				g[i] = [3]int{t[0], t[2], t[1]}
			}
		}
		if pe = a.posEmbeds(g); len(pe) != np*a.hidden {
			return nil, fmt.Errorf("cuda: %d position values for %d patches of %d", len(pe), np, a.hidden)
		}
	}
	if a.planted.oneSegment {
		segs = []int{0, np}
	}
	var out []float32
	err := a.ops.do(func() error {
		defer a.ops.releaseScratch()
		o := a.ops
		H, I, nH := a.hidden, a.inter, a.heads
		x, xn, pt := o.af(np*H), o.af(np*H), o.af(np*a.patchDim)
		q, k, v, ctx, tmp := o.af(np*H), o.af(np*H), o.af(np*H), o.af(np*H), o.af(np*H)
		f1, f2 := o.af(np*I), o.af(np*I)
		csB, snB := o.af(np*hd), o.af(np*hd)
		uploads := []struct {
			b Buffer
			v []float32
		}{{pt, pixels}, {csB, cos}, {snB, sin}}
		var posB Buffer
		if pe != nil {
			posB = o.af(np * H)
			uploads = append(uploads, struct {
				b Buffer
				v []float32
			}{posB, pe})
		}
		for _, u := range uploads {
			if e := gpu.Upload(u.b, u.v); e != nil {
				return e
			}
		}
		pb := a.patchB
		if a.planted.noPatchBias {
			pb = Buffer{}
		}
		a.linear(pt, &gridProj{w: a.patchW, b: pb, out: H, in: a.patchDim}, x, np)
		if pe != nil {
			o.addVec(x, posB, np*H)
		}
		scale := float32(1 / math.Sqrt(float64(hd)))
		if a.planted.noScale {
			scale = 1
		}
		for bi := range a.blocks {
			b := &a.blocks[bi]
			a.norm(x, b.norm1w, b.norm1b, xn, np)
			a.linear(xn, &b.q, q, np)
			a.linear(xn, &b.k, k, np)
			a.linear(xn, &b.v, v, np)
			if a.kind == gridGlmOcr && !a.planted.noQKNorm { // per-head q/k RMSNorm, before RoPE
				o.rms(q, b.qNorm, q, np*nH, hd, a.eps)
				o.rms(k, b.kNorm, k, np*nH, hd, a.eps)
			}
			o.ropeHalfTo(q, csB, snB, np, nH, hd)
			o.ropeHalfTo(k, csB, snB, np, nH, hd)
			for si := 1; si < len(segs); si++ {
				off, T := segs[si-1]*H*4, segs[si]-segs[si-1]
				if e := o.attention(q.At(off), k.At(off), v.At(off), ctx.At(off), T, nH, hd, scale); e != nil {
					return e
				}
			}
			a.linearAdd(ctx, &b.proj, x, tmp, np)
			a.norm(x, b.norm2w, b.norm2b, xn, np)
			if a.kind == gridGlmOcr {
				a.linear(xn, &b.gate, f1, np)
				a.linear(xn, &b.up, f2, np)
				o.cls(clsElem)
				o.launch(o.vit.SiLUMul, gpu.Grid1D(np*I, 256), Arg(f1), Arg(f2), i32(int32(np*I)))
				a.linearAdd(f1, &b.down, x, tmp, np)
			} else {
				a.linear(xn, &b.fc1, f1, np)
				o.geluTanh(f1, np*I)
				a.linearAdd(f1, &b.fc2, x, tmp, np)
			}
		}
		if e := o.finish(); e != nil {
			return e
		}
		out = make([]float32, np*H)
		return gpu.Download(x, out)
	})
	if err != nil {
		return nil, fmt.Errorf("cuda: vision tower: %w", err)
	}
	return out, nil
}

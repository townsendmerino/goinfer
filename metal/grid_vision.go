//go:build darwin

package metal

import (
	"fmt"
	"math"
	"sync"

	"github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The Qwen3.5+ and GLM-OCR vision towers on Metal (S2 of docs/tasks/task-multimodal-support-2026-10.md): aikit's
// ForwardViT up to the last block, in float32, from aikit's export (Weights, RopeTables, PositionEmbeds,
// VisionSegments). The patch embed and every block run here on the Gemma 4 tower's base (eg2Ops: f32 GEMM, RMSNorm,
// rotate-half RoPE, blocked attention) plus aikit gpu.ViT's bias add, LayerNorm, GELU-tanh and SiLU-mul. The tail
// (Qwen3: the merger; GLM-OCR: post-norm, downsample, merger) stays aikit's FinishHidden, on the host.
//
// Both towers' blocks: biased fused qkv (split into three GEMMs at upload), 2-D RoPE (rotate-half over the full head,
// aikit's own tables), attention scale 1/sqrt(head_dim) applied to q after RoPE, full attention within each image
// frame (one eg2Ops.attention per segment), biased proj and residual, then the MLP and residual. Qwen3: LayerNorm with
// bias, eps 1e-6, MLP fc1 -> GELU-tanh -> fc2, and the interpolated position rows added after the patch embed. GLM-OCR:
// RMSNorm, eps rms_norm_eps, per-head RMSNorm of q and k before RoPE, MLP down(silu(gate)·up), no position table.

func init() {
	multimodal.RegisterQwen3Tower("metal", func(enc *vision.Qwen3VisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newQwen3VAccel(enc)
	})
	multimodal.RegisterGlmOcrTower("metal", func(enc *vision.GlmOcrVisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newGlmOcrVAccel(enc)
	})
}

const gvMSL = `
// x[i] *= s: attention's 1/sqrt(head_dim), on q after RoPE.
kernel void gv_scale(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& s [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = x[gid] * s;
}
`

// gvKind is the tower family; it decides the norm, the MLP, the position term, RoPE and the attention segments.
type gvKind uint8

const (
	gvkQwen3  gvKind = iota // LayerNorm, GELU-tanh MLP, interpolated position rows, 2-D RoPE, one segment per frame
	gvkGlmOcr               // RMSNorm, per-head q/k RMSNorm, SiLU-gated MLP, 2-D RoPE, one segment per frame
	gvkQwen25               // RMSNorm, SiLU-gated MLP, no patch bias, 2-D RoPE, windows except the full-attention blocks
	gvkSiglip               // LayerNorm, GELU-tanh MLP, a fixed position table, no RoPE, one segment
)

func (k gvKind) rmsNorm() bool { return k == gvkGlmOcr || k == gvkQwen25 }
func (k gvKind) gated() bool   { return k == gvkGlmOcr || k == gvkQwen25 }
func (k gvKind) rope() bool    { return k != gvkSiglip }

type gvProj struct {
	w, b    Buffer // [out, in]; bias [out] (zero Buffer when none)
	out, in int
}

type gvBlock struct {
	norm1w, norm1b, norm2w, norm2b Buffer // LayerNorm (Qwen3, SigLIP) or RMSNorm weight only (GLM-OCR, Qwen2.5-VL: the b's are zero)
	q, k, v, proj                  gvProj
	qNorm, kNorm                   Buffer // GLM-OCR's per-head q/k RMSNorm; zero otherwise
	fc1, fc2                       gvProj // Qwen3, SigLIP
	gate, up, down                 gvProj // GLM-OCR, Qwen2.5-VL
	full                           bool   // Qwen2.5-VL: a full-attention block (per image frame, not per window)
}

// gridVAccel is a grid-based tower on Metal. One image batch at a time (mu): the scratch and the scalar arena are
// shared.
type gridVAccel struct {
	mu sync.Mutex
	eg2Ops
	scale          Pipeline
	kind           gvKind
	hidden, inter  int
	heads, hd      int
	patchDim       int
	lnEps          float32
	patchW, patchB Buffer
	blocks         []gvBlock
	posEmbeds      func([][3]int) []float32            // Qwen3's interpolated position rows, SigLIP's table; nil otherwise
	ropeTables     func([][3]int) (cos, sin []float32) // Qwen3, GLM-OCR (Qwen2.5-VL's come from its window plan)
	planted        gvDefect                            // test seam: G-S2c's planted defects
	prof           *gvProfile                          // test seam: S17 step 0's per-class GPU split; nil in production
	tap            func(bi, n int)                     // HiddenTaps: called after block bi\'s command buffer, with the residual\'s live length
	cap, segCap    int
	x, xn, pt, pos Buffer
	q, k, v, ctx   Buffer
	o, f1, f2      Buffer
	qh, kh, vh     Buffer
	sc, vt, ob     Buffer
	ropeCos        Buffer
	ropeSin        Buffer
}

// gvProfile is S17 step 0's instrument (docs/tasks/task-multimodal-support-2026-10.md): with it set, run ends its command
// buffer at every change of kernel class and adds each buffer's GPU-busy window to its class, and linear and the
// attention loop add their FLOPs. A test sets it; production never does, and run is then unchanged.
type gvProfile struct {
	cur   string
	gpu   map[string]float64 // seconds of GPU-busy time per class
	flops map[string]float64
	bufs  int
}

func (p *gvProfile) reset() {
	p.cur, p.gpu, p.flops, p.bufs = "", map[string]float64{}, map[string]float64{}, 0
}

// gvDefect is a planted defect (G-S2c); zero in production.
type gvDefect struct {
	noScale, swapRope, transposePos, noQKNorm, noPatchBias bool
	noWindows, noWindowOrder, noPosEmbed                   bool // G-S3a's (3), (4) and (5)
	tapShift                                               int  // G-S10e: DeepStack taps read this many blocks late
}

func (a *gridVAccel) Name() string { return "metal" }

func (a *gridVAccel) Close() error { return nil }

func newGridVAccel(kind gvKind, hidden, inter, heads, patchDim int, eps float64) (*gridVAccel, error) {
	if heads <= 0 || hidden%heads != 0 || (hidden/heads)%2 != 0 {
		return nil, fmt.Errorf("metal: vision tower hidden %d over %d heads: the kernels need an even head_dim dividing hidden", hidden, heads)
	}
	a := &gridVAccel{kind: kind, hidden: hidden, inter: inter, heads: heads, hd: hidden / heads, patchDim: patchDim}
	var err error
	a.eg2Ops, err = newEG2Ops(gvMSL, map[string]*Pipeline{"gv_scale": &a.scale})
	if err != nil {
		return nil, err
	}
	a.eps, a.lnEps = float32(eps), float32(eps)
	return a, nil
}

func (a *gridVAccel) up(v []float32) Buffer {
	if len(v) == 0 {
		return Buffer{}
	}
	return NewBufferFloats(a.d, v)
}

func (a *gridVAccel) proj(p vision.VisionProj) gvProj {
	return gvProj{w: a.up(p.W), b: a.up(p.B), out: p.Out, in: p.In}
}

// splitQKV uploads a fused [3·hidden, hidden] qkv projection (rows q, then k, then v) as three projections.
func (a *gridVAccel) splitQKV(p vision.VisionProj) (q, k, v gvProj, err error) {
	H := a.hidden
	if p.Out != 3*H || p.In != H || len(p.W) != 3*H*H {
		return q, k, v, fmt.Errorf("metal: qkv is %dx%d, want %dx%d", p.Out, p.In, 3*H, H)
	}
	part := func(i int) gvProj {
		r := gvProj{w: a.up(p.W[i*H*H : (i+1)*H*H]), out: H, in: H}
		if len(p.B) == 3*H {
			r.b = a.up(p.B[i*H : (i+1)*H])
		}
		return r
	}
	return part(0), part(1), part(2), nil
}

func newQwen3VAccel(enc *vision.Qwen3VisionEncoder) (*gridVAccel, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}
	c := w.Cfg
	a, err := newGridVAccel(gvkQwen3, c.HiddenSize, c.IntermediateSize, c.NumHeads,
		c.InChannels*c.TemporalPatchSize*c.PatchSize*c.PatchSize, w.LNEps)
	if err != nil {
		return nil, err
	}
	a.patchW, a.patchB = a.up(w.PatchW), a.up(w.PatchB)
	a.posEmbeds, a.ropeTables = enc.PositionEmbeds, enc.RopeTables
	for _, b := range w.Blocks {
		q, k, v, err := a.splitQKV(b.QKV)
		if err != nil {
			return nil, err
		}
		a.blocks = append(a.blocks, gvBlock{norm1w: a.up(b.Norm1W), norm1b: a.up(b.Norm1B), norm2w: a.up(b.Norm2W), norm2b: a.up(b.Norm2B),
			q: q, k: k, v: v, proj: a.proj(b.Proj), fc1: a.proj(b.FC1), fc2: a.proj(b.FC2)})
	}
	return a, nil
}

func newGlmOcrVAccel(enc *vision.GlmOcrVisionEncoder) (*gridVAccel, error) {
	w, err := enc.Weights()
	if err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}
	c := w.Cfg
	a, err := newGridVAccel(gvkGlmOcr, c.HiddenSize, c.IntermediateSize, c.NumHeads,
		c.InChannels*c.TemporalPatchSize*c.PatchSize*c.PatchSize, w.RMSEps)
	if err != nil {
		return nil, err
	}
	a.patchW, a.patchB = a.up(w.PatchW), a.up(w.PatchB)
	a.ropeTables = enc.RopeTables
	for _, b := range w.Blocks {
		q, k, v, err := a.splitQKV(b.QKV)
		if err != nil {
			return nil, err
		}
		if len(b.QNorm) != a.hd || len(b.KNorm) != a.hd {
			return nil, fmt.Errorf("metal: GLM-OCR q/k norm widths %d/%d, want head_dim %d", len(b.QNorm), len(b.KNorm), a.hd)
		}
		a.blocks = append(a.blocks, gvBlock{norm1w: a.up(b.Norm1W), norm2w: a.up(b.Norm2W), q: q, k: k, v: v, proj: a.proj(b.Proj),
			qNorm: a.up(b.QNorm), kNorm: a.up(b.KNorm), gate: a.proj(b.Gate), up: a.proj(b.Up), down: a.proj(b.Down)})
	}
	return a, nil
}

func (a *gridVAccel) grow(np, maxSeg int) {
	if np > a.cap {
		H, I := a.hidden, a.inter
		n := func(cols int) Buffer { return a.d.NewBufferLen(np * cols) }
		a.x, a.xn, a.pt, a.pos = n(H), n(H), n(a.patchDim), n(H)
		a.q, a.k, a.v, a.ctx, a.o = n(H), n(H), n(H), n(H), n(H)
		a.f1, a.f2 = n(I), n(I)
		a.ropeCos, a.ropeSin = n(a.hd/2), n(a.hd/2)
		a.cap = np
	}
	if maxSeg > a.segCap {
		H, B := a.hidden, min(eg2Block, maxSeg)
		a.qh, a.kh, a.vh = a.d.NewBufferLen(maxSeg*H), a.d.NewBufferLen(maxSeg*H), a.d.NewBufferLen(maxSeg*H)
		a.sc, a.vt, a.ob = a.d.NewBufferLen(B*maxSeg), a.d.NewBufferLen(a.hd*maxSeg), a.d.NewBufferLen(B*a.hd)
		a.segCap = maxSeg
	}
}

// linear runs out = x·Wᵀ (+ bias) over rows.
func (a *gridVAccel) linear(e *Encoder, x Buffer, p *gvProj, out Buffer, rows int) {
	if a.prof != nil {
		a.prof.flops["gemm"] += 2 * float64(rows) * float64(p.out) * float64(p.in)
	}
	a.gemm(e, x, p.w, out, rows, p.out, p.in)
	if p.b != (Buffer{}) {
		e.Dispatch(a.vit.AddBias, rows*p.out, 256, out, p.b, a.u32(uint32(rows)), a.u32(uint32(p.out)))
	}
}

// norm is the block's norm: LayerNorm with bias (Qwen3) or the weight-only RMSNorm (GLM-OCR).
func (a *gridVAccel) norm(e *Encoder, x, w, b, out Buffer, rows int) {
	if a.kind.rmsNorm() {
		a.rms(e, x, w, out, rows, a.hidden)
		return
	}
	e.Dispatch(a.vit.LayerNorm, rows*gpu.ViTBlock, gpu.ViTBlock, x, w, b, out, a.u32(uint32(rows)), a.u32(uint32(a.hidden)), a.f32(a.lnEps))
}

// Hidden runs the patch embed and every block (multimodal.GridTowerAccelerator): Qwen3.5+ and GLM-OCR, whose attention
// segments are the image frames.
func (a *gridVAccel) Hidden(pixels []float32, gridTHW [][3]int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.hiddenLocked(pixels, gridTHW)
}

// HiddenTaps is Hidden plus the outputs of the given blocks, in that order (multimodal.GridTowerTapper): Qwen3-VL's
// DeepStack taps (S10, G-S10e). Each tap is the residual right after its block, read where run ends that block's
// command buffer.
func (a *gridVAccel) HiddenTaps(pixels []float32, gridTHW [][3]int, blocks []int) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	taps := make([][]float32, len(blocks))
	a.tap = func(bi, n int) {
		for k, b := range blocks {
			if b+a.planted.tapShift == bi { // G-S10e's planted defect shifts the tap a block late
				taps[k] = append([]float32(nil), a.x.Floats()[:n]...)
			}
		}
	}
	defer func() { a.tap = nil }()
	h, err := a.hiddenLocked(pixels, gridTHW)
	if err != nil {
		return nil, nil, err
	}
	for k, t := range taps {
		if t == nil {
			return nil, nil, fmt.Errorf("metal: DeepStack tap %d (block %d) was not reached in %d blocks", k, blocks[k], len(a.blocks))
		}
	}
	return h, taps, nil
}

func (a *gridVAccel) hiddenLocked(pixels []float32, gridTHW [][3]int) ([]float32, error) {
	segs := vision.VisionSegments(gridTHW)
	np := segs[len(segs)-1]
	if np == 0 || len(pixels) != np*a.patchDim {
		return nil, fmt.Errorf("metal: %d pixel values for %d patches of %d", len(pixels), np, a.patchDim)
	}
	a.grow(np, maxSegment(segs))
	copy(a.pt.Floats()[:np*a.patchDim], pixels)
	cos, sin := a.ropeTables(gridTHW)
	if err := a.stageRope(cos, sin, np); err != nil {
		return nil, err
	}
	if a.posEmbeds != nil {
		g := gridTHW
		if a.planted.transposePos { // G-S2c (3): the position rows for the transposed grid
			g = make([][3]int, len(gridTHW))
			for i, t := range gridTHW {
				g[i] = [3]int{t[0], t[2], t[1]}
			}
		}
		if err := a.stagePos(a.posEmbeds(g), np); err != nil {
			return nil, err
		}
	}
	return a.run(np, func(int) []int { return segs })
}

// maxSegment is the longest segment of a cumulative-offset list.
func maxSegment(segs []int) int {
	m := 0
	for i := 1; i < len(segs); i++ {
		m = max(m, segs[i]-segs[i-1])
	}
	return m
}

// stageRope uploads aikit's cos/sin [np, head_dim] tables. eg2_rope takes [T, hd/2]: the first half of each row, which is
// the whole table (aikit's rows are cat(f, f)).
func (a *gridVAccel) stageRope(cos, sin []float32, np int) error {
	hd := a.hd
	if len(cos) != np*hd || len(sin) != np*hd {
		return fmt.Errorf("metal: rope tables %d/%d for %d patches of head_dim %d", len(cos), len(sin), np, hd)
	}
	hc, hs, half := a.ropeCos.Floats(), a.ropeSin.Floats(), hd/2
	for i := range np {
		if a.planted.swapRope { // G-S2c (2), G-S3a (2): the row and column halves swapped
			q := half / 2
			copy(hc[i*half:], cos[i*hd+q:i*hd+half])
			copy(hc[i*half+q:], cos[i*hd:i*hd+q])
			copy(hs[i*half:], sin[i*hd+q:i*hd+half])
			copy(hs[i*half+q:], sin[i*hd:i*hd+q])
			continue
		}
		copy(hc[i*half:(i+1)*half], cos[i*hd:i*hd+half])
		copy(hs[i*half:(i+1)*half], sin[i*hd:i*hd+half])
	}
	return nil
}

// stagePos uploads the position rows [np, hidden] that run adds after the patch embed.
func (a *gridVAccel) stagePos(pe []float32, np int) error {
	if len(pe) != np*a.hidden {
		return fmt.Errorf("metal: %d position values for %d patches of %d", len(pe), np, a.hidden)
	}
	copy(a.pos.Floats()[:np*a.hidden], pe)
	return nil
}

// segSlots bounds the scalar-arena slots one attention call over T rows takes (eg2Ops.attention: three head-major
// gathers, then per head and 256-row block a V transpose, two GEMMs, a softmax and a scatter).
func (a *gridVAccel) segSlots(T int) int {
	return 9 + a.heads*((T+eg2Block-1)/eg2Block)*20
}

// run is the patch embed over the rows staged in a.pt (plus the position rows in a.pos, when the tower has them, and the
// RoPE tables, when it uses them) and every block; segs(bi) is block bi's attention segments as cumulative row offsets.
// A command buffer per block, so the scalar arena is reused from block to block; a block whose segments would overflow
// it (Qwen2.5-VL's windows) is split across command buffers between segments. The caller holds a.mu.
func (a *gridVAccel) run(np int, segs func(bi int) []int) ([]float32, error) {
	H, I, nH, hd := a.hidden, a.inter, a.heads, a.hd
	a.slot = 0
	e := a.eg2Ops.q.Begin()
	account := func() { // S17 step 0: this buffer's GPU window to the class it ran
		if a.prof != nil {
			a.prof.gpu[a.prof.cur] += e.GPUEnd() - e.GPUStart()
			a.prof.bufs++
		}
	}
	flush := func() error {
		e.End()
		if err := e.Err(); err != nil {
			return err
		}
		account()
		a.slot = 0
		e = a.eg2Ops.q.Begin()
		return nil
	}
	var perr error
	mark := func(class string) { // a class boundary: a no-op unless profiling
		if a.prof == nil || class == a.prof.cur {
			return
		}
		if a.prof.cur != "" {
			if err := flush(); err != nil && perr == nil {
				perr = err
			}
		}
		a.prof.cur = class
	}
	mark("gemm")
	pb := a.patchB
	if a.planted.noPatchBias { // G-S2c (5), G-S3a (6)
		pb = Buffer{}
	}
	a.linear(e, a.pt, &gvProj{w: a.patchW, b: pb, out: H, in: a.patchDim}, a.x, np)
	if a.posEmbeds != nil && !a.planted.noPosEmbed { // G-S3a (5) drops it
		mark("elementwise")
		e.Dispatch(a.p.add, np*H, 256, a.x, a.pos, a.u32(uint32(np*H)))
	}
	nr := np * nH * hd / 2
	scale := float32(1 / math.Sqrt(float64(hd)))
	for bi := range a.blocks {
		b := &a.blocks[bi]
		mark("norm")
		a.norm(e, a.x, b.norm1w, b.norm1b, a.xn, np)
		mark("gemm")
		a.linear(e, a.xn, &b.q, a.q, np)
		a.linear(e, a.xn, &b.k, a.k, np)
		a.linear(e, a.xn, &b.v, a.v, np)
		if a.kind == gvkGlmOcr && !a.planted.noQKNorm { // per-head q/k RMSNorm, before RoPE
			mark("norm")
			a.rms(e, a.q, b.qNorm, a.q, np*nH, hd)
			a.rms(e, a.k, b.kNorm, a.k, np*nH, hd)
		}
		mark("elementwise")
		if a.kind.rope() {
			e.Dispatch(a.p.rope, nr, 64, a.q, a.ropeCos, a.ropeSin, a.u32(uint32(nH)), a.u32(uint32(hd)), a.u32(uint32(nr)))
			e.Dispatch(a.p.rope, nr, 64, a.k, a.ropeCos, a.ropeSin, a.u32(uint32(nH)), a.u32(uint32(hd)), a.u32(uint32(nr)))
		}
		if !a.planted.noScale { // G-S2c (1), G-S3a (1) drop it
			e.Dispatch(a.scale, np*H, 256, a.q, a.u32(uint32(np*H)), a.f32(scale))
		}
		sg := segs(bi)
		mark("attention")
		for si := 1; si < len(sg); si++ {
			off, T := sg[si-1]*H*4, sg[si]-sg[si-1]
			if a.prof != nil {
				a.prof.flops["attention"] += 4 * float64(T) * float64(T) * float64(hd) * float64(nH)
			}
			if a.slot+a.segSlots(T)+64 > eg2ArenaSlots {
				if err := flush(); err != nil {
					return nil, err
				}
			}
			a.attention(e, eg2AttnBufs{q: a.q.At(off), k: a.k.At(off), v: a.v.At(off), qh: a.qh, kh: a.kh, vh: a.vh,
				sc: a.sc, vt: a.vt, ob: a.ob, ctx: a.ctx.At(off)}, T, nH, hd, nH, 0)
		}
		mark("gemm")
		a.linear(e, a.ctx, &b.proj, a.o, np)
		mark("elementwise")
		e.Dispatch(a.p.add, np*H, 256, a.x, a.o, a.u32(uint32(np*H)))
		mark("norm")
		a.norm(e, a.x, b.norm2w, b.norm2b, a.xn, np)
		mark("gemm")
		if a.kind.gated() {
			a.linear(e, a.xn, &b.gate, a.f1, np)
			a.linear(e, a.xn, &b.up, a.f2, np)
			mark("elementwise")
			e.Dispatch(a.vit.SiLUMul, np*I, 256, a.f1, a.f2, a.u32(uint32(np*I)))
			mark("gemm")
			a.linear(e, a.f1, &b.down, a.o, np)
		} else {
			a.linear(e, a.xn, &b.fc1, a.f1, np)
			mark("elementwise")
			e.Dispatch(a.vit.GELUTanh, np*I, 256, a.f1, a.u32(uint32(np*I)))
			mark("gemm")
			a.linear(e, a.f1, &b.fc2, a.o, np)
		}
		mark("elementwise")
		e.Dispatch(a.p.add, np*H, 256, a.x, a.o, a.u32(uint32(np*H)))
		if bi+1 < len(a.blocks) {
			if err := flush(); err != nil {
				return nil, err
			}
			if a.tap != nil {
				a.tap(bi, np*H)
			}
		}
	}
	e.End()
	if err := e.Err(); err != nil {
		return nil, err
	}
	account()
	if perr != nil {
		return nil, perr
	}
	if a.tap != nil && len(a.blocks) > 0 {
		a.tap(len(a.blocks)-1, np*H)
	}
	a.slot = 0
	return append([]float32(nil), a.x.Floats()[:np*H]...), nil
}

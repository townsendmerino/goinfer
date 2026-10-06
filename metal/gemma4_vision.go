//go:build darwin

package metal

import (
	"fmt"
	"math"
	"sync"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// The Gemma 4 vision tower on Metal (Phase VM, docs/tasks/task-embeddinggemma2.md): aikit's Gemma4Encoder.Forward up
// to its pool, in float32, from aikit's export (Gemma4Encoder.Weights). The patch embed and the 16 layers run here,
// with the text encoder's GEMMs, RMSNorm and attention blocks (eg2Ops, embeddinggemma2.go: full attention, scale 1.0,
// one head per KV head), plus the position-table add, the ClippableLinear clamps and the axial 2-D RoPE below. The
// pool and the projection after them stay aikit's (FinishHidden), on the host. Registered as the "metal" tower
// accelerator, so an EmbeddingGemma 2 encoder on Metal runs its tower here too.

func init() {
	embeddinggemma2.RegisterVisionAccelerator("metal", func(enc *vision.Gemma4Encoder) (embeddinggemma2.VisionAccelerator, error) {
		return newG4VAccel(enc)
	})
}

const g4vMSL = `
// x[i, d] += X[pos[i].x, d] + Y[pos[i].y, d]: the patch embedder's two position tables.
kernel void g4v_pos_add(device float* x [[buffer(0)]], device const float* X [[buffer(1)]],
    device const float* Y [[buffer(2)]], device const uint* pos [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint i = gid / H, d = gid % H;
    x[gid] = x[gid] + (X[pos[2u * i] * H + d] + Y[pos[2u * i + 1u] * H + d]);
}

static inline float g4v_clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// A ClippableLinear's output clamp, in place.
kernel void g4v_clamp(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], constant float& lo [[buffer(2)]],
    constant float& hi [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = g4v_clampf(x[gid], lo, hi);
}

// Its input clamp, into a copy (q, k and v read the same normed input under different bounds).
kernel void g4v_clamp_copy(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant float& lo [[buffer(3)]], constant float& hi [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) out[gid] = g4v_clampf(x[gid], lo, hi);
}

// Axial 2-D RoPE in place over x [T, heads, hd] from aikit's tables cos/sin [T, hd]: each half of a head is rotated
// rotate-half within itself (pairs d, d + hd/4), the first half by the patch's x and the second by its y.
kernel void g4v_rope(device float* x [[buffer(0)]], device const float* cs [[buffer(1)]],
    device const float* sn [[buffer(2)]], constant uint& heads [[buffer(3)]], constant uint& hd [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return; // n = T * heads * hd/2
    uint half_ = hd / 2u, q = hd / 4u;
    uint p = gid % half_, th = gid / half_, t = th / heads;
    uint base = (p / q) * half_, d = p % q;
    device float* r = x + th * hd + base;
    device const float* c = cs + t * hd + base;
    device const float* s = sn + t * hd + base;
    float a = r[d], b = r[d + q];
    r[d] = a * c[d] - b * s[d];
    r[d + q] = b * c[d + q] + a * s[d + q];
}
`

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
	up := func(v []float32) Buffer { return NewBufferFloats(d, v) }
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

func (a *g4vAccel) Close() error { return nil }

func (a *g4vAccel) grow(np int) {
	if np <= a.cap {
		return
	}
	c := a.cfg
	H, I, hd := c.HiddenSize, c.IntermediateSize, c.HeadDim
	n := func(cols int) Buffer { return a.d.NewBufferLen(np * cols) }
	B := min(eg2Block, np)
	a.x, a.xn, a.cp, a.pt = n(H), n(H), n(max(H, I)), n(3*c.PatchSize*c.PatchSize)
	a.posBuf = a.d.NewBufferBytes(np * 8)
	a.q, a.k, a.v, a.ctx, a.o, a.f = n(H), n(H), n(H), n(H), n(H), n(H)
	a.g, a.u = n(I), n(I)
	a.qh, a.kh, a.vh = n(H), n(H), n(H)
	a.sc, a.vt, a.ob = a.d.NewBufferLen(B*np), a.d.NewBufferLen(hd*np), a.d.NewBufferLen(B*hd)
	a.ropeCos, a.ropeSin = n(hd), n(hd)
	a.cap = np
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
	H, I, nH, hd := c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads, c.HeadDim
	np, pd := len(pos), 3*c.PatchSize*c.PatchSize
	if np == 0 || len(patches) != np*pd {
		return nil, fmt.Errorf("metal: %d patch values for %d patches of %d", len(patches), np, pd)
	}
	a.grow(np)
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
	nr := np * nH * hd / 2
	if len(a.layers) == 0 {
		e.End()
		if err := e.Err(); err != nil {
			return nil, err
		}
	}
	for li := range a.layers {
		ly := &a.layers[li]
		a.rms(e, a.x, ly.inNorm, a.xn, np, H)
		a.proj(e, a.xn, &ly.q, a.q, np)
		a.proj(e, a.xn, &ly.k, a.k, np)
		a.proj(e, a.xn, &ly.v, a.v, np)
		a.rms(e, a.q, ly.qNorm, a.q, np*nH, hd)
		a.rms(e, a.k, ly.kNorm, a.k, np*nH, hd)
		a.rms(e, a.v, a.ones, a.v, np*nH, hd)
		e.Dispatch(a.gp.rope, nr, 64, a.q, a.ropeCos, a.ropeSin, a.u32(uint32(nH)), a.u32(uint32(hd)), a.u32(uint32(nr)))
		e.Dispatch(a.gp.rope, nr, 64, a.k, a.ropeCos, a.ropeSin, a.u32(uint32(nH)), a.u32(uint32(hd)), a.u32(uint32(nr)))
		a.attention(e, eg2AttnBufs{q: a.q, k: a.k, v: a.v, qh: a.qh, kh: a.kh, vh: a.vh, sc: a.sc, vt: a.vt, ob: a.ob, ctx: a.ctx},
			np, nH, hd, nH, 0)
		a.proj(e, a.ctx, &ly.o, a.o, np)
		a.rms(e, a.o, ly.postAttnNorm, a.o, np, H)
		e.Dispatch(a.p.add, np*H, 256, a.x, a.o, a.u32(uint32(np*H)))
		a.rms(e, a.x, ly.preFFNorm, a.xn, np, H)
		a.proj(e, a.xn, &ly.gate, a.g, np)
		a.proj(e, a.xn, &ly.up, a.u, np)
		e.Dispatch(a.p.geluMul, np*I, 256, a.g, a.u, a.u32(uint32(np*I)))
		a.proj(e, a.g, &ly.down, a.f, np)
		a.rms(e, a.f, ly.postFFNorm, a.f, np, H)
		e.Dispatch(a.p.add, np*H, 256, a.x, a.f, a.u32(uint32(np*H)))
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

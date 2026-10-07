//go:build darwin

package metal

import (
	"fmt"
	"math"
	"sync"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
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

const g4aMSL = `
#define G4A_MAXTG 1024
#define G4A_MAXW 64

kernel void g4a_silu(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) { float v = x[gid]; x[gid] = v / (1.0f + exp(-v)); }
}

// x += a * y
kernel void g4a_axpy(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
    constant float& a [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = x[gid] + a * y[gid];
}

// out[t, j] = g[t, j] * sigmoid(g[t, H + j]) over g [n, 2H].
kernel void g4a_glu(device const float* g [[buffer(0)]], device float* out [[buffer(1)]], constant uint& H [[buffer(2)]],
    constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint t = gid / H, j = gid % H;
    float a = g[t * 2u * H + j], b = g[t * 2u * H + H + j];
    out[gid] = a * (1.0f / (1.0f + exp(-b)));
}

// Causal depthwise conv over time: out[t, c] = sum_k w[c, k] * y[t - (K-1) + k, c], y before 0 being 0.
kernel void g4a_dwconv(device const float* y [[buffer(0)]], device const float* w [[buffer(1)]], device float* out [[buffer(2)]],
    constant uint& H [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint t = gid / H, c = gid % H;
    float s = 0.0f;
    for (uint k = 0; k < K; k++) {
        int ti = int(t) - int(K - 1u) + int(k);
        if (ti >= 0) s += w[c * K + k] * y[uint(ti) * H + c];
    }
    out[gid] = s;
}

// q[i] *= qs[i % hd]; k[i] *= ks.
kernel void g4a_scale_qk(device float* q [[buffer(0)]], device float* k [[buffer(1)]], device const float* qs [[buffer(2)]],
    constant float& ks [[buffer(3)]], constant uint& hd [[buffer(4)]], constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    q[gid] = q[gid] * qs[gid % hd];
    k[gid] = k[gid] * ks;
}

// Sliding-window attention: threadgroup (t, h), one thread per head dim d. Key j is seen iff 0 <= t - j < W; its logit
// is q·k_j + q·relk_{t-j}, softcapped by cap, then a softmax over the window and the weighted sum of v.
kernel void g4a_attn(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
    device const float* relk [[buffer(3)]], device float* ctx [[buffer(4)]], constant uint& H [[buffer(5)]],
    constant uint& nH [[buffer(6)]], constant uint& W [[buffer(7)]], constant float& cap [[buffer(8)]],
    uint d [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]], uint g [[threadgroup_position_in_grid]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[G4A_MAXTG / 32];
    threadgroup float lg[G4A_MAXW];
    uint t = g / nH, h = g % nH, hd = tgsz;
    uint off = h * hd + d;
    float qd = q[t * H + off];
    uint lo = t + 1u > W ? t + 1u - W : 0u;
    uint nsg = (tgsz + 31u) / 32u;
    for (uint j = lo; j <= t; j++) {
        float p = qd * k[j * H + off] + qd * relk[(t - j) * H + off];
        p = simd_sum(p);
        if (sl == 0u) part[sg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d == 0u) {
            float s = 0.0f;
            for (uint i = 0; i < nsg; i++) s += part[i];
            lg[j - lo] = cap * precise::tanh(s / cap);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint m = t - lo + 1u;
    float mx = -INFINITY;
    for (uint i = 0; i < m; i++) mx = max(mx, lg[i]);
    float z = 0.0f;
    for (uint i = 0; i < m; i++) z += exp(lg[i] - mx);
    float acc = 0.0f;
    for (uint j = lo; j <= t; j++) acc += (exp(lg[j - lo] - mx) / z) * v[j * H + off];
    ctx[t * H + off] = acc;
}
`

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
	up := func(v []float32) Buffer { return NewBufferFloats(d, v) }
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

func (a *g4aAccel) Close() error { return nil }

func (a *g4aAccel) grow(n int) {
	if n <= a.cap {
		return
	}
	H := a.cfg.HiddenSize
	b := func(cols int) Buffer { return a.d.NewBufferLen(n * cols) }
	a.x, a.xn, a.cp, a.u, a.f = b(H), b(H), b(4*H), b(4*H), b(H)
	a.q, a.k, a.v, a.ctx, a.o = b(H), b(H), b(H), b(H), b(H)
	a.g, a.gl, a.cv = b(2*H), b(H), b(H)
	a.cap = n
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

func (a *g4aAccel) ffw(e *Encoder, f *g4aFF, n int) {
	H := a.cfg.HiddenSize
	a.rms(e, a.x, f.pre, a.xn, n, H)
	a.proj(e, a.xn, &f.up, a.u, n)
	e.Dispatch(a.ap.silu, n*4*H, 256, a.u, a.u32(uint32(n*4*H)))
	a.proj(e, a.u, &f.down, a.f, n)
	a.rms(e, a.f, f.post, a.f, n, H)
	e.Dispatch(a.ap.axpy, n*H, 256, a.x, a.f, a.u32(uint32(n*H)), a.f32(float32(a.cfg.ResidualWeight)))
}

// Blocks runs every conformer block over h [n, hidden] (see embeddinggemma2.AudioAccelerator), a command buffer per
// block so the scalar arena is reused.
func (a *g4aAccel) Blocks(h []float32, n int) ([]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	c := a.cfg
	H, nH := c.HiddenSize, c.NumAttentionHeads
	hd := H / nH
	if n <= 0 || len(h) != n*H {
		return nil, fmt.Errorf("metal: %d values for %d rows of %d", len(h), n, H)
	}
	a.grow(n)
	copy(a.x.Floats()[:n*H], h)
	nH4 := n * H
	for li := range a.layers {
		ly := &a.layers[li]
		a.slot = 0
		e := a.eg2Ops.q.Begin()
		a.ffw(e, &ly.ff1, n)
		// attention
		a.rms(e, a.x, ly.normPreAttn, a.xn, n, H)
		a.proj(e, a.xn, &ly.q, a.q, n)
		a.proj(e, a.xn, &ly.k, a.k, n)
		a.proj(e, a.xn, &ly.v, a.v, n)
		e.Dispatch(a.ap.scaleQK, nH4, 256, a.q, a.k, ly.qs, a.f32(float32(math.Log(1+math.E)/math.Ln2)), a.u32(uint32(hd)), a.u32(uint32(nH4)))
		e.Dispatch(a.ap.attn, n*nH*hd, hd, a.q, a.k, a.v, ly.relk, a.ctx, a.u32(uint32(H)), a.u32(uint32(nH)),
			a.u32(uint32(c.Window())), a.f32(float32(c.LogitCap)))
		a.proj(e, a.ctx, &ly.post, a.o, n)
		a.rms(e, a.o, ly.normPostAttn, a.o, n, H)
		e.Dispatch(a.p.add, nH4, 256, a.x, a.o, a.u32(uint32(nH4)))
		// light conv
		a.rms(e, a.x, ly.convPre, a.xn, n, H)
		a.proj(e, a.xn, &ly.convStart, a.g, n)
		e.Dispatch(a.ap.glu, nH4, 256, a.g, a.gl, a.u32(uint32(H)), a.u32(uint32(nH4)))
		e.Dispatch(a.ap.dwconv, nH4, 256, a.gl, ly.convW, a.cv, a.u32(uint32(H)), a.u32(uint32(c.ConvKernelSize)), a.u32(uint32(nH4)))
		a.rms(e, a.cv, ly.convNorm, a.cv, n, H)
		e.Dispatch(a.ap.silu, nH4, 256, a.cv, a.u32(uint32(nH4)))
		a.proj(e, a.cv, &ly.convEnd, a.o, n)
		e.Dispatch(a.p.add, nH4, 256, a.x, a.o, a.u32(uint32(nH4)))
		a.ffw(e, &ly.ff2, n)
		a.rms(e, a.x, ly.out, a.x, n, H)
		e.End()
		if err := e.Err(); err != nil {
			return nil, err
		}
	}
	return append([]float32(nil), a.x.Floats()[:n*H]...), nil
}

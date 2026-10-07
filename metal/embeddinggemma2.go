//go:build darwin

package metal

import (
	"fmt"
	"math"
	"sync"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// EmbeddingGemma 2's text encoder on Metal (Phase M, docs/tasks/task-embeddinggemma2.md): the CPU forward of
// embeddinggemma2/model.go in float32 on the GPU, one command buffer per input. The GEMMs and the row RMSNorm are
// aikit's ViT kernels (an f32 simdgroup GEMM of A·Bᵀ with B as nn.Linear stores it, [N, K]); attention runs as those
// GEMMs over blocks of query rows (attention, below), and this file adds the windowed row softmax and the data
// movement around it, RoPE from the CPU's own tables, the GELU-tanh multiplies and the residual adds. The embedding
// gather and pooling stay on the host. Registered as the "metal" accelerator.

func init() {
	embeddinggemma2.RegisterAccelerator("metal", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2Accel(m)
	})
}

const eg2MSL = `
#include <metal_stdlib>
using namespace metal;

#define EG2_TG 128
#define EG2_MAXHD 512

static inline float eg2_gelu(float v) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * v * (1.0f + precise::tanh(c * (v + 0.044715f * v * v * v)));
}

// x[i] = gelu_tanh(x[i]) * u[i]
kernel void eg2_gelu_mul(device float* x [[buffer(0)]], device const float* u [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = eg2_gelu(x[gid]) * u[gid];
}

// The PLE block's gate: g[t*P + j] = gelu_tanh(g[t*P + j]) * ple[(t*L + li)*P + j].
kernel void eg2_gelu_mul_ple(device float* g [[buffer(0)]], device const float* ple [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& P [[buffer(3)]], constant uint& L [[buffer(4)]],
    constant uint& li [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint t = gid / P, j = gid % P;
    g[gid] = eg2_gelu(g[gid]) * ple[(t * L + li) * P + j];
}

kernel void eg2_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = x[gid] + y[gid];
}

// The layer's last step: x = (x + y) * s.
kernel void eg2_add_scale(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant float& s [[buffer(3)]], uint gid [[thread_position_in_grid]])
{
    if (gid < n) x[gid] = (x[gid] + y[gid]) * s;
}

// Rotate-half RoPE in place over x [T, heads, hd], from host tables cos/sin [T, hd/2].
kernel void eg2_rope(device float* x [[buffer(0)]], device const float* cs [[buffer(1)]],
    device const float* sn [[buffer(2)]], constant uint& heads [[buffer(3)]], constant uint& hd [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return; // n = T * heads * hd/2
    uint half_ = hd / 2u;
    uint i = gid % half_, th = gid / half_, t = th / heads;
    device float* r = x + th * hd;
    float x1 = r[i], x2 = r[i + half_];
    float c = cs[t * half_ + i], s = sn[t * half_ + i];
    r[i] = x1 * c - x2 * s;
    r[i + half_] = x2 * c + x1 * s;
}

// Attention as matmuls (the GEMMs are aikit's): these kernels only move data and run the softmax.
// Head-major copy: x [T, nh, hd] -> out [nh, T, hd].
kernel void eg2_headmajor(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& T [[buffer(2)]], constant uint& nh [[buffer(3)]], constant uint& hd [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint n = T * nh * hd;
    if (gid >= n) return;
    uint d = gid % hd, th = gid / hd, h = th % nh, t = th / nh;
    out[(h * T + t) * hd + d] = x[gid];
}

// One block's values transposed: vh [nkv, T, hd] head kvh, rows [klo, klo+kc) -> vt [hd, kc].
kernel void eg2_vt_block(device const float* vh [[buffer(0)]], device float* vt [[buffer(1)]],
    constant uint& T [[buffer(2)]], constant uint& hd [[buffer(3)]], constant uint& kvh [[buffer(4)]],
    constant uint& klo [[buffer(5)]], constant uint& kc [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= hd * kc) return;
    uint j = gid % kc, d = gid / kc;
    vt[d * kc + j] = vh[(kvh * T + klo + j) * hd + d];
}

// Row softmax over a block of scores [rows, kc] (scale 1.0): row r is query i0+r, column c is key klo+c. Keys
// outside |i - j| <= W (all keys when W == 0) get weight 0. One threadgroup of EG2_TG per row.
kernel void eg2_softmax_rows(device float* sc [[buffer(0)]], constant uint& kc [[buffer(1)]],
    constant uint& i0 [[buffer(2)]], constant uint& klo [[buffer(3)]], constant uint& W [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint r [[threadgroup_position_in_grid]])
{
    threadgroup float red[EG2_TG];
    device float* row = sc + r * kc;
    uint i = i0 + r;
    uint lo = 0u, hi = kc;
    if (W > 0u) {
        uint a = i > W ? i - W : 0u;
        lo = a > klo ? a - klo : 0u;
        hi = min(kc, i + W + 1u - klo);
    }
    float m = -INFINITY;
    for (uint c = lo + tid; c < hi; c += EG2_TG) m = max(m, row[c]);
    red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = EG2_TG / 2u; o > 0u; o >>= 1) {
        if (tid < o) red[tid] = max(red[tid], red[tid + o]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint c = lo + tid; c < hi; c += EG2_TG) {
        float e = exp(row[c] - m);
        row[c] = e;
        sum += e;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = EG2_TG / 2u; o > 0u; o >>= 1) {
        if (tid < o) red[tid] += red[tid + o];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / red[0];
    for (uint c = tid; c < kc; c += EG2_TG) row[c] = (c >= lo && c < hi) ? row[c] * inv : 0.0f;
}

// A block's output [rows, hd] into ctx [T, nH*hd] at head h, rows from i0.
kernel void eg2_scatter_head(device const float* ob [[buffer(0)]], device float* ctx [[buffer(1)]],
    constant uint& rows [[buffer(2)]], constant uint& hd [[buffer(3)]], constant uint& nH [[buffer(4)]],
    constant uint& h [[buffer(5)]], constant uint& i0 [[buffer(6)]], uint gid [[thread_position_in_grid]])
{
    if (gid >= rows * hd) return;
    uint d = gid % hd, r = gid / hd;
    ctx[((i0 + r) * nH + h) * hd + d] = ob[gid];
}
`

const eg2TG = 128

type eg2Pipes struct{ geluMul, geluMulPLE, add, addScale, rope, headMajor, vtBlock, softmax, scatter Pipeline }

// eg2Block is how many query rows one attention block takes. A var only so a test can make blocks smaller than the
// tiny fixture's inputs and the sliding window.
var eg2Block = 256

type eg2Layer struct {
	w                                  embeddinggemma2.LayerWeights
	inNorm, q, k, v, o, qNorm, kNorm   Buffer
	postAttnNorm, preFFNorm, gate, up  Buffer
	down, postFFNorm, pleGate, pleProj Buffer
	plePostNorm                        Buffer
}

// eg2Ops is what the text encoder and the vision tower (gemma4_vision.go) share: the device, a queue, aikit's ViT
// kernels, this file's pipelines and the scalar arena.
type eg2Ops struct {
	d     *Device
	q     Queue
	vit   gpu.ViT
	p     eg2Pipes
	arena Buffer // one scalar per 256-byte slot, so no two dispatches in a command buffer share one
	slot  int
	eps   float32
}

// eg2AttnBufs are attention's inputs, scratch and output: q, k, v [T, heads, hd]; their head-major copies; a block's
// scores and transposed values and output; ctx [T, nH*hd].
type eg2AttnBufs struct{ q, k, v, qh, kh, vh, sc, vt, ob, ctx Buffer }

// newEG2Ops opens the device and compiles this file's kernels with extra (more kernels in the same precise library),
// whose pipelines it builds into more, by kernel name.
func newEG2Ops(extra string, more map[string]*Pipeline) (eg2Ops, error) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		return eg2Ops{}, fmt.Errorf("metal: %w", err)
	}
	vit, err := d.NewViT()
	if err != nil {
		return eg2Ops{}, err
	}
	lib, err := d.CompileLibraryPrecise(eg2MSL+extra, MSL3_1)
	if err != nil {
		return eg2Ops{}, fmt.Errorf("metal: compile the EmbeddingGemma 2 kernels: %w", err)
	}
	o := eg2Ops{d: d, q: d.NewCommandQueue(), vit: vit}
	for _, b := range []struct {
		name string
		dst  *Pipeline
	}{{"eg2_gelu_mul", &o.p.geluMul}, {"eg2_gelu_mul_ple", &o.p.geluMulPLE}, {"eg2_add", &o.p.add},
		{"eg2_add_scale", &o.p.addScale}, {"eg2_rope", &o.p.rope}, {"eg2_headmajor", &o.p.headMajor},
		{"eg2_vt_block", &o.p.vtBlock}, {"eg2_softmax_rows", &o.p.softmax}, {"eg2_scatter_head", &o.p.scatter}} {
		if *b.dst, err = d.NewComputePipeline(lib, b.name); err != nil {
			return eg2Ops{}, err
		}
	}
	for name, dst := range more {
		if *dst, err = d.NewComputePipeline(lib, name); err != nil {
			return eg2Ops{}, err
		}
	}
	o.arena = d.NewBufferBytes(eg2ArenaSlots * 256)
	return o, nil
}

// eg2Accel is the Metal accelerator. One forward at a time (mu): its scratch and scalar arena are shared.
type eg2Accel struct {
	mu sync.Mutex
	eg2Ops
	cfg       embeddinggemma2.Config
	embed     []float32
	pleProj   Buffer // scaled by hidden^-0.5 at upload, as the CPU scales the product
	pleNorm   Buffer
	finalNorm Buffer
	proj      Buffer
	ones      Buffer // the unweighted v norm
	layers    []eg2Layer
	ropes     map[eg2RopeKey][2]Buffer // immutable cos/sin tables, by input length, head dim and theta
	cap       int                      // scratch rows allocated
	scr       eg2Scratch
}

type eg2Scratch struct{ x, xn, q, k, v, ctx, o, g, u, f, gg, pr, ple, out, qh, kh, vh, sc, vt, ob Buffer }

type eg2RopeKey struct {
	T, hd int
	theta float64
}

const eg2ArenaSlots = 1 << 14

func newEG2Accel(m *embeddinggemma2.Model) (*eg2Accel, error) {
	ops, err := newEG2Ops("", nil)
	if err != nil {
		return nil, err
	}
	d := ops.d
	a := &eg2Accel{eg2Ops: ops, ropes: map[eg2RopeKey][2]Buffer{}}
	w := m.Weights()
	a.cfg, a.embed = w.Config, w.Embed
	a.eps = float32(a.cfg.RMSNormEps)
	for _, ly := range w.Layers {
		if ly.HeadDim > 512 {
			return nil, fmt.Errorf("metal: EmbeddingGemma 2 head dim %d, over the attention kernel's 512", ly.HeadDim)
		}
	}
	up := func(v []float32) Buffer { return NewBufferFloats(d, v) }
	ps := float32(1 / math.Sqrt(float64(a.cfg.Hidden)))
	scaled := make([]float32, len(w.PLEProj))
	for i, x := range w.PLEProj {
		scaled[i] = x * ps
	}
	a.pleProj, a.pleNorm, a.finalNorm, a.proj = up(scaled), up(w.PLENorm), up(w.FinalNorm), up(w.Projection)
	ones := make([]float32, 512)
	for i := range ones {
		ones[i] = 1
	}
	a.ones = up(ones)
	for _, ly := range w.Layers {
		a.layers = append(a.layers, eg2Layer{w: ly, inNorm: up(ly.InNorm), q: up(ly.Q), k: up(ly.K), v: up(ly.V), o: up(ly.O),
			qNorm: up(ly.QNorm), kNorm: up(ly.KNorm), postAttnNorm: up(ly.PostAttnNorm), preFFNorm: up(ly.PreFFNorm),
			gate: up(ly.Gate), up: up(ly.Up), down: up(ly.Down), postFFNorm: up(ly.PostFFNorm), pleGate: up(ly.PLEGate),
			pleProj: up(ly.PLEProj), plePostNorm: up(ly.PLEPostNorm)})
	}
	return a, nil
}

func (a *eg2Accel) Name() string { return "metal" }

func (a *eg2Accel) Close() error { return nil }

// u32 and f32 write one scalar into its own arena slot and return the slot to bind.
func (a *eg2Ops) u32(v uint32) Buffer {
	if a.slot >= eg2ArenaSlots {
		panic("metal: EmbeddingGemma 2 scalar arena exhausted")
	}
	a.arena.U32s()[a.slot*64] = v
	b := a.arena.At(a.slot * 256)
	a.slot++
	return b
}
func (a *eg2Ops) f32(v float32) Buffer { return a.u32(math.Float32bits(v)) }

func (a *eg2Accel) grow(T int) {
	if T <= a.cap {
		return
	}
	c := a.cfg
	maxQD, maxKVD, maxHD := 0, 0, 0
	for _, ly := range a.layers {
		maxQD = max(maxQD, c.Heads*ly.w.HeadDim)
		maxKVD = max(maxKVD, ly.w.KVHeads*ly.w.HeadDim)
		maxHD = max(maxHD, ly.w.HeadDim)
	}
	n := func(cols int) Buffer { return a.d.NewBufferLen(T * cols) }
	a.scr = eg2Scratch{x: n(c.Hidden), xn: n(c.Hidden), q: n(maxQD), k: n(maxKVD), v: n(maxKVD), ctx: n(maxQD),
		o: n(c.Hidden), g: n(c.Intermediate), u: n(c.Intermediate), f: n(c.Hidden), gg: n(c.PLEDim), pr: n(c.Hidden),
		ple: n(c.Layers * c.PLEDim), out: n(c.EmbeddingDim), qh: n(maxQD), kh: n(maxKVD), vh: n(maxKVD),
		sc: a.d.NewBufferLen(min(eg2Block, T) * T), vt: a.d.NewBufferLen(maxHD * T), ob: a.d.NewBufferLen(min(eg2Block, T) * maxHD)}
	a.cap = T
}

func (a *eg2Ops) gemm(e *Encoder, A, B, C Buffer, M, N, K int) {
	p, gx, gy, tgx, tgy := a.vit.GEMMF32Plan(M, N, K)
	e.Dispatch2D(p, gx, gy, tgx, tgy, A, B, C, a.u32(uint32(M)), a.u32(uint32(N)), a.u32(uint32(K)))
}

func (a *eg2Ops) rms(e *Encoder, x, w, out Buffer, rows, dim int) {
	e.Dispatch(a.vit.RMSNorm, rows*256, 256, x, w, out, a.u32(uint32(rows)), a.u32(uint32(dim)), a.f32(a.eps))
}

// Forward runs the encoder over ids (see embeddinggemma2.Accelerator): the embedding gather on the host, then
// ForwardEmbeds.
func (a *eg2Accel) Forward(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	c := a.cfg
	H := c.Hidden
	if len(ids) == 0 {
		return nil, nil, fmt.Errorf("metal: no input ids")
	}
	x := make([]float32, len(ids)*H)
	scale := float32(math.Sqrt(float64(H)))
	for t, id := range ids {
		if id < 0 || id >= c.VocabSize {
			return nil, nil, fmt.Errorf("metal: token id %d out of range [0, %d)", id, c.VocabSize)
		}
		for j, v := range a.embed[id*H : (id+1)*H] {
			x[t*H+j] = v * scale
		}
	}
	return a.ForwardEmbeds(x, len(ids), keepLayers)
}

// ForwardEmbeds runs the encoder from T prepared input rows (see embeddinggemma2.Accelerator).
func (a *eg2Accel) ForwardEmbeds(x0 []float32, T int, keepLayers bool) ([]float32, [][]float32, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	c := a.cfg
	H, I, P, L, E := c.Hidden, c.Intermediate, c.PLEDim, c.Layers, c.EmbeddingDim
	if T == 0 || len(x0) != T*H {
		return nil, nil, fmt.Errorf("metal: %d input values for %d rows of %d", len(x0), T, H)
	}
	a.grow(T)
	s := &a.scr
	x := s.x.Floats()[:T*H]
	copy(x, x0)
	var layers [][]float32
	if keepLayers {
		layers = append(layers, append([]float32(nil), x...))
	}
	a.slot = 0
	e := a.q.Begin()
	a.gemm(e, s.x, a.pleProj, s.ple, T, L*P, H)
	a.rms(e, s.ple, a.pleNorm, s.ple, T*L, P)
	for li := range a.layers {
		ly := &a.layers[li]
		hd, nKV := ly.w.HeadDim, ly.w.KVHeads
		qd, kvd := c.Heads*hd, nKV*hd
		// attention
		a.rms(e, s.x, ly.inNorm, s.xn, T, H)
		a.gemm(e, s.xn, ly.q, s.q, T, qd, H)
		a.gemm(e, s.xn, ly.k, s.k, T, kvd, H)
		a.gemm(e, s.xn, ly.v, s.v, T, kvd, H)
		a.rms(e, s.q, ly.qNorm, s.q, T*c.Heads, hd)
		a.rms(e, s.k, ly.kNorm, s.k, T*nKV, hd)
		a.rms(e, s.v, a.ones, s.v, T*nKV, hd)
		cb, sb := a.ropeTables(T, hd, ly.w.RopeTheta)
		e.Dispatch(a.p.rope, T*c.Heads*hd/2, 64, s.q, cb, sb, a.u32(uint32(c.Heads)), a.u32(uint32(hd)), a.u32(uint32(T*c.Heads*hd/2)))
		e.Dispatch(a.p.rope, T*nKV*hd/2, 64, s.k, cb, sb, a.u32(uint32(nKV)), a.u32(uint32(hd)), a.u32(uint32(T*nKV*hd/2)))
		win := uint32(c.SlidingWindow)
		if ly.w.Full {
			win = 0
		}
		a.attention(e, eg2AttnBufs{q: s.q, k: s.k, v: s.v, qh: s.qh, kh: s.kh, vh: s.vh, sc: s.sc, vt: s.vt, ob: s.ob, ctx: s.ctx},
			T, c.Heads, hd, nKV, win)
		a.gemm(e, s.ctx, ly.o, s.o, T, H, qd)
		a.rms(e, s.o, ly.postAttnNorm, s.o, T, H)
		e.Dispatch(a.p.add, T*H, 256, s.x, s.o, a.u32(uint32(T*H)))
		// MLP
		a.rms(e, s.x, ly.preFFNorm, s.xn, T, H)
		a.gemm(e, s.xn, ly.gate, s.g, T, I, H)
		a.gemm(e, s.xn, ly.up, s.u, T, I, H)
		e.Dispatch(a.p.geluMul, T*I, 256, s.g, s.u, a.u32(uint32(T*I)))
		a.gemm(e, s.g, ly.down, s.f, T, H, I)
		a.rms(e, s.f, ly.postFFNorm, s.f, T, H)
		e.Dispatch(a.p.add, T*H, 256, s.x, s.f, a.u32(uint32(T*H)))
		// PLE block, then the layer scalar
		a.gemm(e, s.x, ly.pleGate, s.gg, T, P, H)
		e.Dispatch(a.p.geluMulPLE, T*P, 256, s.gg, s.ple, a.u32(uint32(T*P)), a.u32(uint32(P)), a.u32(uint32(L)), a.u32(uint32(li)))
		a.gemm(e, s.gg, ly.pleProj, s.pr, T, H, P)
		a.rms(e, s.pr, ly.plePostNorm, s.pr, T, H)
		e.Dispatch(a.p.addScale, T*H, 256, s.x, s.pr, a.u32(uint32(T*H)), a.f32(ly.w.LayerScalar))
		if keepLayers { // a command buffer per layer, so the host can read the layer's output
			e.End()
			if err := e.Err(); err != nil {
				return nil, nil, err
			}
			layers = append(layers, append([]float32(nil), s.x.Floats()[:T*H]...))
			a.slot = 0
			e = a.q.Begin()
		}
	}
	a.rms(e, s.x, a.finalNorm, s.xn, T, H)
	a.gemm(e, s.xn, a.proj, s.out, T, E, H)
	e.End()
	if err := e.Err(); err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), s.out.Floats()[:T*E]...), layers, nil
}

// attention is one layer's bidirectional grouped-query attention over b.q/b.k/b.v into b.ctx, as matmuls: q, k and v
// laid out head-major, then for each block of eg2Block query rows the scores Q·Kᵀ over only the keys the block can
// reach (the window plus the block on a sliding layer), a windowed row softmax, and the output scores × V against the
// block's values transposed (the GEMM is A·Bᵀ, so V must be [hd, keys]). Memory is a block's scores, never T x T.
func (a *eg2Ops) attention(e *Encoder, s eg2AttnBufs, T, nH, hd, nKV int, win uint32) {
	e.Dispatch(a.p.headMajor, T*nH*hd, 256, s.q, s.qh, a.u32(uint32(T)), a.u32(uint32(nH)), a.u32(uint32(hd)))
	e.Dispatch(a.p.headMajor, T*nKV*hd, 256, s.k, s.kh, a.u32(uint32(T)), a.u32(uint32(nKV)), a.u32(uint32(hd)))
	e.Dispatch(a.p.headMajor, T*nKV*hd, 256, s.v, s.vh, a.u32(uint32(T)), a.u32(uint32(nKV)), a.u32(uint32(hd)))
	B, W := min(eg2Block, T), int(win)
	group := nH / nKV
	for g := range nKV {
		for i0 := 0; i0 < T; i0 += B {
			i1 := min(T, i0+B)
			rows := i1 - i0
			klo, khi := 0, T
			if W > 0 {
				klo, khi = max(0, i0-W), min(T, i1+W)
			}
			kc := khi - klo
			e.Dispatch(a.p.vtBlock, hd*kc, 256, s.vh, s.vt, a.u32(uint32(T)), a.u32(uint32(hd)), a.u32(uint32(g)),
				a.u32(uint32(klo)), a.u32(uint32(kc)))
			for h := g * group; h < (g+1)*group; h++ {
				a.gemm(e, s.qh.At(((h*T)+i0)*hd*4), s.kh.At(((g*T)+klo)*hd*4), s.sc, rows, kc, hd)
				e.Dispatch(a.p.softmax, rows*eg2TG, eg2TG, s.sc, a.u32(uint32(kc)), a.u32(uint32(i0)), a.u32(uint32(klo)), a.u32(win))
				a.gemm(e, s.sc, s.vt, s.ob, rows, hd, kc)
				e.Dispatch(a.p.scatter, rows*hd, 256, s.ob, s.ctx, a.u32(uint32(rows)), a.u32(uint32(hd)), a.u32(uint32(nH)),
					a.u32(uint32(h)), a.u32(uint32(i0)))
			}
		}
	}
}

// ropeTables returns the CPU's own cos/sin tables for T positions at this head dim and theta, uploaded once and never
// written again, so every layer and every in-flight command buffer can read the same buffers. At most 16 lengths are
// kept; the 17th clears the cache (a forward holds the lock, so no command buffer is using a dropped table).
func (a *eg2Accel) ropeTables(T, hd int, theta float64) (Buffer, Buffer) {
	k := eg2RopeKey{T, hd, theta}
	if b, ok := a.ropes[k]; ok {
		return b[0], b[1]
	}
	if len(a.ropes) >= 16 {
		clear(a.ropes)
	}
	cs, sn := embeddinggemma2.RopeTables(T, hd, theta)
	b := [2]Buffer{NewBufferFloats(a.d, cs), NewBufferFloats(a.d, sn)}
	a.ropes[k] = b
	return b[0], b[1]
}

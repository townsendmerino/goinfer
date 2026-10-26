//go:build darwin

package metal

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
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
	w32   Pipeline // tower_gemm_w32: gemm's kernel (KS-D4)
	// vitGemmForTest runs gemm on aikit's GEMMF32Plan kernel, which tower_gemm_w32 matches byte for byte: the old arm of
	// KS-D4's A/B.
	vitGemmForTest bool
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
	lib, err := d.CompileLibraryPrecise(eg2MSL+towerGemmMSL+extra, MSL3_1)
	if err != nil {
		return eg2Ops{}, fmt.Errorf("metal: compile the EmbeddingGemma 2 kernels: %w", err)
	}
	o := eg2Ops{d: d, q: d.NewCommandQueue(), vit: vit}
	for _, b := range []struct {
		name string
		dst  *Pipeline
	}{{"eg2_gelu_mul", &o.p.geluMul}, {"eg2_gelu_mul_ple", &o.p.geluMulPLE}, {"eg2_add", &o.p.add},
		{"eg2_add_scale", &o.p.addScale}, {"eg2_rope", &o.p.rope}, {"eg2_headmajor", &o.p.headMajor},
		{"eg2_vt_block", &o.p.vtBlock}, {"eg2_softmax_rows", &o.p.softmax}, {"eg2_scatter_head", &o.p.scatter},
		{"tower_gemm_w32", &o.w32}} {
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

// errTowerClosed is what a forward returns after Close.
var errTowerClosed = errors.New("metal: the accelerator is closed")

// open reports errTowerClosed once closeDevice has run. The caller holds the accelerator's lock.
func (o *eg2Ops) open() error {
	if o.d == nil {
		return errTowerClosed
	}
	return nil
}

// release frees each buffer from the device ledger and zeroes it. The caller holds the accelerator's lock, and every
// forward waits for its command buffers, so no GPU work reads them.
func (o *eg2Ops) release(bufs ...*Buffer) {
	for _, b := range bufs {
		o.d.ReleaseBuf(*b)
		*b = Buffer{}
	}
}

// closeDevice releases every buffer and object on the device ledger (weights, scratch, pipelines, the queue) and the
// device handle. Idempotent; the caller holds the accelerator's lock.
func (o *eg2Ops) closeDevice() {
	if o.d == nil {
		return
	}
	o.d.ReleaseAll()
	o.d.ReleaseObjects()
	o.d = nil
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

// Close frees the encoder's device memory; a forward after it is an error.
func (a *eg2Accel) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.closeDevice()
	a.layers, a.ropes, a.cap, a.scr = nil, nil, 0, eg2Scratch{}
	return nil
}

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

// grow sizes the scratch for T rows, releasing what it replaces.
func (a *eg2Accel) grow(T int) error {
	if err := a.open(); err != nil {
		return err
	}
	if T <= a.cap {
		return nil
	}
	s := &a.scr
	a.release(&s.x, &s.xn, &s.q, &s.k, &s.v, &s.ctx, &s.o, &s.g, &s.u, &s.f, &s.gg, &s.pr, &s.ple, &s.out, &s.qh, &s.kh,
		&s.vh, &s.sc, &s.vt, &s.ob)
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
	return nil
}

// gemm is C[M,N] = A[M,K]·Bᵀ in f32 through tower_gemm_w32 (its bias bound but unread).
func (a *eg2Ops) gemm(e *Encoder, A, B, C Buffer, M, N, K int) {
	if a.vitGemmForTest {
		p, gx, gy, tgx, tgy := a.vit.GEMMF32Plan(M, N, K)
		e.Dispatch2D(p, gx, gy, tgx, tgy, A, B, C, a.u32(uint32(M)), a.u32(uint32(N)), a.u32(uint32(K)))
		return
	}
	e.Dispatch2D(a.w32, (N+63)/64, (M+63)/64, 512, 1, A, B, C, B, a.u32(uint32(M)), a.u32(uint32(N)), a.u32(uint32(K)),
		a.u32(0), a.u32(0))
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
	if err := a.grow(T); err != nil {
		return nil, nil, err
	}
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
// kept; the 17th clears the cache and releases its tables (a forward holds the lock, so no command buffer is using a
// dropped table).
func (a *eg2Accel) ropeTables(T, hd int, theta float64) (Buffer, Buffer) {
	k := eg2RopeKey{T, hd, theta}
	if b, ok := a.ropes[k]; ok {
		return b[0], b[1]
	}
	if len(a.ropes) >= 16 {
		for k, b := range a.ropes {
			a.release(&b[0], &b[1])
			delete(a.ropes, k)
		}
	}
	cs, sn := embeddinggemma2.RopeTables(T, hd, theta)
	b := [2]Buffer{NewBufferFloats(a.d, cs), NewBufferFloats(a.d, sn)}
	a.ropes[k] = b
	return b[0], b[1]
}

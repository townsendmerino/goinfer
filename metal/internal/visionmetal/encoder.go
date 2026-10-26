//go:build darwin

// Package visionmetal is the native-Metal vision.ResidentEncoder — a whole SigLIP
// tower running on Apple GPUs, cgo-free (docs/task-native-gpu.md, Phase 3). It is the
// Apple counterpart of gpu/visioncuda, and the vision analogue of what gpu/annmetal is
// for ANN: aikit's Metal device substrate plus one real consumer.
//
// Importing this package is the opt-in. aikit's core `vision` never imports `gpu`, so
// the default build stays pure-Go CPU; this package's init() plugs the factory into
// vision.RegisterResident, and Encoder.EnableResident then routes Forward here.
//
// The forward mirrors gpu/visioncuda/encoder.go op for op (same order, same
// formulations) — the ViT kernels (gpu/metal_vit.go) carry the arithmetic-matching
// detail. The device-level differences from the CUDA path are all here: Apple UMA
// means a buffer's Floats() is a zero-copy live view (no explicit upload/download);
// scalars bind as 1-element buffers (SetU32 / a written float view); and the whole
// forward runs under one runtime.LockOSThread, because each dispatch's Run1D allocates
// and drains an NSAutoreleasePool that objc requires on a single thread.
package visionmetal

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/townsendmerino/goinfer/internal/vision"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

func init() {
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		w, err := e.GPUWeights()
		if err != nil {
			return nil, err
		}
		return newEncoder(w)
	})
}

// mat is one resident W8A8 weight: int8 rows [Rows,Cols] plus per-row scales.
type mat struct {
	q, s       gpu.Buffer
	rows, cols int
}

// layer is one transformer block's resident weights.
type layer struct {
	ln1w, ln1b, ln2w, ln2b gpu.Buffer
	qb, kb, vb, ob         gpu.Buffer
	fc1b, fc2b             gpu.Buffer
	qw, kw, vw, ow         mat
	fc1w, fc2w             mat
}

// encoder is the device-resident SigLIP tower.
type encoder struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	w   vision.GPUWeights

	patchW, patchB, posEmb gpu.Buffer
	postLNw, postLNb       gpu.Buffer
	layers                 []layer

	// Per-forward scratch, allocated once and reused across layers/forwards.
	patches, h, n1, n2 gpu.Buffer
	qa, ka, va, att, o gpu.Buffer
	mid, mlp, out      gpu.Buffer
	qi8, qs            gpu.Buffer // quantized activation + per-row scale

	// PER-DISPATCH scalar slots (audit M-13). Metal has no by-value kernel arg, so
	// every int argument is a one-word buffer. This used to be three shared slots
	// rewritten before each dispatch, which was safe only because each Run1D
	// committed AND WAITED — the scalar was fully consumed before the next write.
	// Batching a layer into one command buffer removes that guarantee: every
	// dispatch in the buffer would see the LAST value written. So each dispatch
	// now takes its own slot from this ring, reset once per command buffer.
	iv     []gpu.Buffer // uint32 slots
	ivNext int
	fv     []gpu.Buffer // float32 slots (eps, attention scale)
	fvNext int

	cpp int
	mu  sync.Mutex
}

// newEncoder uploads the tower and allocates scratch. A device OOM surfaces from the
// gpu layer as a panic (MustBuf's loud-failure contract); recover it into an error so
// EnableResident declines and the caller keeps the CPU path.
func newEncoder(w vision.GPUWeights) (enc *encoder, err error) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	defer func() {
		if r := recover(); r != nil {
			dev.ReleaseAll()
			dev.ReleaseObjects()
			enc, err = nil, fmt.Errorf("visionmetal: device upload failed: %v", r)
		}
	}()
	kern, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	np, hidden, inter := w.NumPatches, w.Hidden, w.Inter
	if np <= 0 || hidden <= 0 || inter <= 0 {
		dev.ReleaseObjects()
		return nil, fmt.Errorf("visionmetal: degenerate tower (np=%d hidden=%d inter=%d)", np, hidden, inter)
	}
	// Threadgroup-memory budget check (audit C-06, mirroring qwenmetal's own C-02 fix —
	// qwenmetal/encoder.go's attnThreadgroupBytes guards the SAME shared kernel, gpu.ViT's
	// `attention`). The kernel stages a per-query score row in DYNAMIC threadgroup memory
	// (np*4 bytes — one float per patch, metal_vit.go's `sc [[threadgroup(0)]]`) on top of its
	// two STATIC threadgroup arrays (smax/ssum, ViTBlock floats each). A dispatch whose total
	// exceeds the device maximum (~32 KiB on Apple GPUs) aborts the command buffer, and
	// waitUntilCompleted returns cleanly from an abort: e.att simply keeps the previous layer's
	// contents and Forward returns a plausible, wrong hidden state — metal.go's own doc records
	// this as "silently tolerates ... over-budget threadgroup memory ... status Completed".
	// Unlike qwenmetal's per-image patch count (checked per ForwardViT call), this tower's np is
	// FIXED at build time (w.NumPatches), so the check belongs here, once, rather than per
	// forward. Not a shipped shape today (so400m/896 = 4,096 patches, well under the limit), but
	// a wider image resolution would hit it silently without this.
	if need, have := attnThreadgroupBytes(np), dev.MaxThreadgroupMemoryLength(); need > have {
		dev.ReleaseObjects()
		return nil, fmt.Errorf("visionmetal: ViT attention needs %d B of threadgroup memory for %d patches but the device allows %d — this tower's resolution exceeds what Metal supports", need, np, have)
	}
	cpp := w.NumChannels * w.PatchSize * w.PatchSize

	e := &encoder{dev: dev, q: dev.NewCommandQueue(), k: kern, w: w, cpp: cpp}
	up := func(x []float32) gpu.Buffer { return gpu.NewBufferOf(dev, x) }
	upMat := func(m vision.GPUMat) mat {
		return mat{q: gpu.NewBufferOf(dev, m.Q), s: gpu.NewBufferOf(dev, m.Scales), rows: m.Rows, cols: m.Cols}
	}
	e.patchW, e.patchB, e.posEmb = up(w.PatchW), up(w.PatchB), up(w.PosEmb)
	e.postLNw, e.postLNb = up(w.PostLNw), up(w.PostLNb)
	e.layers = make([]layer, len(w.Layers))
	for i, L := range w.Layers {
		e.layers[i] = layer{
			ln1w: up(L.LN1w), ln1b: up(L.LN1b), ln2w: up(L.LN2w), ln2b: up(L.LN2b),
			qb: up(L.Qb), kb: up(L.Kb), vb: up(L.Vb), ob: up(L.Ob),
			fc1b: up(L.FC1b), fc2b: up(L.FC2b),
			qw: upMat(L.Qw), kw: upMat(L.Kw), vw: upMat(L.Vw), ow: upMat(L.Ow),
			fc1w: upMat(L.FC1w), fc2w: upMat(L.FC2w),
		}
	}
	f32 := func(n int) gpu.Buffer { return dev.NewBufferLen(n) }
	e.patches = f32(np * cpp)
	e.h, e.n1, e.n2 = f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.qa, e.ka, e.va = f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.att, e.o, e.mlp, e.out = f32(np*hidden), f32(np*hidden), f32(np*hidden), f32(np*hidden)
	e.mid = f32(np * inter)
	wide := max(inter, hidden)
	e.qi8 = gpu.NewBufferOf(dev, make([]int8, np*wide))
	e.qs = f32(np)
	// Enough slots for a whole batched layer: ~24 dispatches x up to 3 ints each,
	// plus headroom. Reset per command buffer, so this is a ceiling on ONE
	// buffer's dispatches, not on the forward.
	const ivSlots, fvSlots = 128, 16
	e.iv = make([]gpu.Buffer, ivSlots)
	for i := range e.iv {
		e.iv[i] = gpu.NewBufferOf(dev, []uint32{0})
	}
	e.fv = make([]gpu.Buffer, fvSlots)
	for i := range e.fv {
		e.fv[i] = gpu.NewBufferOf(dev, []float32{0})
	}
	return e, nil
}

// attnThreadgroupBytes is the threadgroup memory one Attention dispatch needs for np patches:
// the dynamic per-query score row the caller binds at index 0 (np floats), plus the kernel's two
// STATIC threadgroup arrays, smax[LNBLOCK] and ssum[LNBLOCK], which are ViTBlock floats each
// (gpu/metal_vit.go, `attention`). Mirrors qwenmetal/encoder.go's identical helper for the SAME
// shared kernel — duplicated rather than shared because qwenmetal and visionmetal are separate Go
// modules. Counting the static pair matters: against a 32 KiB device budget the extra 2 KiB
// brings the real np limit down from 8,192 (the dynamic term alone) to 7,680.
func attnThreadgroupBytes(np int) int { return np*4 + 2*gpu.ViTBlock*4 }

func vitTG(n int) int {
	if n < gpu.ViTBlock {
		return n
	}
	return gpu.ViTBlock
}

// u32 takes the next uint32 slot, writes v, and returns it. Panics rather than
// wrapping if a command buffer exceeds the ring: silently reusing a slot would
// feed one dispatch another's argument, which is a wrong ANSWER, not a crash.
func (e *encoder) u32(v int) gpu.Buffer {
	if e.ivNext >= len(e.iv) {
		panic("visionmetal: uint32 scalar slots exhausted in one command buffer")
	}
	b := e.iv[e.ivNext]
	e.ivNext++
	b.SetU32(uint32(int32(v)))
	return b
}

// f32 is u32's float sibling.
func (e *encoder) f32(v float32) gpu.Buffer {
	if e.fvNext >= len(e.fv) {
		panic("visionmetal: float scalar slots exhausted in one command buffer")
	}
	b := e.fv[e.fvNext]
	e.fvNext++
	b.Floats()[0] = v
	return b
}

// resetSlots rewinds the rings; call once per command buffer.
func (e *encoder) resetSlots() { e.ivNext, e.fvNext = 0, 0 }

// The dispatch helpers below ENCODE into a caller-supplied command buffer
// (audit M-13). They used to be one Queue.Run1D/Run2D each — one command
// buffer, one commit, one wait, per op — which is ~24 commit+waits per layer
// and a commit floor of roughly 160 ms per so400m image before any arithmetic.
// Now ForwardPatches opens one Encoder per layer and these append to it, so a
// layer costs one commit and one wait.
//
// Each takes its scalar arguments from the per-dispatch slot ring (u32/f32),
// not from shared slots: within one command buffer nothing waits between
// dispatches, so a shared slot would hand every dispatch the last value
// written.

func (e *encoder) gemmF32(enc *gpu.Encoder, a, b, c gpu.Buffer, M, N, K int) {
	p, gx, gy, tgx, tgy := e.k.GEMMF32Plan(M, N, K) // aligned → sg_big, else sg
	enc.Dispatch2D(p, gx, gy, tgx, tgy, a, b, c, e.u32(M), e.u32(N), e.u32(K))
}

func (e *encoder) addBias(enc *gpu.Encoder, x, bias gpu.Buffer, rows, dim int) {
	enc.Dispatch(e.k.AddBias, rows*dim, vitTG(rows*dim), x, bias, e.u32(rows), e.u32(dim))
}

func (e *encoder) addVec(enc *gpu.Encoder, x, v gpu.Buffer, n int) {
	enc.Dispatch(e.k.AddVec, n, vitTG(n), x, v, e.u32(n))
}

func (e *encoder) norm(enc *gpu.Encoder, x, w, b, out gpu.Buffer, rows, dim int) {
	enc.Dispatch(e.k.LayerNorm, rows*gpu.ViTBlock, gpu.ViTBlock, x, w, b, out,
		e.u32(rows), e.u32(dim), e.f32(e.w.Eps))
}

func (e *encoder) gelu(enc *gpu.Encoder, x gpu.Buffer, n int) {
	enc.Dispatch(e.k.GELUTanh, n, vitTG(n), x, e.u32(n))
}

func (e *encoder) attn(enc *gpu.Encoder, q, k, v, out gpu.Buffer, np, nH, hd int, scale float32) {
	if gpu.AttentionTiledEnabledOnMetal && gpu.AttentionTiledEligible(hd) && np >= gpu.AttentionTiledMinNP {
		n, tg := gpu.AttentionTiledDispatch(np, nH)
		enc.Dispatch(e.k.AttentionTiled, n, tg,
			q, k, v, out, e.u32(np), e.u32(nH), e.u32(hd), e.f32(scale))
		return
	}
	enc.DispatchTG(e.k.Attention, np*nH*gpu.ViTBlock, gpu.ViTBlock, np*4,
		q, k, v, out, e.u32(np), e.u32(nH), e.u32(hd), e.f32(scale))
}

// quantRows quantizes src[M,K] into the shared qi8/qs scratch.
//
// SPLIT OUT OF proj so q/k/v can share ONE quantisation (audit M-13). They all
// project the SAME normed activation, and proj used to re-quantise it for each,
// so a layer ran three identical QuantRows dispatches where one serves all
// three. Bit-identical by definition — it is the same kernel on the same input,
// run once instead of thrice.
func (e *encoder) quantRows(enc *gpu.Encoder, src gpu.Buffer, M, K int) {
	enc.Dispatch(e.k.QuantRows, M*gpu.ViTBlock, gpu.ViTBlock, src, e.qi8, e.qs, e.u32(M), e.u32(K))
}

// projQuantized is unfused GEMM + add_bias on already-quantized qi8/qs.
func (e *encoder) projQuantized(enc *gpu.Encoder, m mat, bias, dst gpu.Buffer, M int) {
	K, N := m.cols, m.rows
	p, gx, gy, tgx, tgy := e.k.GEMMW8A8Plan(M, N, K)
	enc.Dispatch2D(p, gx, gy, tgx, tgy,
		e.qi8, e.qs, m.q, m.s, dst, e.u32(M), e.u32(N), e.u32(K))
	enc.Dispatch(e.k.AddBias, M*N, vitTG(M*N), dst, bias, e.u32(M), e.u32(N))
}

// projQuantizedBias runs gemm_w8a8_bias on already-quantized activation in qi8/qs.
func (e *encoder) projQuantizedBias(enc *gpu.Encoder, m mat, bias, dst gpu.Buffer, M int) {
	K, N := m.cols, m.rows
	p, gx, gy, tgx, tgy := e.k.GEMMW8A8BiasPlan(M, N, K)
	if p != (gpu.Pipeline{}) {
		enc.Dispatch2D(p, gx, gy, tgx, tgy,
			e.qi8, e.qs, m.q, m.s, bias, dst, e.u32(M), e.u32(N), e.u32(K))
		return
	}
	// Fallback to unfused
	e.projQuantized(enc, m, bias, dst, M)
}

// projBias is quantRows + projQuantizedBias.
func (e *encoder) projBias(enc *gpu.Encoder, src gpu.Buffer, m mat, bias, dst gpu.Buffer, M int) {
	e.quantRows(enc, src, M, m.cols)
	e.projQuantizedBias(enc, m, bias, dst, M)
}

// projBiasAdd quantizes src, then runs gemm_w8a8_bias_add directly accumulating into residual.
func (e *encoder) projBiasAdd(enc *gpu.Encoder, src gpu.Buffer, m mat, bias, residual gpu.Buffer, M int) {
	e.quantRows(enc, src, M, m.cols)
	K, N := m.cols, m.rows
	p, gx, gy, tgx, tgy := e.k.GEMMW8A8BiasAddPlan(M, N, K)
	if p != (gpu.Pipeline{}) {
		enc.Dispatch2D(p, gx, gy, tgx, tgy,
			e.qi8, e.qs, m.q, m.s, bias, residual, e.u32(M), e.u32(N), e.u32(K))
		return
	}
	// Fallback to unfused
	e.projQuantized(enc, m, bias, e.o, M)
	e.addVec(enc, residual, e.o, M*N)
}

// ForwardPatches runs the resident SigLIP forward on im2col patches [np*(C*P*P)]
// (vision.Encoder.GridPatches) and returns last_hidden_state [np*hidden] — op for op
// the same sequence as vision/encoder.go's Forward and gpu/visioncuda's.
func (e *encoder) ForwardPatches(patches []float32) ([]float32, error) {
	np, hidden, inter := e.w.NumPatches, e.w.Hidden, e.w.Inter
	nH, hd := e.w.NumHeads, e.w.HeadDim
	if len(patches) != np*e.cpp {
		return nil, fmt.Errorf("visionmetal: patches len %d, want %d (np=%d cpp=%d)", len(patches), np*e.cpp, np, e.cpp)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	// One OS-thread pin for the whole forward: every Run1D allocs+drains an
	// NSAutoreleasePool, which objc requires on one thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	copy(e.patches.Floats(), patches) // UMA upload (zero-copy view)

	// ONE COMMAND BUFFER PER LAYER (audit M-13). Each op used to be its own
	// Queue.Run1D/Run2D — commit, wait, drain a pool — about 24 per layer, a
	// commit floor of roughly 160 ms per so400m image before any arithmetic
	// happened. runBatch opens an Encoder, lets the caller append every dispatch
	// of a layer, then commits and waits ONCE.
	runBatch := func(fn func(enc *gpu.Encoder)) error {
		e.resetSlots()
		enc := e.q.Begin()
		fn(enc)
		enc.End()
		return enc.Err()
	}

	// patch embed: h = patches[np,cpp]·patchW[hidden,cpp]ᵀ + patchB + posEmb
	if err := runBatch(func(enc *gpu.Encoder) {
		e.gemmF32(enc, e.patches, e.patchW, e.h, np, hidden, e.cpp)
		e.addBias(enc, e.h, e.patchB, np, hidden)
		e.addVec(enc, e.h, e.posEmb, np*hidden)
	}); err != nil {
		return nil, fmt.Errorf("visionmetal: patch embed: %w", err)
	}

	scale := float32(1.0 / math.Sqrt(float64(hd)))
	for i := range e.layers {
		L := &e.layers[i]
		if err := runBatch(func(enc *gpu.Encoder) {
			// attention block (pre-LN, residual)
			e.norm(enc, e.h, L.ln1w, L.ln1b, e.n1, np, hidden)
			// q/k/v all project the SAME normed activation, so quantise it ONCE
			// and bind the result three times rather than running three
			// identical QuantRows dispatches.
			e.quantRows(enc, e.n1, np, L.qw.cols)
			e.projQuantizedBias(enc, L.qw, L.qb, e.qa, np)
			e.projQuantizedBias(enc, L.kw, L.kb, e.ka, np)
			e.projQuantizedBias(enc, L.vw, L.vb, e.va, np)
			e.attn(enc, e.qa, e.ka, e.va, e.att, np, nH, hd, scale)
			// Fused O projection + bias + residual add directly into h
			e.projBiasAdd(enc, e.att, L.ow, L.ob, e.h, np)
			// MLP block (pre-LN, residual): fc2(geluTanh(fc1(x)))
			e.norm(enc, e.h, L.ln2w, L.ln2b, e.n2, np, hidden)
			e.projBias(enc, e.n2, L.fc1w, L.fc1b, e.mid, np)
			e.gelu(enc, e.mid, np*inter)
			// Fused FC2 projection + bias + residual add directly into h
			e.projBiasAdd(enc, e.mid, L.fc2w, L.fc2b, e.h, np)
		}); err != nil {
			return nil, fmt.Errorf("visionmetal: layer %d: %w", i, err)
		}
	}
	if err := runBatch(func(enc *gpu.Encoder) {
		e.norm(enc, e.h, e.postLNw, e.postLNb, e.out, np, hidden)
	}); err != nil {
		return nil, fmt.Errorf("visionmetal: post-LN: %w", err)
	}

	dst := make([]float32, np*hidden)
	copy(dst, e.out.Floats()) // UMA download (the last Run1D already waited)
	return dst, nil
}

// Close releases the device buffers and objects (buffers first, so no in-flight work
// references freed memory; the forward's last Run1D already waited).
func (e *encoder) Close() {
	e.dev.ReleaseAll()
	e.dev.ReleaseObjects()
}

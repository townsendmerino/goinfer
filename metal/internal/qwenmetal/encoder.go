//go:build darwin

// Package qwenmetal is the native-Metal vision.QwenResidentEncoder — the Qwen2.5-VL
// vision tower running on Apple GPUs, cgo-free (docs/task-native-gpu.md, Phase 3). It is
// the Apple counterpart of gpu/qwencuda, and mirrors it op for op; the device-level
// differences are all here.
//
// Importing this package is the opt-in; aikit's core `vision` never imports `gpu`, so the
// default build stays pure-Go CPU. This package's init() plugs the factory into
// vision.RegisterQwenResident, and QwenVisionEncoder.EnableResident then routes here.
//
// WHAT RUNS WHERE, AND WHY (identical contract to gpu/qwencuda)
// ------------------------------------------------------------
// The transformer blocks run on the device. Two things stay on the host, both on purpose:
//
//   - The WINDOW PERMUTATION. Qwen reorders patches into window groups before the blocks
//     and back afterwards. Rather than a gather kernel, this permutes the PIXEL ROWS
//     before upload: the patch embed is row-wise, so permuting its input permutes its
//     output identically, and the de-window is a permutation of the result on the way
//     back. The index arithmetic stays in vision.BuildWindowPlan where the CPU path's
//     version already lives — reimplementing it is exactly where a ViT goes silently
//     wrong.
//   - The PATCH MERGER, per the seam's contract: three small ops over n_patches/merge²
//     groups against many blocks of tower, left on the CPU (vision.MergeHidden).
//
// Metal ↔ CUDA divergences (inverses of gpu/qwencuda's): dispatchThreads launches exactly
// n threads (no bounds checks); Apple UMA makes a buffer's Floats()/U32s() a zero-copy
// live view (no explicit upload/download); scalars bind as 1-element buffers; and the
// whole forward runs under ONE runtime.LockOSThread, because each dispatch's Run1D
// allocates and drains an NSAutoreleasePool that objc requires on a single thread.
package qwenmetal

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/townsendmerino/goinfer/internal/vision"
	gpu "github.com/townsendmerino/goinfer/metal/internal/gpudev"
)

func init() {
	vision.RegisterQwenResident(func(e *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newEncoder(e)
	})
}

// mat is one resident weight. Qwen towers load fp32 (the parity configuration) or int8;
// both are supported, and the GEMM used per call follows this flag.
type mat struct {
	quant      bool
	a, b       gpu.Buffer // int8 codes + scales when quant; a is the f32 matrix otherwise
	rows, cols int
}

type block struct {
	norm1w, norm2w gpu.Buffer
	qkvw           mat
	qkvb           gpu.Buffer
	projw          mat
	projb          gpu.Buffer
	gatew, upw     mat
	gateb, upb     gpu.Buffer
	downw          mat
	downb          gpu.Buffer
}

type encoder struct {
	dev *gpu.Device
	q   gpu.Queue
	k   gpu.ViT
	src *vision.QwenVisionEncoder
	w   vision.QwenGPUWeights

	patchW gpu.Buffer
	blocks []block

	// PER-DISPATCH scalar slots (M-15, audit-metal-2026-09-12.md, porting visionmetal's
	// runBatch + ring — see that package's own encoder.go for the full rationale). Metal
	// has no by-value kernel arg, so every int/float argument is a one-word buffer. This
	// used to be three shared uint32 slots plus two shared float slots, rewritten before
	// each dispatch — safe only because each Run1D committed AND WAITED, so a scalar was
	// fully consumed before the next write. Batching a block into one command buffer
	// removes that guarantee: every dispatch in the buffer would see the LAST value
	// written, not its own. So each dispatch now takes its own slot from this ring, reset
	// once per command buffer. Sized like visionmetal's (128/16): the worst-case block
	// (every projection int8-quantized) uses ~48 uint32 slots and 3 float slots, so this
	// carries the same >2x headroom visionmetal's own sizing note describes.
	iv     []gpu.Buffer // uint32 slots
	ivNext int
	fv     []gpu.Buffer // float32 slots (eps, attention scale)
	fvNext int

	mu sync.Mutex
	// Scratch is sized to the patch count of the first forward and grown on demand:
	// Qwen is dynamic-resolution, so unlike SigLIP there is no fixed np to preallocate.
	cap               int
	pix, h, n1, n2    gpu.Buffer
	qkv, att, projOut gpu.Buffer
	gate, up          gpu.Buffer
	qi8, qs           gpu.Buffer
	cosB, sinB        gpu.Buffer
	segS, segE        gpu.Buffer
}

func newEncoder(src *vision.QwenVisionEncoder) (enc *encoder, err error) {
	w := src.GPUWeights()
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	defer func() {
		if r := recover(); r != nil {
			dev.ReleaseAll()
			dev.ReleaseObjects()
			enc, err = nil, fmt.Errorf("qwenmetal: device upload failed: %v", r)
		}
	}()
	kern, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	e := &encoder{dev: dev, q: dev.NewCommandQueue(), k: kern, src: src, w: w}
	upF := func(x []float32) gpu.Buffer { return gpu.NewBufferOf(dev, x) }
	upM := func(m vision.QwenGPUMat) mat {
		if m.Quantized {
			return mat{quant: true, a: gpu.NewBufferOf(dev, m.Q), b: gpu.NewBufferOf(dev, m.Scales), rows: m.Rows, cols: m.Cols}
		}
		return mat{a: gpu.NewBufferOf(dev, m.F32), rows: m.Rows, cols: m.Cols}
	}
	e.patchW = upF(w.PatchW)
	e.blocks = make([]block, len(w.Blocks))
	for i, B := range w.Blocks {
		e.blocks[i] = block{
			norm1w: upF(B.Norm1w), norm2w: upF(B.Norm2w),
			qkvw: upM(B.QKVw), qkvb: upF(B.QKVb),
			projw: upM(B.Projw), projb: upF(B.Projb),
			gatew: upM(B.Gatew), upw: upM(B.Upw),
			gateb: upF(B.Gateb), upb: upF(B.Upb),
			downw: upM(B.Downw), downb: upF(B.Downb),
		}
	}
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

// scratchSet is every per-call scratch buffer, so a growth can build the new set whole and install it in one step.
type scratchSet struct {
	pix, h, n1, n2    gpu.Buffer
	qkv, att, projOut gpu.Buffer
	gate, up          gpu.Buffer
	qi8, qs           gpu.Buffer
	cosB, sinB        gpu.Buffer
	segS, segE        gpu.Buffer
}

func (s *scratchSet) all() []gpu.Buffer {
	return []gpu.Buffer{s.pix, s.h, s.n1, s.n2, s.qkv, s.att, s.projOut, s.gate, s.up, s.qi8, s.qs, s.cosB, s.sinB, s.segS, s.segE}
}

func (e *encoder) scratch() scratchSet {
	return scratchSet{pix: e.pix, h: e.h, n1: e.n1, n2: e.n2, qkv: e.qkv, att: e.att, projOut: e.projOut, gate: e.gate, up: e.up, qi8: e.qi8, qs: e.qs,
		cosB: e.cosB, sinB: e.sinB, segS: e.segS, segE: e.segE}
}

func (e *encoder) setScratch(s scratchSet, cap int) {
	e.pix, e.h, e.n1, e.n2, e.qkv, e.att, e.projOut, e.gate, e.up, e.qi8, e.qs = s.pix, s.h, s.n1, s.n2, s.qkv, s.att, s.projOut, s.gate, s.up, s.qi8, s.qs
	e.cosB, e.sinB, e.segS, e.segE = s.cosB, s.sinB, s.segS, s.segE
	e.cap = cap
}

// allocScratch builds a scratch set for n patches. The gpu layer's buffer constructors PANIC on an allocation failure (MustBuf's contract), so this recovers: every buffer
// already made is released and the failure comes back as an error, never as a panic on the caller's goroutine.
func (e *encoder) allocScratch(n int) (s scratchSet, err error) {
	var made []gpu.Buffer
	defer func() {
		if r := recover(); r != nil {
			for _, b := range made {
				e.dev.ReleaseBuf(b)
			}
			s, err = scratchSet{}, fmt.Errorf("qwenmetal: scratch for %d patches: %v", n, r)
		}
	}()
	H, I, hd := e.w.Hidden, e.w.Inter, e.w.HeadDim
	keep := func(b gpu.Buffer) gpu.Buffer { made = append(made, b); return b }
	f32 := func(c int) gpu.Buffer { return keep(e.dev.NewBufferLen(c)) }
	s.pix = f32(n * e.w.PatchDim)
	s.h, s.n1, s.n2 = f32(n*H), f32(n*H), f32(n*H)
	s.qkv = f32(n * 3 * H)
	s.att, s.projOut = f32(n*H), f32(n*H)
	s.gate, s.up = f32(n*I), f32(n*I)
	wide := max(I, H)
	s.qi8 = keep(gpu.NewBufferOf(e.dev, make([]int8, n*wide)))
	s.qs = f32(n)
	s.cosB, s.sinB = f32(n*hd), f32(n*hd)
	s.segS = keep(gpu.NewBufferOf(e.dev, make([]uint32, n)))
	s.segE = keep(gpu.NewBufferOf(e.dev, make([]uint32, n)))
	return s, nil
}

// ensure grows the per-call scratch to hold n patches. The new set is allocated BEFORE the old one is released and installed only if every buffer was made, so a
// failed growth leaves the encoder exactly as it was (it used to release the old buffers first and allocate in place: a failure then left e.cap pointing at buffers that had
// been freed). Releasing first is what let a growth that fits alone succeed under a tight budget, so when the new-first attempt fails the old set is released and the
// allocation retried once; if that fails too the encoder is left cold (cap 0, no buffers) and the next call reallocates. Either way: an error, never a panic and never a
// dangling buffer. Buffers are not kept at a high-water mark for every shape, because a dynamic-resolution tower can see a huge image once and small ones thereafter.
// (Same change as gpu/qwencuda's; compiled for darwin but NOT run on Metal by the session that wrote it.)
func (e *encoder) ensure(n int) error {
	if n <= e.cap {
		return nil
	}
	s, err := e.allocScratch(n)
	if err == nil {
		e.releaseScratch()
		e.setScratch(s, n)
		return nil
	}
	if e.cap == 0 {
		return err
	}
	e.releaseScratch()
	e.setScratch(scratchSet{}, 0)
	if s, err2 := e.allocScratch(n); err2 == nil {
		e.setScratch(s, n)
		return nil
	}
	return err
}

func (e *encoder) releaseScratch() {
	if e.cap == 0 {
		return
	}
	cur := e.scratch()
	for _, b := range cur.all() {
		e.dev.ReleaseBuf(b)
	}
}

// u32 takes the next uint32 slot, writes v, and returns it. Panics rather than
// wrapping if a command buffer exceeds the ring: silently reusing a slot would
// feed one dispatch another's argument, which is a wrong ANSWER, not a crash.
func (e *encoder) u32(v int) gpu.Buffer {
	if e.ivNext >= len(e.iv) {
		panic("qwenmetal: uint32 scalar slots exhausted in one command buffer")
	}
	b := e.iv[e.ivNext]
	e.ivNext++
	b.SetU32(uint32(int32(v)))
	return b
}

// f32 is u32's float sibling.
func (e *encoder) f32(v float32) gpu.Buffer {
	if e.fvNext >= len(e.fv) {
		panic("qwenmetal: float scalar slots exhausted in one command buffer")
	}
	b := e.fv[e.fvNext]
	e.fvNext++
	b.Floats()[0] = v
	return b
}

// resetSlots rewinds the rings; call once per command buffer.
func (e *encoder) resetSlots() { e.ivNext, e.fvNext = 0, 0 }

func vitTG(n int) int {
	if n < gpu.ViTBlock {
		return n
	}
	return gpu.ViTBlock
}

// uploadSeg writes int32 per-patch bounds into a resident uint32 buffer in place (UMA).
// Bounds are non-negative, so the uint32 bit pattern equals the int32 the kernel reads.
func uploadSeg(b gpu.Buffer, s []int32) {
	v := b.U32s()
	for i, x := range s {
		v[i] = uint32(x)
	}
}

// proj encodes one projection into enc: dst[M,N] = src[M,K] · w[N,K]ᵀ + bias, quantizing
// the activation first when the weight is int8 — the same op sequence as gpu/qwencuda's.
func (e *encoder) proj(enc *gpu.Encoder, src gpu.Buffer, m mat, bias, dst gpu.Buffer, M int) {
	K, N := m.cols, m.rows
	if m.quant {
		// W8A8 has no simdgroup_matrix form; GEMMW8A8Plan picks the register-blocked
		// fast path when K%16==0 (M-15), else the bounds-checked tiled fallback.
		enc.Dispatch(e.k.QuantRows, M*gpu.ViTBlock, gpu.ViTBlock, src, e.qi8, e.qs, e.u32(M), e.u32(K))
		if bias.Len() > 0 {
			p, gx, gy, tgx, tgy := e.k.GEMMW8A8BiasPlan(M, N, K)
			if p != (gpu.Pipeline{}) {
				enc.Dispatch2D(p, gx, gy, tgx, tgy, e.qi8, e.qs, m.a, m.b, bias, dst, e.u32(M), e.u32(N), e.u32(K))
				return
			}
		}
		p, gx, gy, tgx, tgy := e.k.GEMMW8A8Plan(M, N, K)
		enc.Dispatch2D(p, gx, gy, tgx, tgy, e.qi8, e.qs, m.a, m.b, dst, e.u32(M), e.u32(N), e.u32(K))
	} else {
		if bias.Len() > 0 {
			p, gx, gy, tgx, tgy := e.k.GEMMF32BiasPlan(M, N, K)
			if p != (gpu.Pipeline{}) {
				enc.Dispatch2D(p, gx, gy, tgx, tgy, src, m.a, bias, dst, e.u32(M), e.u32(N), e.u32(K))
				return
			}
		}
		p, gx, gy, tgx, tgy := e.k.GEMMF32Plan(M, N, K) // aligned → sg_big, else sg
		enc.Dispatch2D(p, gx, gy, tgx, tgy, src, m.a, dst, e.u32(M), e.u32(N), e.u32(K))
	}
	if bias.Len() == 0 {
		return
	}
	enc.Dispatch(e.k.AddBias, M*N, vitTG(M*N), dst, bias, e.u32(M), e.u32(N))
}

// projBiasAdd encodes projection with fused bias and residual addition:
// residual[M,N] += src[M,K] · w[N,K]ᵀ + bias.
func (e *encoder) projBiasAdd(enc *gpu.Encoder, src gpu.Buffer, m mat, bias, residual gpu.Buffer, M int) {
	K, N := m.cols, m.rows
	if m.quant && bias.Len() > 0 {
		p, gx, gy, tgx, tgy := e.k.GEMMW8A8BiasAddPlan(M, N, K)
		if p != (gpu.Pipeline{}) {
			enc.Dispatch(e.k.QuantRows, M*gpu.ViTBlock, gpu.ViTBlock, src, e.qi8, e.qs, e.u32(M), e.u32(K))
			enc.Dispatch2D(p, gx, gy, tgx, tgy, e.qi8, e.qs, m.a, m.b, bias, residual, e.u32(M), e.u32(N), e.u32(K))
			return
		}
	} else if !m.quant && bias.Len() > 0 {
		p, gx, gy, tgx, tgy := e.k.GEMMF32BiasAddPlan(M, N, K)
		if p != (gpu.Pipeline{}) {
			enc.Dispatch2D(p, gx, gy, tgx, tgy, src, m.a, bias, residual, e.u32(M), e.u32(N), e.u32(K))
			return
		}
	}
	// Fallback to unfused proj + AddVec
	e.proj(enc, src, m, bias, e.projOut, M)
	enc.Dispatch(e.k.AddVec, M*N, vitTG(M*N), residual, e.projOut, e.u32(M*N))
}

func (e *encoder) rms(enc *gpu.Encoder, src, w, dst gpu.Buffer, rows, dim int) {
	enc.Dispatch(e.k.RMSNorm, rows*gpu.ViTBlock, gpu.ViTBlock, src, w, dst, e.u32(rows), e.u32(dim), e.f32(1e-6))
}

// ForwardViT runs the ViT blocks on the device and returns the pre-merge hidden state in
// ORIGINAL patch order, matching the CPU contract. One OS-thread pin spans the whole
// forward (every Run1D allocs+drains an NSAutoreleasePool, which objc requires on one
// thread).
func (e *encoder) ForwardViT(pixelValues []float32, gridTHW [][3]int) (out []float32, err error) {
	plan, err := e.src.BuildWindowPlan(gridTHW)
	if err != nil {
		return nil, err
	}
	n := plan.NPatches

	// Threadgroup-memory budget check (audit C-02). AttentionSeg stages a
	// per-query score row in DYNAMIC threadgroup memory of maxSeg*4 bytes, on top
	// of the kernel's two static threadgroup arrays (smax/ssum, ViTBlock floats
	// each). A dispatch whose total exceeds the device maximum — ~32 KiB on Apple
	// GPUs — aborts the command buffer, and waitUntilCompleted returns cleanly
	// from an abort: e.att simply keeps the previous layer's contents and the
	// forward returns a plausible, wrong hidden state. mustCmdBufOK now turns
	// that into a loud panic rather than silence, but declining here is better
	// still: it is the documented obligation on MaxThreadgroupMemoryLength, it
	// costs nothing, and it names the real limit instead of reporting a GPU
	// fault. CUDA's analogue already fails the launch with an error.
	//
	// Both branches of the per-layer dispatch are covered by taking the larger of
	// the window and full-attention segment bounds.
	H, I, nH, hd := e.w.Hidden, e.w.Inter, e.w.NumHeads, e.w.HeadDim
	maxSegAny := max(plan.MaxWinSeg, plan.MaxFullSeg)
	if !gpu.AttentionTiledEligible(hd) {
		if need, have := attnThreadgroupBytes(maxSegAny), e.dev.MaxThreadgroupMemoryLength(); need > have {
			return nil, fmt.Errorf("gpu: qwen ViT attention needs %d B of threadgroup memory for %d patches (max segment %d) but the device allows %d — reduce max_pixels", need, n, maxSegAny, have)
		}
	}
	pd := e.w.PatchDim
	if len(pixelValues) != n*pd {
		return nil, fmt.Errorf("qwenmetal: pixel_values len %d, want %d", len(pixelValues), n*pd)
	}
	merge := e.w.SpatialMergeSize
	mergeUnit := merge * merge
	groups := n / mergeUnit

	e.mu.Lock()
	defer e.mu.Unlock()
	defer func() {
		// A panic anywhere below (a Metal buffer failure is a panic by the gpu layer's contract) comes back as an error rather than killing the caller's goroutine.
		if r := recover(); r != nil {
			out, err = nil, fmt.Errorf("qwenmetal: forward failed on the device: %v", r)
		}
	}()
	if err := e.ensure(n); err != nil {
		return nil, err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Permute pixel rows into WINDOW order on the host, then UMA-copy — the patch embed is
	// row-wise, so this makes the device's hidden state window-ordered without a gather.
	pix := e.pix.Floats()
	for g := range groups {
		src := plan.WinIdx[g]
		for u := range mergeUnit {
			dp, sp := (g*mergeUnit+u)*pd, (src*mergeUnit+u)*pd
			copy(pix[dp:dp+pd], pixelValues[sp:sp+pd])
		}
	}
	copy(e.cosB.Floats(), plan.Cos)
	copy(e.sinB.Floats(), plan.Sin)

	// ONE COMMAND BUFFER PER BLOCK (M-15, audit-metal-2026-09-12.md — porting visionmetal's
	// own M-13 fix). Each op used to be its own Queue.Run1D/Run1DTG/Run2D — commit, wait,
	// drain a pool — 17-22 per block (544-704 over a 32-block tower), a submit floor of
	// roughly 136-176 ms/image before any arithmetic. runBatch opens an Encoder, lets the
	// caller append every dispatch of one unit of work, then commits and waits ONCE.
	// Scoped per BLOCK (not per whole forward): segS/segE's conditional host re-upload
	// below must land before the command buffer containing that block's AttentionSeg
	// dispatch is opened, and each block's own quant/fp32 branching plus its two AddVec
	// residual adds are independent of every other block's — a per-block boundary is the
	// natural, and only safe, batching granularity here (the point ForwardEmbPipe's own
	// AttentionSeg dependency on segS/segE already forces this decoder-side).
	runBatch := func(fn func(enc *gpu.Encoder)) error {
		e.resetSlots()
		enc := e.q.Begin()
		fn(enc)
		enc.End()
		return enc.Err()
	}

	// patch embed: h = pixels[n,pd] · patchW[H,pd]ᵀ (no bias).
	if err := runBatch(func(enc *gpu.Encoder) {
		pp, pgx, pgy, ptgx, ptgy := e.k.GEMMF32Plan(n, H, pd) // aligned → sg_big, else sg
		enc.Dispatch2D(pp, pgx, pgy, ptgx, ptgy, e.pix, e.patchW, e.h, e.u32(n), e.u32(H), e.u32(pd))
	}); err != nil {
		return nil, fmt.Errorf("qwenmetal: patch embed: %w", err)
	}

	scale := float32(1.0 / math.Sqrt(float64(hd)))
	curFull := -1 // which segment bounds are currently uploaded: 1 full, 0 windowed
	for li := range e.blocks {
		B := &e.blocks[li]
		full := e.src.IsFullAtt(li)
		want := 0
		if full {
			want = 1
		}
		if want != curFull {
			ss, se := plan.WinStart, plan.WinEnd
			if full {
				ss, se = plan.FullStart, plan.FullEnd
			}
			uploadSeg(e.segS, ss)
			uploadSeg(e.segE, se)
			curFull = want
		}
		maxSeg := plan.MaxWinSeg
		if full {
			maxSeg = plan.MaxFullSeg
		}

		if err := runBatch(func(enc *gpu.Encoder) {
			// --- attention block ---
			e.rms(enc, e.h, B.norm1w, e.n1, n, H)
			e.proj(enc, e.n1, B.qkvw, B.qkvb, e.qkv, n)
			enc.Dispatch(e.k.RopeQK, n*nH*(hd/2), vitTG(n*nH*(hd/2)), e.qkv, e.cosB, e.sinB, e.u32(n), e.u32(nH), e.u32(hd))
			if gpu.AttentionTiledEnabledOnMetal && gpu.AttentionTiledEligible(hd) && n >= gpu.AttentionTiledMinNP {
				nGrid, tg := gpu.AttentionSegTiledDispatch(n, nH)
				enc.Dispatch(e.k.AttentionSegTiled, nGrid, tg,
					e.qkv, e.att, e.segS, e.segE, e.u32(n), e.u32(nH), e.u32(hd), e.f32(scale))
			} else {
				enc.DispatchTG(e.k.AttentionSeg, n*nH*gpu.ViTBlock, gpu.ViTBlock, maxSeg*4,
					e.qkv, e.att, e.segS, e.segE, e.u32(n), e.u32(nH), e.u32(hd), e.f32(scale))
			}
			e.projBiasAdd(enc, e.att, B.projw, B.projb, e.h, n)
			// --- gated SiLU MLP ---
			e.rms(enc, e.h, B.norm2w, e.n2, n, H)
			e.proj(enc, e.n2, B.gatew, B.gateb, e.gate, n)
			e.proj(enc, e.n2, B.upw, B.upb, e.up, n)
			enc.Dispatch(e.k.SiLUMul, n*I, vitTG(n*I), e.gate, e.up, e.u32(n*I))
			e.projBiasAdd(enc, e.gate, B.downw, B.downb, e.h, n)
		}); err != nil {
			return nil, fmt.Errorf("qwenmetal: block %d: %w", li, err)
		}
	}

	// de-window back to original patch order (the last dispatch already waited; UMA view).
	hWin := e.h.Floats()
	out = make([]float32, n*H)
	for g := range groups {
		dst := plan.WinIdx[g]
		for u := range mergeUnit {
			dp, sp := (dst*mergeUnit+u)*H, (g*mergeUnit+u)*H
			copy(out[dp:dp+H], hWin[sp:sp+H])
		}
	}
	return out, nil
}

// Close releases the device buffers and objects (buffers first, so no in-flight work
// references freed memory; the forward's last Run1D already waited).
func (e *encoder) Close() {
	e.dev.ReleaseAll()
	e.dev.ReleaseObjects()
}

// attnThreadgroupBytes is the threadgroup memory one AttentionSeg dispatch needs
// when the largest per-patch segment in the plan is maxSeg: the dynamic score row
// the caller binds at index 0 (maxSeg floats), plus the kernel's two STATIC
// threadgroup arrays, smax[LNBLOCK] and ssum[LNBLOCK], which are ViTBlock floats
// each (gpu/metal_vit.go, attention_seg).
//
// Counting the static pair matters: the audit put the abort threshold at
// n > 8192 patches from the dynamic term alone, but against a 32 KiB device
// budget the extra 2 KiB brings the real limit down to a segment of 7680.
// Factored out of ForwardViT so it can be tested without a checkpoint — the
// qwenmetal suite skips entirely when testdata/qwen25vl-vision-tiny is absent.
func attnThreadgroupBytes(maxSeg int) int { return maxSeg*4 + 2*gpu.ViTBlock*4 }

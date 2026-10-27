//go:build linux

// Package qwencuda is the native-CUDA vision.QwenResidentEncoder — the Qwen2.5-VL
// vision tower running on the GPU, cgo-free (docs/task-native-gpu.md, Phase 3).
//
// Importing this package is the opt-in; aikit's core `vision` never imports `gpu`, so
// the default build stays pure-Go CPU.
//
// WHAT RUNS WHERE, AND WHY
// ------------------------
// The transformer blocks run on the device. Two things stay on the host, both on
// purpose:
//
//   - The WINDOW PERMUTATION. Qwen reorders patches into window groups before the
//     blocks and back afterwards. Rather than a gather kernel, this permutes the
//     PIXEL ROWS before upload: the patch embed is row-wise, so permuting its input
//     permutes its output identically, and the de-window is a permutation of the
//     result on the way back. Two kernels saved, and the index arithmetic stays in
//     vision.BuildWindowPlan where the CPU path's version already lives — reusing it
//     rather than reimplementing it is the point, since patch indexing is exactly
//     where a ViT goes silently wrong.
//   - The PATCH MERGER, per the seam's contract: it is three small ops over
//     n_patches/merge² groups, against 32 blocks of tower.
//
// Everything else — patch embed, RMSNorm, fused QKV, 2D rotary, windowed/full
// attention, gated SiLU MLP, residuals — is on-device, with ONE Sync per forward.
package qwencuda

import (
	"fmt"
	"math"
	"sync"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/mem"
	"github.com/townsendmerino/goinfer/internal/vision"
)

func init() {
	vision.RegisterQwenResident(func(e *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newEncoder(e)
	})
}

// mat is one resident weight. Qwen towers load fp32 (the parity configuration) or
// int8; both are supported, and the GEMM used per call follows this flag.
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

	mu sync.Mutex
	// Scratch is sized to the patch count of the first forward and grown on demand:
	// Qwen is dynamic-resolution, so unlike SigLIP there is no fixed np to preallocate.
	cap               int
	pix, h, n1, n2    gpu.Buffer
	qkv, att, projOut gpu.Buffer
	gate, up          gpu.Buffer
	qi8, qs           gpu.Buffer
	cosB, sinB        gpu.Buffer
	// TWO resident bound pairs, not one rewritten mid-forward (audit C-01).
	// segSWin/segEWin hold the windowed bounds, segSFull/segEFull the
	// full-attention ones; both are uploaded once before the layer loop and
	// the loop only chooses which pair to BIND. See the note at the upload.
	segSWin, segEWin   gpu.Buffer
	segSFull, segEFull gpu.Buffer
}

func newEncoder(src *vision.QwenVisionEncoder) (enc *encoder, err error) {
	w := src.GPUWeights()
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return nil, err
	}
	defer func() {
		if r := recover(); r != nil {
			dev.ReleaseObjects()
			enc, err = nil, fmt.Errorf("qwencuda: device upload failed: %v", r)
		}
	}()
	kern, err := dev.NewViT()
	if err != nil {
		dev.ReleaseObjects()
		return nil, err
	}
	e := &encoder{dev: dev, q: dev.NewCommandQueue(), k: kern, src: src, w: w}
	upF := func(x []float32) gpu.Buffer { return gpu.NewBufferOfOn(dev, mem.LineCompanion, x) }
	upM := func(m vision.QwenGPUMat) mat {
		if m.Quantized {
			return mat{quant: true, a: gpu.NewBufferOfOn(dev, mem.LineCompanion, m.Q), b: gpu.NewBufferOfOn(dev, mem.LineCompanion, m.Scales), rows: m.Rows, cols: m.Cols}
		}
		return mat{a: gpu.NewBufferOfOn(dev, mem.LineCompanion, m.F32), rows: m.Rows, cols: m.Cols}
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
	return e, nil
}

// scratchSet is every per-call scratch buffer, so a growth can build the new set whole and install it in one step.
type scratchSet struct {
	pix, h, n1, n2     gpu.Buffer
	qkv, att, projOut  gpu.Buffer
	gate, up           gpu.Buffer
	qi8, qs            gpu.Buffer
	cosB, sinB         gpu.Buffer
	segSWin, segEWin   gpu.Buffer
	segSFull, segEFull gpu.Buffer
}

func (s *scratchSet) all() []gpu.Buffer {
	return []gpu.Buffer{s.pix, s.h, s.n1, s.n2, s.qkv, s.att, s.projOut, s.gate, s.up, s.qi8, s.qs, s.cosB, s.sinB, s.segSWin, s.segEWin, s.segSFull, s.segEFull}
}

func (e *encoder) scratch() scratchSet {
	return scratchSet{pix: e.pix, h: e.h, n1: e.n1, n2: e.n2, qkv: e.qkv, att: e.att, projOut: e.projOut, gate: e.gate, up: e.up, qi8: e.qi8, qs: e.qs,
		cosB: e.cosB, sinB: e.sinB, segSWin: e.segSWin, segEWin: e.segEWin, segSFull: e.segSFull, segEFull: e.segEFull}
}

func (e *encoder) setScratch(s scratchSet, cap int) {
	e.pix, e.h, e.n1, e.n2, e.qkv, e.att, e.projOut, e.gate, e.up, e.qi8, e.qs = s.pix, s.h, s.n1, s.n2, s.qkv, s.att, s.projOut, s.gate, s.up, s.qi8, s.qs
	e.cosB, e.sinB, e.segSWin, e.segEWin, e.segSFull, e.segEFull = s.cosB, s.sinB, s.segSWin, s.segEWin, s.segSFull, s.segEFull
	e.cap = cap
}

// allocScratch builds a scratch set for n patches. gpu.NewBufferLenOf PANICS on an allocation failure (its contract), so this recovers: every buffer already made
// is released and the failure comes back as an error, never as a panic on the caller's goroutine.
func (e *encoder) allocScratch(n int) (s scratchSet, err error) {
	var made []gpu.Buffer
	defer func() {
		if r := recover(); r != nil {
			for _, b := range made {
				e.dev.ReleaseBuf(b)
			}
			s, err = scratchSet{}, fmt.Errorf("qwencuda: scratch for %d patches: %v", n, r)
		}
	}()
	H, I, hd := e.w.Hidden, e.w.Inter, e.w.HeadDim
	f32 := func(c int) gpu.Buffer {
		b := gpu.NewBufferLenOfOn[float32](e.dev, mem.LineCompanion, c)
		made = append(made, b)
		return b
	}
	i32 := func(c int) gpu.Buffer {
		b := gpu.NewBufferLenOfOn[int32](e.dev, mem.LineCompanion, c)
		made = append(made, b)
		return b
	}
	s.pix = f32(n * e.w.PatchDim)
	s.h, s.n1, s.n2 = f32(n*H), f32(n*H), f32(n*H)
	s.qkv = f32(n * 3 * H)
	s.att, s.projOut = f32(n*H), f32(n*H)
	s.gate, s.up = f32(n*I), f32(n*I)
	wide := max(I, H)
	s.qi8 = func(c int) gpu.Buffer {
		b := gpu.NewBufferLenOfOn[int8](e.dev, mem.LineCompanion, c)
		made = append(made, b)
		return b
	}(n * wide)
	s.qs = f32(n)
	s.cosB, s.sinB = f32(n*hd), f32(n*hd)
	s.segSWin, s.segEWin, s.segSFull, s.segEFull = i32(n), i32(n), i32(n), i32(n)
	return s, nil
}

// ensure grows the per-call scratch to hold n patches. The new set is allocated BEFORE the old one is released and installed only if every buffer was made, so a
// failed growth leaves the encoder exactly as it was (it used to release the old buffers first and allocate in place: a failure then left e.cap pointing at buffers
// that had been freed). Releasing first is what let a growth that fits alone succeed under a tight budget, so when the new-first attempt fails the old set is released
// and the allocation retried once; if that fails too the encoder is left cold (cap 0, no buffers) and the next call reallocates. Either way: an error, never a panic and
// never a dangling buffer. Buffers are not kept at a high-water mark for every shape, because a dynamic-resolution tower can see a huge image once and small ones thereafter.
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

// proj runs one projection: dst[M,N] = src[M,K] · w[N,K]ᵀ + bias, quantizing the
// activation first when the weight is int8.
func (e *encoder) proj(src gpu.Buffer, m mat, bias, dst gpu.Buffer, M int) error {
	K, N := m.cols, m.rows
	if m.quant {
		if err := e.q.Launch(e.k.QuantRows, gpu.RowGrid(M),
			gpu.Arg(src), gpu.Arg(e.qi8), gpu.Arg(e.qs),
			gpu.ArgValue(int32(M)), gpu.ArgValue(int32(K))); err != nil {
			return err
		}
		if bias.Len() > 0 {
			gp, gc := e.k.GEMMW8A8BiasPlan(M, N, K)
			if gp != (gpu.Pipeline{}) {
				return e.q.Launch(gp, gc,
					gpu.Arg(e.qi8), gpu.Arg(e.qs), gpu.Arg(m.a), gpu.Arg(m.b), gpu.Arg(bias), gpu.Arg(dst),
					gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K)))
			}
		}
		// GEMMW8A8Plan takes the register-blocked int8 kernel on aligned shapes
		// and gemm_w8a8_tiled otherwise (audit M-12). Identical bits either way,
		// so this is a pure dispatch change.
		gp, gc := e.k.GEMMW8A8Plan(M, N, K)
		if err := e.q.Launch(gp, gc,
			gpu.Arg(e.qi8), gpu.Arg(e.qs), gpu.Arg(m.a), gpu.Arg(m.b), gpu.Arg(dst),
			gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K))); err != nil {
			return err
		}
	} else {
		if bias.Len() > 0 {
			gp, gcfg := e.k.GEMMF32BiasPlan(M, N, K)
			if gp != (gpu.Pipeline{}) {
				return e.q.Launch(gp, gcfg,
					gpu.Arg(src), gpu.Arg(m.a), gpu.Arg(bias), gpu.Arg(dst),
					gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K)))
			}
		}
		gp, gcfg := e.k.GEMMF32Plan(M, N, K)
		if err := e.q.Launch(gp, gcfg,
			gpu.Arg(src), gpu.Arg(m.a), gpu.Arg(dst),
			gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K))); err != nil {
			return err
		}
	}
	if bias.Len() == 0 {
		return nil
	}
	return e.q.Launch(e.k.AddBias, gpu.Grid1D(M*N, 256),
		gpu.Arg(dst), gpu.Arg(bias), gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)))
}

// projBiasAdd runs a projection with fused bias and residual addition:
// residual[M,N] += src[M,K] · w[N,K]ᵀ + bias.
func (e *encoder) projBiasAdd(src gpu.Buffer, m mat, bias, residual gpu.Buffer, M int) error {
	K, N := m.cols, m.rows
	if m.quant && bias.Len() > 0 {
		gp, gc := e.k.GEMMW8A8BiasAddPlan(M, N, K)
		if gp != (gpu.Pipeline{}) {
			if err := e.q.Launch(e.k.QuantRows, gpu.RowGrid(M),
				gpu.Arg(src), gpu.Arg(e.qi8), gpu.Arg(e.qs),
				gpu.ArgValue(int32(M)), gpu.ArgValue(int32(K))); err != nil {
				return err
			}
			return e.q.Launch(gp, gc,
				gpu.Arg(e.qi8), gpu.Arg(e.qs), gpu.Arg(m.a), gpu.Arg(m.b), gpu.Arg(bias), gpu.Arg(residual),
				gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K)))
		}
	} else if !m.quant && bias.Len() > 0 {
		gp, gcfg := e.k.GEMMF32BiasAddPlan(M, N, K)
		if gp != (gpu.Pipeline{}) {
			return e.q.Launch(gp, gcfg,
				gpu.Arg(src), gpu.Arg(m.a), gpu.Arg(bias), gpu.Arg(residual),
				gpu.ArgValue(int32(M)), gpu.ArgValue(int32(N)), gpu.ArgValue(int32(K)))
		}
	}
	if err := e.proj(src, m, bias, e.projOut, M); err != nil {
		return err
	}
	return e.q.Launch(e.k.AddVec, gpu.Grid1D(M*N, 256),
		gpu.Arg(residual), gpu.Arg(e.projOut), gpu.ArgValue(int32(M*N)))
}

func (e *encoder) rms(src, w, dst gpu.Buffer, rows, dim int) error {
	return e.q.Launch(e.k.RMSNorm, gpu.RowGrid(rows),
		gpu.Arg(src), gpu.Arg(w), gpu.Arg(dst),
		gpu.ArgValue(int32(rows)), gpu.ArgValue(int32(dim)), gpu.ArgValue(float32(1e-6)))
}

// ForwardViT runs the ViT blocks on the device and returns the pre-merge hidden state
// in ORIGINAL patch order, matching the CPU contract.
func (e *encoder) ForwardViT(pixelValues []float32, gridTHW [][3]int) (out []float32, err error) {
	plan, err := e.src.BuildWindowPlan(gridTHW)
	if err != nil {
		return nil, err
	}
	n := plan.NPatches
	H, I, nH, hd := e.w.Hidden, e.w.Inter, e.w.NumHeads, e.w.HeadDim
	pd := e.w.PatchDim
	if len(pixelValues) != n*pd {
		return nil, fmt.Errorf("qwencuda: pixel_values len %d, want %d", len(pixelValues), n*pd)
	}
	merge := e.w.SpatialMergeSize
	mergeUnit := merge * merge
	groups := n / mergeUnit

	e.mu.Lock()
	defer e.mu.Unlock()
	defer func() {
		// A panic anywhere below (a device allocation failure is a panic by gpu's contract) comes back as an error, and a failed forward is drained before
		// returning: kernels still queued against the scratch must not outlive the call that owns it, or the next growth would free buffers they are reading.
		if r := recover(); r != nil {
			out, err = nil, fmt.Errorf("qwencuda: forward failed on the device: %v", r)
		}
		if err != nil {
			if serr := e.q.Sync(); serr != nil {
				err = fmt.Errorf("%w (and draining the queue: %v)", err, serr)
			}
		}
	}()
	if err := e.ensure(n); err != nil {
		return nil, err
	}

	// Permute pixel rows into WINDOW order on the host — the patch embed is row-wise,
	// so this makes the device's hidden state window-ordered without a gather kernel.
	pw := make([]float32, n*pd)
	for g := range groups {
		src := plan.WinIdx[g]
		for u := range mergeUnit {
			dp, sp := (g*mergeUnit+u)*pd, (src*mergeUnit+u)*pd
			copy(pw[dp:dp+pd], pixelValues[sp:sp+pd])
		}
	}
	if err := gpu.Upload(e.pix, pw); err != nil {
		return nil, err
	}
	if err := gpu.Upload(e.cosB, plan.Cos); err != nil {
		return nil, err
	}
	if err := gpu.Upload(e.sinB, plan.Sin); err != nil {
		return nil, err
	}

	// patch embed: h = pixels[n,pd] · patchW[H,pd]ᵀ (no bias).
	pep, pecfg := e.k.GEMMF32Plan(n, H, pd)
	if err := e.q.Launch(pep, pecfg,
		gpu.Arg(e.pix), gpu.Arg(e.patchW), gpu.Arg(e.h),
		gpu.ArgValue(int32(n)), gpu.ArgValue(int32(H)), gpu.ArgValue(int32(pd))); err != nil {
		return nil, err
	}

	scale := float32(1.0 / math.Sqrt(float64(hd)))
	// Both bound pairs go up ONCE, before the first launch (audit C-01).
	//
	// This used to upload inside the layer loop whenever the block kind
	// switched. gpu.Upload copies on the LEGACY NULL STREAM and then
	// Synchronizes, which orders the copy before LATER launches — the RAW race
	// it was written for — but nothing orders it after EARLIER ones. The queue's
	// stream is CU_STREAM_NON_BLOCKING, which by definition does not order
	// against the null stream. So at the first full-attention block (layer 7 in
	// Qwen2.5-VL) the CPU issued the full-image bounds while layers 0-6 were
	// still queued; the DMA could land under them, and their attention_seg would
	// then read n = s1-s0 up to the whole image with shared memory sized for
	// MaxWinSeg — an out-of-bounds shared write, i.e. garbage or a
	// context-killing fault.
	//
	// Uploading both pairs up front removes the hazard by construction rather
	// than by adding another sync: after this point nothing writes a buffer any
	// queued launch reads. It also restores the "ONE Sync per forward" property
	// the file's header claims — the old form made 14 full-device syncs.
	//
	// NOTE the tiny fixture cannot catch a regression here: each layer's GPU
	// work is shorter than its own enqueue, so the queue never builds and the
	// race never opens. The parity tests passing is necessary, not sufficient.
	if err := gpu.Upload(e.segSWin, plan.WinStart); err != nil {
		return nil, err
	}
	if err := gpu.Upload(e.segEWin, plan.WinEnd); err != nil {
		return nil, err
	}
	if err := gpu.Upload(e.segSFull, plan.FullStart); err != nil {
		return nil, err
	}
	if err := gpu.Upload(e.segEFull, plan.FullEnd); err != nil {
		return nil, err
	}
	for li := range e.blocks {
		B := &e.blocks[li]
		full := e.src.IsFullAtt(li)
		segS, segE := e.segSWin, e.segEWin
		maxSeg := plan.MaxWinSeg
		if full {
			segS, segE = e.segSFull, e.segEFull
			maxSeg = plan.MaxFullSeg
		}

		// --- attention block ---
		if err := e.rms(e.h, B.norm1w, e.n1, n, H); err != nil {
			return nil, err
		}
		if err := e.proj(e.n1, B.qkvw, B.qkvb, e.qkv, n); err != nil {
			return nil, err
		}
		if err := e.q.Launch(e.k.RopeQK, gpu.Grid1D(n*nH*(hd/2), 256),
			gpu.Arg(e.qkv), gpu.Arg(e.cosB), gpu.Arg(e.sinB),
			gpu.ArgValue(int32(n)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(hd))); err != nil {
			return nil, err
		}
		if gpu.AttentionTiledEligible(hd) && n >= gpu.AttentionTiledMinNP {
			if err := e.q.Launch(e.k.AttentionSegTiled, gpu.AttentionSegTiledLaunchConfig(n, nH),
				gpu.Arg(e.qkv), gpu.Arg(e.att), gpu.Arg(segS), gpu.Arg(segE),
				gpu.ArgValue(int32(n)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(scale)); err != nil {
				return nil, err
			}
		} else {
			if err := e.q.Launch(e.k.AttentionSeg, gpu.SegAttentionGrid(n, nH, maxSeg),
				gpu.Arg(e.qkv), gpu.Arg(e.att), gpu.Arg(segS), gpu.Arg(segE),
				gpu.ArgValue(int32(n)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(scale)); err != nil {
				return nil, err
			}
		}
		if err := e.projBiasAdd(e.att, B.projw, B.projb, e.h, n); err != nil {
			return nil, err
		}
		// --- gated SiLU MLP ---
		if err := e.rms(e.h, B.norm2w, e.n2, n, H); err != nil {
			return nil, err
		}
		if err := e.proj(e.n2, B.gatew, B.gateb, e.gate, n); err != nil {
			return nil, err
		}
		if err := e.proj(e.n2, B.upw, B.upb, e.up, n); err != nil {
			return nil, err
		}
		if err := e.q.Launch(e.k.SiLUMul, gpu.Grid1D(n*I, 256),
			gpu.Arg(e.gate), gpu.Arg(e.up), gpu.ArgValue(int32(n*I))); err != nil {
			return nil, err
		}
		if err := e.projBiasAdd(e.gate, B.downw, B.downb, e.h, n); err != nil {
			return nil, err
		}
	}
	if err := e.q.Sync(); err != nil {
		return nil, err
	}
	hWin := make([]float32, n*H)
	if err := gpu.Download(e.h, hWin); err != nil {
		return nil, err
	}
	// de-window back to original patch order
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

func (e *encoder) Close() { e.dev.ReleaseObjects() }

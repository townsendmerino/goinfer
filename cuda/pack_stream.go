//go:build cuda

package cuda

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/decoder"
)

// A dense projection is planned while the build walks the layers and packed while it uploads. planWeight makes packWeight's accept-or-decline
// decision without packing, so a model that cannot be built is declined before a byte of device memory is taken; the hostW it returns names the weight
// and the sizes of the three device buffers. upW then allocates those buffers and streams the packed bytes into them in row blocks through one
// reused staging block, so the host holds one block of packed data, not a packed copy of the model.
// Evidence and gates: docs/measurements/cuda-stream-pack-2026-10-18/README.md.

// planCompleteForTest, when set (tests only), is called once BuildResident has accepted every layer and the LM head; between planStartForTest and this call
// the build must allocate no device buffer.
var planCompleteForTest func()

// planStartForTest, when set (tests only), is called as BuildResident begins planning its layers.
var planStartForTest func()

// stagePackWords is the staging block, in uint32 words (8 MiB). A test sets it small to force several blocks.
var stagePackWords = 2 << 20

// planWeight is packWeight's validation and nothing else: the same error for the same weight, no allocation beyond a few slice headers, and a hostW
// with the weight itself (src) and the element counts of the buffers upW will allocate.
func planWeight(w *linalg.WeightMat) (hostW, error) {
	N, K := w.Rows(), w.Cols()
	if K%4 != 0 {
		return hostW{}, fmt.Errorf("cuda: K=%d not a multiple of 4", K)
	}
	switch w.Kind() {
	case "int4":
		if K%32 != 0 {
			return hostW{}, fmt.Errorf("cuda: int4 K=%d not a multiple of 32", K)
		}
		// Kind() reports precision, not layout, and stays "int4" for a repacked-only tensor, so this switch alone cannot
		// tell canonical from repacked-only; without the ok, a repacked-only tensor's nil q4 would index out of range
		// below instead of declining cleanly (decoder/weightmat.go's wantsCanonicalInt4 doc has the policy). Repacked-only
		// activates only for Options.Backend=="cpu", never "cuda", so this is defense in depth, not a path expected to
		// fire (docs/code-notes/cuda.md#packWeight.int4).
		_, s16, _, ok := w.Int4F16()
		if !ok {
			return hostW{}, fmt.Errorf("cuda: int4 tensor has no canonical bytes (layout %s-only): "+
				"model was loaded for a CPU-only backend; load with Options.Backend set to "+
				"\"cuda\" to keep canonical bytes for residency", w.Int4Layout())
		}
		return hostW{kind: "int4", N: N, K: K, src: w, nWpk: N * (K / 8), nWs16: len(s16)}, nil
	case "int8":
		_, sc, _, _ := w.Int8()
		return hostW{kind: "int8", N: N, K: K, src: w, nWpk: N * (K / 4), nWs: len(sc)}, nil
	case "q4k":
		// GGUF Q4_K super-blocks verbatim (--quant q4k), as little-endian words: 36 per 256 weights.
		raw, _ := w.Q4K()
		if K%256 != 0 || len(raw) != N*K/256*144 {
			return hostW{}, fmt.Errorf("cuda: q4k tensor %dx%d with %d bytes", N, K, len(raw))
		}
		return hostW{kind: "q4k", N: N, K: K, src: w, nWpk: len(raw) / 4}, nil
	case "f32":
		return hostW{kind: "int8", N: N, K: K, src: w, nWpk: N * (K / 4), nWs: N}, nil
	default:
		return hostW{}, fmt.Errorf("cuda: unsupported projection kind %q", w.Kind())
	}
}

// packStage is one staging block of the build: scratch for the packed words and scales of a run of rows, and the slices the upload reads. The build keeps a
// small pool of them (packPool), reused for every tensor and dropped when the build's setup job ends (cudaResident.pk).
type packStage struct {
	u32 []uint32
	u16 []uint16
	f32 []float32
	// what pack produced for upload: the packed words, and the block's scales (f16 for int4, f32 for a quantized f32 weight)
	outW []uint32
	outH []uint16
	outS []float32
}

func (s *packStage) words(n int) []uint32 {
	if cap(s.u32) < n {
		s.u32 = make([]uint32, max(n, stagePackWords))
	}
	return s.u32[:n]
}
func (s *packStage) halves(n int) []uint16 {
	if cap(s.u16) < n {
		s.u16 = make([]uint16, max(n, stagePackWords/4))
	}
	return s.u16[:n]
}
func (s *packStage) floats(n int) []float32 {
	if cap(s.f32) < n {
		s.f32 = make([]float32, max(n, stagePackWords/4))
	}
	return s.f32[:n]
}

// packPool is the staging blocks and the channel that hands them to packers: a block is free until a packer takes it and again once its upload is done,
// so the host never holds more than len(stages) blocks of packed data.
type packPool struct {
	stages []*packStage
	free   chan *packStage
}

// packStageCount is how many staging blocks the build keeps: packing reads the (mapped) source at a few ns per word on one core, so several blocks are packed
// on other cores while the executor thread uploads. A test sets it to pin the host bound.
var packStageCount = 0

func stageCount() int {
	if packStageCount > 0 {
		return packStageCount
	}
	return max(1, min(4, runtime.GOMAXPROCS(0)/2))
}

func newPackPool() *packPool {
	n := stageCount()
	p := &packPool{free: make(chan *packStage, n)}
	for range n {
		st := &packStage{}
		p.stages = append(p.stages, st)
		p.free <- st
	}
	return p
}

// streamBlocks packs the N rows of a tensor in blocks of `block` rows and calls up on each packed block. pack runs on goroutines (at most one per free staging
// block), up runs on the calling goroutine, which owns the device context, in whatever order the blocks finish: every block names its own offset.
func (r *cudaResident) streamBlocks(N, block int, pack func(st *packStage, r0, rows int), up func(st *packStage, r0, rows int)) {
	if r.pk == nil {
		r.pk = newPackPool()
	}
	nb := (N + block - 1) / block
	type done struct {
		st       *packStage
		r0, rows int
	}
	ready := make(chan done, len(r.pk.stages))
	go func() {
		for b := 0; b < nb; b++ {
			if r.loadAborted() { // the tripwire fired: pack nothing more, but tell the consumer every remaining block is skipped
				ready <- done{}
				continue
			}
			st := <-r.pk.free
			r0 := b * block
			rows := min(block, N-r0)
			go func() {
				pack(st, r0, rows)
				ready <- done{st, r0, rows}
			}()
		}
	}()
	for range nb {
		d := <-ready
		if d.st == nil || r.loadAborted() {
			r.recordUpload(fmt.Errorf("cuda: resident build aborted while uploading a weight: %w", decoder.ErrLoadAborted))
			if d.st == nil {
				continue
			}
		} else {
			up(d.st, d.r0, d.rows)
		}
		r.pk.free <- d.st
	}
}

// streamW is upW for a planned weight: it allocates the weight's device buffers (the same sizes upW allocates for a packed one), then packs row blocks into
// staging blocks (streamBlocks) and uploads each at its offset. Every block holds whole rows, and each row's packed words and scales depend on that row
// alone, so the device bytes are the bytes packWeight produces. An upload error is recorded as upW's are (recordUpload).
func (r *cudaResident) streamW(h hostW) cudaWQ {
	w := cudaWQ{kind: h.kind, N: h.N, K: h.K, W: r.au32(h.nWpk)}
	N, K := h.N, h.K
	rowsFor := func(wordsPerRow int) int { return max(1, stagePackWords/max(wordsPerRow, 1)) }
	putWords := func(st *packStage, r0, wpr int) { r.recordUpload(gpu.Upload(w.W.At(r0*wpr*4), st.outW)) }
	switch h.kind {
	case "int4":
		q4, s16, _, ok := h.src.Int4F16()
		if !ok {
			r.recordUpload(fmt.Errorf("cuda: int4 tensor lost its canonical bytes between planning and upload"))
			return w
		}
		allocSeen(h.nWs16 * 2)
		w.ws16 = gpu.NewBufferLenOf[uint16](r.dev, h.nWs16)
		wpr, gpr := K/8, K/32
		r.streamBlocks(N, rowsFor(wpr), func(st *packStage, r0, rows int) {
			words := st.words(rows * wpr)
			src := q4[r0*wpr*4 : (r0+rows)*wpr*4]
			for i := range words {
				b := src[i*4 : i*4+4]
				words[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
			}
			st.outW = words
			// The scales go through the same f16 -> f32 -> f16 trip packWeight makes, so a non-canonical payload comes out the same.
			n := rows * gpr
			wide := st.floats(n)
			linalg.F16ToF32Slice(wide, s16[r0*gpr:r0*gpr+n])
			gs := st.halves(n)
			for i, v := range wide {
				gs[i] = f32tof16(v)
			}
			st.outH = gs
		}, func(st *packStage, r0, rows int) {
			putWords(st, r0, wpr)
			r.recordUpload(gpu.Upload(w.ws16.At(r0*gpr*2), st.outH))
		})
	case "int8":
		wpr := K / 4
		if h.src.Kind() == "int8" {
			q8, sc, _, _ := h.src.Int8()
			r.streamBlocks(N, rowsFor(wpr), func(st *packStage, r0, rows int) {
				words := st.words(rows * wpr)
				packI8Into(words, q8[r0*K:(r0+rows)*K])
				st.outW = words
			}, func(st *packStage, r0, rows int) { putWords(st, r0, wpr) })
			w.ws = r.up32(sc)
			return w
		}
		// An f32 weight, quantized to int8 row by row here as packWeight does.
		f32, _ := h.src.F32()
		w.ws = r.af(h.nWs)
		r.streamBlocks(N, rowsFor(wpr), func(st *packStage, r0, rows int) {
			q8, sc := linalg.QuantizeRowsInt8(f32[r0*K:(r0+rows)*K], rows, K)
			words := st.words(rows * wpr)
			packI8Into(words, q8)
			st.outW, st.outS = words, sc
		}, func(st *packStage, r0, rows int) {
			putWords(st, r0, wpr)
			r.recordUpload(gpu.Upload(w.ws.At(r0*4), st.outS))
		})
	case "q4k":
		raw, _ := h.src.Q4K()
		wpr := K / 256 * 36
		r.streamBlocks(N, rowsFor(wpr), func(st *packStage, r0, rows int) {
			words := st.words(rows * wpr)
			src := raw[r0*wpr*4 : (r0+rows)*wpr*4]
			for i := range words {
				words[i] = uint32(src[4*i]) | uint32(src[4*i+1])<<8 | uint32(src[4*i+2])<<16 | uint32(src[4*i+3])<<24
			}
			st.outW = words
		}, func(st *packStage, r0, rows int) { putWords(st, r0, wpr) })
	default:
		r.recordUpload(fmt.Errorf("cuda: unsupported planned kind %q", h.kind))
	}
	return w
}

// packI8Into is packI8 into a caller's slice: four int8 per word, little-endian.
func packI8Into(dst []uint32, q8 []int8) {
	for i := range dst {
		dst[i] = uint32(uint8(q8[i*4])) | uint32(uint8(q8[i*4+1]))<<8 | uint32(uint8(q8[i*4+2]))<<16 | uint32(uint8(q8[i*4+3]))<<24
	}
}

var _ = decoder.Int4F32 // packWeight still widens whole tensors through it

// planWeightStack is packWeightStack's validation and nothing else: the same errors in the same order for the same weights, no allocation beyond the
// returned header, and a hostW that names the member weights (srcs) and the element counts of the stack's two host buffers. Only the C' expert cache
// consumes it (cacheWQ packs the members straight into the pinned copy); every other path takes packWeightStack's packed slices.
func planWeightStack(ws ...*linalg.WeightMat) (hostW, error) {
	if len(ws) == 0 {
		return hostW{}, fmt.Errorf("cuda: packWeightStack needs at least one weight")
	}
	var out hostW
	for i, w := range ws {
		h, err := planWeight(w)
		if err != nil {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: %w", i, err)
		}
		if i == 0 {
			out.kind, out.K = h.kind, h.K
		}
		if h.kind != out.kind {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: kind %q != %q — a mixed-precision "+
				"stack cannot share one kernel's unpack path", i, h.kind, out.kind)
		}
		if h.K != out.K {
			return hostW{}, fmt.Errorf("cuda: packWeightStack[%d]: K=%d != %d — the kernel strides by a "+
				"single Kwords, so a ragged stack reads across row boundaries", i, h.K, out.K)
		}
		out.nWpk += h.nWpk
		out.nWs += h.nWs
		out.nWs16 += h.nWs16
		out.N += h.N
	}
	out.srcs = ws
	return out, nil
}

// packInt4Into is packWeight's int4 layout written into caller-owned storage: the nibble-permuted words, and the group scales through the f16 -> f32 -> f16
// trip packWeight makes. It allocates nothing.
func packInt4Into(dstW []uint32, dstS []uint16, q4 []byte, s16 []uint16) {
	for i := range dstW {
		b := q4[i*4 : i*4+4]
		dstW[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	}
	for i, h := range s16 {
		dstS[i] = f32tof16(linalg.F16ToF32(h))
	}
}

// packStackWorkers bounds the goroutines that pack one expert stack: the sources are mapped files, so more than a few readers only adds page-fault traffic.
const packStackWorkers = 8

// packStackPinned allocates the pinned host copy of a planned int4 stack (outside the Go heap) and packs every member weight into its row range of it, in
// parallel, so the host holds the pinned copy and nothing else: the stack is never built in Go memory first. Each member's bytes are the bytes
// packWeight produces for it, at the offset packWeightStack puts it. An abort, a lost canonical payload or a malformed member is recorded as an upload
// error and leaves the copy partly filled, which the failed build discards.
func (r *cudaResident) packStackPinned(w *cudaWQ, h hostW) {
	srcW, err := r.dev.NewMappedHostBuffer(h.nWpk * 4)
	if err != nil {
		panic(fmt.Sprintf("cacheWQ: NewMappedHostBuffer(%d): %v", h.nWpk*4, err))
	}
	w.srcW = srcW
	srcS, err := r.dev.NewMappedHostBuffer(h.nWs16 * 2)
	if err != nil {
		panic(fmt.Sprintf("cacheWQ: NewMappedHostBuffer(%d): %v", h.nWs16*2, err))
	}
	w.srcS = srcS
	var dstW []uint32
	var dstS []uint16
	if h.nWpk > 0 {
		dstW = unsafe.Slice((*uint32)(unsafe.Pointer(&srcW.Bytes()[0])), h.nWpk)
	}
	if h.nWs16 > 0 {
		dstS = unsafe.Slice((*uint16)(unsafe.Pointer(&srcS.Bytes()[0])), h.nWs16)
	}
	n := len(h.srcs)
	rowOff := make([]int, n+1)
	for i, s := range h.srcs {
		rowOff[i+1] = rowOff[i] + s.Rows()
	}
	wpr, gpr := h.K/8, h.K/32
	var next atomic.Int64
	var aborted, bad atomic.Bool
	var wg sync.WaitGroup
	for range min(packStackWorkers, runtime.GOMAXPROCS(0), n) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= n {
					return
				}
				if r.loadAborted() {
					aborted.Store(true)
					return
				}
				q4, s16, _, ok := h.srcs[i].Int4F16()
				dw, ds := dstW[rowOff[i]*wpr:rowOff[i+1]*wpr], dstS[rowOff[i]*gpr:rowOff[i+1]*gpr]
				if !ok || len(q4) < len(dw)*4 || len(s16) != len(ds) {
					bad.Store(true)
					return
				}
				packInt4Into(dw, ds, q4, s16)
			}
		}()
	}
	wg.Wait()
	switch {
	case aborted.Load():
		r.recordUpload(fmt.Errorf("cuda: resident build aborted while packing an expert stack: %w", decoder.ErrLoadAborted))
	case bad.Load():
		r.recordUpload(fmt.Errorf("cuda: an int4 expert lost its canonical bytes between planning and upload, or its payload is the wrong size"))
	}
}

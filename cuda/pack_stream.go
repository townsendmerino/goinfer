//go:build cuda

package cuda

import (
	"fmt"

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

// packStage is the build's staging memory: one block of packed words and one of scales, reused for every tensor and dropped when the build's setup job
// ends (cudaResident.pk).
type packStage struct {
	u32 []uint32
	u16 []uint16
	f32 []float32
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

// streamW is upW for a planned weight: it allocates the weight's device buffers (the same sizes upW allocates for a packed one), then packs row blocks into
// the staging block and uploads each at its offset. Every block holds whole rows, and each row's packed words and scales depend on that row alone, so
// the device bytes are the bytes packWeight produces. An upload error is recorded as upW's are (recordUpload).
func (r *cudaResident) streamW(h hostW) cudaWQ {
	if r.pk == nil {
		r.pk = &packStage{}
	}
	w := cudaWQ{kind: h.kind, N: h.N, K: h.K, W: r.au32(h.nWpk)}
	N, K := h.N, h.K
	rowsFor := func(wordsPerRow int) int { return max(1, stagePackWords/max(wordsPerRow, 1)) }
	putWords := func(dst Buffer, off int, src []uint32) { r.recordUpload(gpu.Upload(dst.At(off*4), src)) }
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
		block := rowsFor(wpr)
		for r0 := 0; r0 < N; r0 += block {
			rows := min(block, N-r0)
			words := r.pk.words(rows * wpr)
			src := q4[r0*wpr*4 : (r0+rows)*wpr*4]
			for i := range words {
				b := src[i*4 : i*4+4]
				words[i] = permuteFast(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
			}
			putWords(w.W, r0*wpr, words)
			// The scales go through the same f16 -> f32 -> f16 trip packWeight makes, so a non-canonical payload comes out the same.
			n := rows * gpr
			wide := r.pk.floats(n)
			linalg.F16ToF32Slice(wide, s16[r0*gpr:r0*gpr+n])
			gs := r.pk.halves(n)
			for i, v := range wide {
				gs[i] = f32tof16(v)
			}
			r.recordUpload(gpu.Upload(w.ws16.At(r0*gpr*2), gs))
		}
	case "int8":
		if h.src.Kind() == "int8" {
			q8, sc, _, _ := h.src.Int8()
			wpr := K / 4
			block := rowsFor(wpr)
			for r0 := 0; r0 < N; r0 += block {
				rows := min(block, N-r0)
				words := r.pk.words(rows * wpr)
				packI8Into(words, q8[r0*K:(r0+rows)*K])
				putWords(w.W, r0*wpr, words)
			}
			w.ws = r.up32(sc)
			return w
		}
		// An f32 weight, quantized to int8 row by row here as packWeight does.
		f32, _ := h.src.F32()
		w.ws = r.af(h.nWs)
		wpr := K / 4
		block := rowsFor(wpr)
		for r0 := 0; r0 < N; r0 += block {
			rows := min(block, N-r0)
			q8, sc := linalg.QuantizeRowsInt8(f32[r0*K:(r0+rows)*K], rows, K)
			words := r.pk.words(rows * wpr)
			packI8Into(words, q8)
			putWords(w.W, r0*wpr, words)
			r.recordUpload(gpu.Upload(w.ws.At(r0*4), sc))
		}
	case "q4k":
		raw, _ := h.src.Q4K()
		wpr := K / 256 * 36
		block := rowsFor(wpr)
		for r0 := 0; r0 < N; r0 += block {
			rows := min(block, N-r0)
			words := r.pk.words(rows * wpr)
			src := raw[r0*wpr*4 : (r0+rows)*wpr*4]
			for i := range words {
				words[i] = uint32(src[4*i]) | uint32(src[4*i+1])<<8 | uint32(src[4*i+2])<<16 | uint32(src[4*i+3])<<24
			}
			putWords(w.W, r0*wpr, words)
		}
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

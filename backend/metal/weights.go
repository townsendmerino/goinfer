//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

func byteBufOn(line string, d *Device, n int) Buffer {
	return d.NewBufferBytesOn(line, n)
}

// byteBuf is byteBufOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func byteBuf(d *Device, n int) Buffer {
	return byteBufOn(gpu.UnlinedLine, d, n)
}

// q4kSeg is one run of a q4k-lane projection's output rows that share a weight kind, written at row off of the projection's
// output (uN holds rows): Q4_K or Q5_K super-blocks as the GGUF stores them, Q6_K split into planes over the segment
// (q6kPlanarRow; s unused for all three), or int8 rows with per-row scales s.
type q4kSeg struct {
	kind      string // "q4k", "q5k", "q6k" or "int8"
	w, s, uN  Buffer
	rows, off int
}

// q4kNativeOffForTest keeps a q4k resident on the int8 copies of its Q5_K and Q6_K tensors instead of the native GEMVs; read at
// build.
var q4kNativeOffForTest bool

// q4kPartKind is the kind a q4k-lane part uploads as: its own kind for Q4_K, the file's Q5_K or Q6_K when kq serves it (the
// part is then an int8 copy of it), else int8.
func q4kPartKind(kq *decoder.KQuantReader, w *linalg.WeightMat) string {
	if w.Kind() == "q4k" {
		return "q4k"
	}
	if k := kq.Kind(w); k != "" && !q4kNativeOffForTest {
		return k
	}
	return "int8"
}

// q4kLaneProj uploads a projection, or a fused group with its parts' rows in order, as runs of consecutive parts of one
// kind: a GGUF q4_k_m file mixes them inside a group (Qwen2.5's attn_v is Q6_K in half the layers, beside Q4_K
// attn_q/attn_k). kq serves the file's Q5_K and Q6_K blocks; nil keeps those parts on their int8 copies.
func q4kLaneProjOn(line string, d *Device, kq *decoder.KQuantReader, wms ...*linalg.WeightMat) []q4kSeg {
	var segs []q4kSeg
	off := 0
	for i := 0; i < len(wms); {
		j, kind := i+1, q4kPartKind(kq, wms[i])
		for j < len(wms) && q4kPartKind(kq, wms[j]) == kind {
			j++
		}
		rows := 0
		for _, w := range wms[i:j] {
			rows += w.Rows()
		}
		w, s := q4kLaneBufOn(line, d, kq, kind, wms[i:j]...)
		segs = append(segs, q4kSeg{kind: kind, w: w, s: s, rows: rows, off: off}) // uN: the decode list's, once the layer holds the slice
		off += rows
		i = j
	}
	return segs
}

// q4kLaneBuf uploads parts of one kind for the q4k lane: GGUF Q4_K super-blocks
// verbatim; the file's Q5_K blocks verbatim and its Q6_K blocks in planes over the segment (q6kPlanarRow), read through kq;
// else int8 rows with their per-row scales (the scale buffer is unused for the block kinds). A part of another kind panics;
// buildResident recovers that into a decline. Each part is copied straight into the device buffer at its offset, with no
// host-side concatenation: a staging slice per projection is garbage the build does not collect, and on a 16 GB Mac it
// pushed phi3's load into swap.
func q4kLaneBufOn(line string, d *Device, kq *decoder.KQuantReader, kind string, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	switch kind {
	case "q4k":
		n := 0
		for _, w := range wms {
			raw, ok := w.Q4K()
			if !ok {
				panic(fmt.Sprintf("metal: q4k lane: a fused group mixes Q4_K with %q", w.Kind()))
			}
			n += len(raw)
		}
		buf := gpu.NewBufferLenOfOn[int8](d, line, n)
		off := 0
		for _, w := range wms {
			raw, _ := w.Q4K()
			mustUpload(buf.At(off), raw)
			off += len(raw)
		}
		return buf, Buffer{}
	case "q5k", "q6k":
		bb := map[string]int{"q5k": 176, "q6k": 210}[kind]
		n := 0
		for _, w := range wms {
			n += w.Rows() * (w.Cols() / 256) * bb
		}
		buf := gpu.NewBufferLenOfOn[int8](d, line, n)
		iv := buf.Int8s() // the shared contents, fetched once: Upload per row costs an objc call and heap each time
		dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(iv))), len(iv))
		row0 := 0 // the part's first row in the segment
		for _, w := range wms {
			rb := w.Cols() / 256 * bb
			err := kq.Rows(w, func(r int, raw []byte) error {
				if kind == "q6k" {
					q6kPlanarRow(dst, n/210, (row0+r)*(w.Cols()/256), raw)
				} else {
					copy(dst[(row0+r)*rb:], raw)
				}
				return nil
			})
			if err != nil {
				panic(err)
			}
			row0 += w.Rows()
		}
		return buf, Buffer{}
	}
	nq, ns := 0, 0
	for _, w := range wms {
		q8, s8, _, ok := w.Int8()
		if !ok {
			panic(fmt.Sprintf("metal: q4k lane: weight kind %q is neither q4k nor int8", w.Kind()))
		}
		nq, ns = nq+len(q8), ns+len(s8)
	}
	q, sc := gpu.NewBufferLenOfOn[int8](d, line, nq), gpu.NewBufferLenOfOn[float32](d, line, ns)
	offQ, offS := 0, 0
	for _, w := range wms {
		q8, s8, _, _ := w.Int8()
		mustUpload(q.At(offQ), q8)
		mustUpload(sc.At(4*offS), s8)
		offQ, offS = offQ+len(q8), offS+len(s8)
	}
	return q, sc
}

// q4kLaneBuf is q4kLaneBufOn on gpu.UnlinedLine: tests call it; production code names its plan line.
func q4kLaneBuf(d *Device, kq *decoder.KQuantReader, kind string, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	return q4kLaneBufOn(gpu.UnlinedLine, d, kq, kind, wms...)
}

// q6kPlanarRow writes one row of GGUF Q6_K blocks (ql[128], qh[64], scales[16], d; 210 bytes each, not word-aligned) into a
// segment of T blocks laid out as the q4k lane's Q6_K kernels read it: T*128 bytes of ql, T*64 of qh, T*16 of scales, then T
// halves of d. g0 is the row's first block in the segment (row * blocks a row).
func q6kPlanarRow(dst []byte, T, g0 int, src []byte) {
	for b := range len(src) / 210 {
		blk, g := src[b*210:(b+1)*210], g0+b
		copy(dst[g*128:], blk[:128])
		copy(dst[T*128+g*64:], blk[128:192])
		copy(dst[T*192+g*16:], blk[192:208])
		copy(dst[T*208+g*2:], blk[208:210])
	}
}

// mustUpload copies src into b at b's offset, panicking on an overrun as a short buffer is a build bug (buildResident
// recovers a panic into a decline).
func mustUpload[T gpu.Scalar](b Buffer, src []T) {
	if err := gpu.Upload(b, src); err != nil {
		panic(err)
	}
}

func int8BufOn(line string, d *Device, w *linalg.WeightMat) (Buffer, Buffer, error) {
	q8, sc, _, ok := w.Int8()
	if !ok {
		return Buffer{}, Buffer{}, fmt.Errorf("metal: weight kind %q is not int8 (load Options{Quant:\"int8int8\"})", w.Kind())
	}
	return NewBufferInt8On(line, d, q8), NewBufferFloatsOn(line, d, sc), nil
}

// int4DirectWords converts a decoder int4 WeightMat's packed nibbles and f32 group scales straight into Metal's W4A8
// buffers (uint32 words plus f16 scales), with no int8 intermediate. aikit's group=32 packing (nib = q+8, byte k/2
// low/high) and Metal's packW4A8Row (element k to word k/8, bit 4*(k%8)) are the same bytes on little-endian, so the
// nibbles copy verbatim; only the group scales narrow f32 to f16. Returns ok=false if the weight is not group-32 int4.
// Consuming the int4 directly avoids the double quantization (f32 to int8 to int4) of BuildResident's default path,
// which Gemma's low-magnitude attention contexts amplify into a large context error (metal/gemma_sublayer_test.go).
func int4DirectWords(w *linalg.WeightMat) (words []uint32, scales []uint16, ok bool) {
	q4, q4s, group, ok := decoder.Int4F32(w)
	if !ok || group != 32 {
		return nil, nil, false
	}
	words = bytesToU32(q4)
	scales = make([]uint16, len(q4s))
	for i, s := range q4s {
		scales[i] = f32ToF16(s)
	}
	return words, scales, true
}

// bytesToU32 reinterprets a little-endian byte slice as uint32 words (len must be a multiple of 4) with one bulk copy:
// b already holds the target word bytes, and w is freshly allocated, so it is 4-aligned, unlike the mmap-backed
// sources int4DirectBytesOnly exists to avoid reinterpreting.
func bytesToU32(b []byte) []uint32 {
	w := make([]uint32, len(b)/4)
	if len(w) == 0 {
		return w
	}
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), len(w)*4), b)
	return w
}

// int4DirectBytesOnly returns a canonical group-32 int4 WeightMat's packed nibble bytes aliased straight from the
// mmap: no bytesToU32 reconstruction and no per-stage []uint32 allocation. They are byte for byte the words
// int4DirectWords builds (little-endian), so a byte copy into a uint32 slot buffer reproduces them exactly
// (copyBytesToU32Buf). The paged MoE stage functions (buildMoELayer, buildGemma4MoELayer) pair it with the WeightMat's
// own Int4ScalesF16, rather than re-deriving f16 scales from an f32 copy on every page-in or caching a duplicate of
// what the mapping already holds.
func int4DirectBytesOnly(w *linalg.WeightMat) (q4 []byte, ok bool) {
	b, _, group, ok := w.Int4F16()
	if !ok || group != 32 {
		return nil, false
	}
	return b, true
}

// parallelF32ToF16 converts src (f32 group scales) to dst (f16 bits) across up to 8 workers. Every element is
// independent and f32ToF16 is deterministic, so the split is byte-identical to the serial loop. It exists for the
// expert-paging stage path, where the conversion runs once per expert per stage and is arithmetic-dominated; small
// inputs (the one-time non-paged build) stay serial, where goroutine spawn would not pay.
func parallelF32ToF16(dst []uint16, src []float32) {
	n := len(src)
	workers := min(runtime.GOMAXPROCS(0), 8)
	if n < 8192 || workers <= 1 {
		for i, s := range src {
			dst[i] = f32ToF16(s)
		}
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Go(func() {
			for i := lo; i < hi; i++ {
				dst[i] = f32ToF16(src[i])
			}
		})
	}
	wg.Wait()
}

// maxThreadgroupStageBytes returns the largest threadgroup staging allocation (bytes) any resident dispatch will
// request. The SA-GEMV and MoE kernels stage the contraction row into threadgroup memory at 2 bytes per element
// (DispatchTG tgBytes = 2*K); the widest staged K is hidden (qkv, gate-up), qWidth = nH*hd (o-proj), the MoE expert
// intermediate moeInter or g4moeInter (the expert down-proj stages inter), or dnValueDim (DeltaNet's out-projection,
// which deltanet.go dispatches with dp.valueDim*2 bytes). The dense down-proj uses the non-staging pGemv, so the dense
// intermediate is deliberately not counted. A separate function so the budget arithmetic is unit-testable.
func maxThreadgroupStageBytes(hidden, qWidth, moeInter, g4moeInter, dnValueDim int) int {
	return 2 * max(max(max(hidden, qWidth), max(moeInter, g4moeInter)), dnValueDim)
}

// nativeInt8 turns the native int8 path on (w8Eligible). Tests turn it off for the int4 re-quant arm; it is not an
// option or an environment variable.
var nativeInt8 = true

// nativeInt4Mix turns on int4mix's native path (w8AttnEligible: attention W8A8, FFN int4). Tests turn it off for the
// re-quant arm.
var nativeInt4Mix = true

// nativeInt8MoE admits a generic resident MoE to the native int8 path (w8Eligible; int8 expert GEMVs gemv_w8a8_moe*);
// tests turn it off.
var nativeInt8MoE = true

// w8FastMath keeps fast math for a native int8 model (w8PreciseMath off): test-only, gate S's fast-math arm, which
// prices the owner's precise-math decision.
var w8FastMath = false

// w8Eligible reports whether m runs on the native int8 path (r.w8): every dense body projection is int8-kind
// (int8int8, or weight-only int8, which CUDA and WebGPU also run against int8 activations) with K a multiple of 4,
// and the model is one slice 1 covers: no MoE, no Gemma-4 MoE, no DeltaNet
// (whose families also carry the gated-softmax layers). Anything else keeps the int4 re-quantization. A K=V layer
// has no v_proj and fails the check, so it is never fused as int8.
func w8Eligible(m *decoder.Model, r *resident) bool {
	if !nativeInt8 || r.g4moe != nil || r.recurrentModel() {
		return false
	}
	// slice 4: a generic resident MoE (Qwen-style experts, not gpt-oss's biased ones, not paged: the pool stages int4)
	if r.moe != nil && (!nativeInt8MoE || r.moe.isGptOss || r.moe.paged) {
		return false
	}
	return w8Weights(m)
}

// w8Weights is w8Eligible's weight half, which needs only the model: every dense body projection int8-kind with K a
// multiple of 4. An MoE or DeltaNet layer has empty dense projections and fails it.
func w8Weights(m *decoder.Model) bool {
	w := m.Weights()
	if len(w.Layers) == 0 {
		return false
	}
	shape, err := program.ShapeOf(m.Program())
	if err != nil {
		return false
	}
	nonGated := shape.NonGated
	for l := range w.Layers {
		lw := &w.Layers[l]
		mats := []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj}
		switch {
		case len(lw.Experts) > 0: // an MoE layer: its routed experts and shared expert carry the FFN (slice 4)
			for e := range lw.Experts {
				mats = append(mats, &lw.Experts[e].Gate, &lw.Experts[e].Up, &lw.Experts[e].Down)
			}
			if lw.SharedExpert.Up.Rows() > 0 {
				mats = append(mats, &lw.SharedExpert.Gate, &lw.SharedExpert.Up, &lw.SharedExpert.Down)
			}
		case nonGated:
			mats = append(mats, &lw.UpProj, &lw.DownProj)
		default:
			mats = append(mats, &lw.GateProj, &lw.UpProj, &lw.DownProj)
		}
		for _, wm := range mats {
			if _, _, _, ok := wm.Int8(); !ok || wm.Rows() == 0 || wm.Cols()%4 != 0 {
				return false
			}
		}
	}
	return true
}

// w8AttnEligible reports whether m runs int4mix natively (r.w8Attn slice 4):
// every layer's attention projections int8-kind and its FFN int4-kind (int4mix's per-tensor split), on the plain
// dense path whose qkv and o-proj run pSABias and pSAResid alone. Sandwich, post-only and parallel-block layers run the
// o-proj on pSA, which the FFN shares; non-gated MLPs and output biases run other handles; MoE, DeltaNet and K=V layers
// are not slice 1's shapes. Any of those keeps the int4 re-quantization.
func w8AttnEligible(m *decoder.Model, r *resident) bool {
	if !nativeInt8 || !nativeInt4Mix || r.moe != nil || r.g4moe != nil || r.recurrentModel() || r.sandwich || r.postOnly || r.parallelBlock ||
		r.nonGatedMLP || r.outBias || r.layerNorm {
		return false
	}
	return w8AttnWeights(m)
}

// w8AttnWeights is w8AttnEligible's weight half: attention int8 (K a multiple of 4), gate, up and down int4.
func w8AttnWeights(m *decoder.Model) bool {
	w := m.Weights()
	if len(w.Layers) == 0 {
		return false
	}
	for l := range w.Layers {
		lw := &w.Layers[l]
		for _, wm := range []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj, &lw.OProj} {
			if _, _, _, ok := wm.Int8(); !ok || wm.Rows() == 0 || wm.Cols()%4 != 0 {
				return false
			}
		}
		for _, wm := range []*linalg.WeightMat{&lw.GateProj, &lw.UpProj, &lw.DownProj} {
			if wm.Kind() != "int4" || wm.Rows() == 0 {
				return false
			}
		}
	}
	return true
}

// f32Projection names the first dense body projection still at f32, "" when there is none (every projection int4 or
// int8). A model's router and other f32-kept tensors are not body projections and do not count.
func f32Projection(m *decoder.Model) string {
	w := m.Weights()
	for l := range w.Layers {
		lw := &w.Layers[l]
		for _, c := range []struct {
			name string
			wm   *linalg.WeightMat
		}{{"q_proj", &lw.QProj}, {"o_proj", &lw.OProj}, {"up_proj", &lw.UpProj}, {"down_proj", &lw.DownProj}} {
			if c.wm.Rows() > 0 && c.wm.Kind() == "f32" {
				return fmt.Sprintf("layer %d %s", l, c.name)
			}
		}
	}
	return ""
}

// attnSoftcapDropForTest builds every resident without its attention-score softcap while the CPU keeps it: the planted
// defect the Gemma 2 softcap gates must catch.
// Tests in this package set it directly.
var attnSoftcapDropForTest bool

// w8PreciseMath compiles the library of a model headed for the native int8 path without fast math: at int8int8 any
// difference from the CPU is amplified by the activation quantization, and fast math widens it.
// It is decided before the build from the model alone, so a model that passes
// w8Weights but is MoE (Gemma 4's parallel dense||MoE) is excluded here too.
func w8PreciseMath(m *decoder.Model, plan metalCtxPlan) bool {
	if !nativeInt8 || w8FastMath || m.BackendView().HasGemma4MoE() {
		return false
	}
	if _, moe := m.BackendView().MoE(); moe {
		// slice 4's MoE, as w8Eligible will admit it: not gpt-oss, not paged
		if _, _, gptoss := m.BackendView().GptOssAct(); !nativeInt8MoE || gptoss || metalMoESlotsRequest(m, plan) != "" {
			return false
		}
		return w8Weights(m)
	}
	return w8Weights(m) || nativeInt4Mix && w8AttnWeights(m) // int4mix's attention runs the same int8 kernels
}

// int8Concat row-concatenates same-K int8 WeightMats into ONE int8 buffer and its per-row f32 scales: int4Concat's
// twin for a w8 model's fused QKV and gate|up. Codes and scales are copied, not re-quantized, so the fused GEMV
// computes exactly what the separate ones would. Slice 1 copies them even from a .giw mapping (the single tensors
// alias through int8BufA); the copy is counted in the alias stats.
func int8ConcatOn(line string, d *Device, a *weightAlias, wms ...*linalg.WeightMat) (Buffer, Buffer) {
	var nCodes, nRows int
	for _, w := range wms {
		nCodes += w.Rows() * w.Cols()
		nRows += w.Rows()
	}
	codes := make([]int8, 0, nCodes)
	scales := make([]float32, 0, nRows)
	for _, w := range wms {
		q8, sc, _, ok := w.Int8()
		if !ok {
			panic(fmt.Sprintf("metal: int8Concat weight kind %q is not int8", w.Kind())) // buildResident's recover declines
		}
		codes = append(codes, q8[:w.Rows()*w.Cols()]...)
		scales = append(scales, sc[:w.Rows()]...)
	}
	a.addCopy(int64(len(codes) + 4*len(scales)))
	return NewBufferInt8On(line, d, codes), NewBufferFloatsOn(line, d, scales)
}

package wbuild

import (
	"runtime"
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// Weight matrices are linalg.WeightMat (aikit): one type that hides f32, per-row int8 and group-wise int4 storage behind
// uniform accessors and the linalg kernels. goinfer keeps the model policy here (which table gets which precision, the
// int4 group size, the matmul backend routing) while the storage wrapper, quantize primitives and Row dequant live in
// linalg.

// QuantMode selects the resident weight precision the loader streams into (see
// loadWeights). The f32 path keeps the widened weights; int8 is per-row symmetric
// (¼ f32); int4 is group-wise symmetric (~⅛ f32).
type QuantMode uint8

const (
	QuantNone QuantMode = iota
	QuantInt8
	QuantInt4
	QuantInt8I8 // weights int8 (as QuantInt8) but matmul is full int8×int8 (W8A8)
	// QuantInt4Mix is a per-tensor mixed mode: attention (and embed/head/router) at int8, the FFN bulk
	// (gate/up/down/experts) at int4. It is a load-time policy only: matmulQuant resolves it per tensor, so the resident
	// weights and the .giw never carry QuantInt4Mix itself. GGUF load path only.
	QuantInt4Mix
	// QuantQ4K keeps every tensor the GGUF stores as Q4_K in its own super-block layout
	// (linalg.WrapQ4K, exact, 4.5 bpw) and puts every other layer matmul at int8 W8A8, with per-32
	// activations throughout (the re-quantization
	// of Q4_K into symmetric int4 is the int4 quality loss). Like QuantInt4Mix it is a LOAD-TIME
	// policy resolved per tensor at the GGUF load sites. GGUF only; its .giw form is kind 8 (serialize.go).
	QuantQ4K
	// QuantBF16 and QuantF16 keep a matmul tensor as 16-bit bit patterns (linalg.WeightMat kinds "bf16" and "f16"), half of f32. The
	// matmul widens a tile of rows and runs the f32 kernel, so over a bf16 (or f16) checkpoint the result is bit-identical to f32 and
	// the resident weights are half the size. CPU path only: no backend uploads this kind, and a bundle cannot hold it. A source
	// that is not already that precision is rounded to it.
	QuantBF16
	QuantF16
)

// Half16 reports whether the mode stores weights as 16-bit bit patterns.
func (q QuantMode) Half16() bool { return q == QuantBF16 || q == QuantF16 }

// MatmulQuant resolves a matmul tensor's resident precision under the base quant.
// Uniform for every mode except the mixed mode (QuantInt4Mix), which keeps the
// (cheap, sensitive) attention tensors at int8 and the (large, int4-tolerant) FFN
// tensors at int4 — keyed off llama.cpp's tensor names (ffn_* vs attn_*).
func MatmulQuant(base QuantMode, name string) QuantMode {
	if base == QuantQ4K && strings.Contains(name, "ffn_gate_inp") {
		return QuantInt8I8 // the router, as in int4mix below: never the Q4_K kind
	}
	if base != QuantInt4Mix {
		return base
	}
	// The router (ffn_gate_inp) must stay out of the "ffn_" bulk though its name contains that prefix: top-k selection is
	// discrete, so quantizing it flips which experts win rather than adding rounding noise. Current call sites send the router
	// through streamMat(..., QuantNone, ...) and never reach matmulQuant; this guards a future family that adds a router
	// through the generic mat() helper, which would then get int8 rather than int4.
	if strings.Contains(name, "ffn_gate_inp") {
		return QuantInt8
	}
	if strings.Contains(name, "ffn_") {
		return QuantInt4
	}
	return QuantInt8
}

// Embedding returns the precision for the token-embedding table and the LM head, tied or not. In the int4 modes it is
// full W8A8 (int8 weights and int8 activations), not weight-only Q8: the LM head on weight-only Q8 was the largest
// per-token cost in int4 decode. It is not int4 either: int4-weight quantizing these tensors flips the argmax and tanks
// the cosine (GGUF Q4_K_M likewise keeps token_embd/output above 4 bits). The pin is the unconditional int4-mode default.
// int8 and f32 modes use themselves. Evidence: docs/code-notes/decoder.md#QuantMode.embedding.
func (q QuantMode) Embedding() QuantMode {
	if q == QuantInt4 || q == QuantInt4Mix || q == QuantQ4K {
		return QuantInt8I8
	}
	return q
}

// EmbeddingWith resolves the embed/head precision, allowing the int8 pin to be relaxed to int4 (Options.EmbedInt4): in
// int4 mode the table goes int4 too, halving the largest resident tensor on a big-vocab small model. Lossy and opt-in;
// off (the pin) is the default and the bit-exact path.
func (q QuantMode) EmbeddingWith(embedInt4 bool) QuantMode {
	if embedInt4 && q == QuantInt4 {
		return QuantInt4
	}
	return q.Embedding()
}

// GatherTable is the precision of a table that is only gathered by row on the host, never multiplied (Gemma 4's per-layer
// token table): int4 at int4, whatever the head pin (EmbedInt4) says, since the pin exists for the logit GEMV and the Metal
// head kernel, neither of which reads it.
func (q QuantMode) GatherTable() QuantMode { return q.EmbeddingWith(true) }

// Int4GroupSize is the number of consecutive input features that share one f32
// scale in the int4 path. 32 matches GGUF Q4_K's sub-block granularity — small
// enough to keep 4-bit accuracy, large enough that the per-group scale overhead
// stays ~0.125 byte/element.
const Int4GroupSize = 32

// StreamQuantized builds a [rows, cols] linalg.WeightMat in the target precision
// by dequantizing each row through rowInto (into a reused cols-wide scratch) and
// quantizing it straight into the resident arrays — never materializing the whole
// [rows*cols] f32. This is the load-time memory-bandwidth win: a big GGUF tensor's
// f32 intermediate stays one row wide (in cache) instead of streaming to DRAM and
// back. Bit-identical to h.QuantizeWM(WrapF32(fullF32), mode): the per-row primitives
// here are exactly the ones linalg.Quantize{Rows,Groups} call internally, just
// driven one row at a time; the result is WrapInt8/WrapInt4/WrapF32'd (no copy).
func (h *Hooks) StreamQuantized(rows, cols int, mode QuantMode, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return h.streamQuantizedRow4(rows, cols, mode, true, rowInto)
}

// StreamQuantizedSkipRow4 is streamQuantized with the arm64 row4 side-copy skipped, for a tensor reached only through
// streamQuantized's non-batched callers (o_proj/down_proj/router via streamMat, MoE experts via stackedExperts, gpt-oss's
// expert gate/up/down, fused-tensor splits via fusedSplit). It is the GGUF/streaming twin of quantizeWMSkipRow4, with the
// same scope: canonical alone, no needCanonical branch.
func (h *Hooks) StreamQuantizedSkipRow4(rows, cols int, mode QuantMode, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return h.streamQuantizedRow4(rows, cols, mode, false, rowInto)
}

// streamQuantizedRow4 is streamQuantized/streamQuantizedSkipRow4's shared body; row4 selects
// which of repackW4A8IfEligible (canonical+row4, today's default) or
// repackW4A8IfEligibleSkipRow4 (canonical alone) runs the int4 case — every other quant mode is
// row4-independent (row4 is int4-only).
func (h *Hooks) streamQuantizedRow4(rows, cols int, mode QuantMode, row4 bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	scratch := make([]float32, cols)
	if mode == QuantQ4K {
		// Reached only where a load site has no raw Q4_K bytes to wrap (a non-Q4_K tensor, or a site
		// q4kMat does not serve): the rest of QuantQ4K is int8 W8A8.
		mode = QuantInt8I8
	}
	switch mode {
	case QuantInt8, QuantInt8I8:
		q8 := make([]int8, rows*cols)
		scales := make([]float32, rows)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			scales[r] = linalg.QuantizeRowInt8(scratch, q8[r*cols:(r+1)*cols])
		}
		return linalg.WrapInt8(q8, scales, rows, cols, mode == QuantInt8I8), nil
	case QuantInt4:
		const group = Int4GroupSize
		nGroups := (cols + group - 1) / group
		bpr := (cols + 1) / 2
		q4 := make([]byte, rows*bpr)
		q4s := make([]uint16, rows*nGroups)
		rowScales := make([]float32, nGroups)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			linalg.QuantizeGroupInt4Row(scratch, cols, group, q4[r*bpr:(r+1)*bpr], rowScales)
			linalg.F32ToF16Slice(q4s[r*nGroups:(r+1)*nGroups], rowScales) // int4 scales are stored as binary16
		}
		canon, _ := linalg.WrapInt4(q4, q4s, rows, cols, group, linalg.LayoutCanonical)
		if row4 {
			return h.repackW4A8IfEligible(canon), nil
		}
		return h.repackW4A8IfEligibleSkipRow4(canon), nil
	case QuantBF16, QuantF16:
		u := make([]uint16, rows*cols)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			if mode == QuantBF16 {
				for k, f := range scratch {
					u[r*cols+k] = linalg.F32ToBF16(f)
				}
			} else {
				linalg.F32ToF16Slice(u[r*cols:(r+1)*cols], scratch)
			}
		}
		if mode == QuantBF16 {
			return linalg.WrapBF16(u, rows, cols), nil
		}
		return linalg.WrapF16(u, rows, cols), nil
	default: // QuantNone — no quant target, keep the full f32
		f32 := make([]float32, rows*cols)
		for r := range rows {
			if err := rowInto(r, f32[r*cols:(r+1)*cols]); err != nil {
				return linalg.WeightMat{}, err
			}
		}
		return linalg.WrapF32(f32, rows, cols), nil
	}
}

// streamQuantizedRepackable is streamQuantized's twin for the GGUF streaming path, behind streamQuantizedBatchedProj (the
// logit tables have a row-parallel form of their own, streamQuantizedEmbed); the policy and each caller's safety argument are on quantizeEmbedWM,
// quantizeBatchedProjWM and repackedOnlyOrCanonical. The row-quantize loop is the same whatever the final layout; the
// twin differs only in the last step for QuantInt4, where the repack decision goes through repackedOnlyOrCanonical
// (row4-only, or canonical optionally row4-skipped) instead of streamQuantized's fixed canonical plus row4.
func (h *Hooks) streamQuantizedRepackable(rows, cols int, mode QuantMode, needCanonical, skipRow4 bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	if mode != QuantInt4 || h.fakeQuantScheme() != "" {
		return h.StreamQuantized(rows, cols, mode, rowInto)
	}
	scratch := make([]float32, cols)
	const group = Int4GroupSize
	nGroups := (cols + group - 1) / group
	bpr := (cols + 1) / 2
	q4 := make([]byte, rows*bpr)
	q4s := make([]uint16, rows*nGroups)
	rowScales := make([]float32, nGroups)
	for r := range rows {
		if err := rowInto(r, scratch); err != nil {
			return linalg.WeightMat{}, err
		}
		linalg.QuantizeGroupInt4Row(scratch, cols, group, q4[r*bpr:(r+1)*bpr], rowScales)
		linalg.F32ToF16Slice(q4s[r*nGroups:(r+1)*nGroups], rowScales) // int4 scales are stored as binary16
	}
	canon, _ := linalg.WrapInt4(q4, q4s, rows, cols, group, linalg.LayoutCanonical)
	return h.repackedOnlyOrCanonical(canon, needCanonical, skipRow4), nil
}

// RowReader fills dst with row r of a table.
type RowReader func(r int, dst []float32) error

// streamEmbedParallelMinElems is the table size, in elements, from which a logit table's rows are quantized on several
// goroutines; a smaller one is not worth the fan-out.
const streamEmbedParallelMinElems = 1 << 22

// ForRowRanges runs do over [0, rows) split into contiguous ranges, one goroutine per range, each with a reader of its own
// from newReader, so a reader may keep state. When workers <= 0, the worker count defaults to GOMAXPROCS for tables at or
// above streamEmbedParallelMinElems and 1 below. Every range but the last is a multiple of align rows long (a reader that
// caches blocks of align rows then never shares a block with another range). do must write only its own rows. The first
// range's error wins, so a failure reads the same however the ranges were scheduled.
func ForRowRanges(rows, cols, align, workers int, newReader func() RowReader, do func(lo, hi int, read RowReader) error) error {
	if workers <= 0 {
		workers = 1
		if rows*cols >= streamEmbedParallelMinElems {
			workers = runtime.GOMAXPROCS(0)
		}
	}
	align = max(align, 1)
	per := (rows + workers - 1) / max(workers, 1)
	per = (per + align - 1) / align * align
	if workers <= 1 || per >= rows {
		return do(0, rows, newReader())
	}
	errs := make([]error, (rows+per-1)/per)
	var wg sync.WaitGroup
	for i := range errs {
		lo := i * per
		wg.Go(func() { errs[i] = do(lo, min(lo+per, rows), newReader()) })
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// StreamEmbedWorkers returns the worker count for parallel embedding row streaming.
// streamQuantizedEmbed builds Embed/LMHead (and Gemma 4's per-layer token table) a row at a time, as
// streamQuantizedRepackable does and with the same per-row primitives, on several goroutines over contiguous row ranges
// (forRowRanges): these are single tensors that run before any layer, so nothing else is in flight. Each row is
// quantized on its own, so the result is the sequential one bit for bit. The safety argument for the repacked-only
// layout is quantizeEmbedWM's.
//
// newReader makes a row reader, once per range; align is the reader's block size in rows, or 1 for one that keeps no
// state.
//
// skipRow4 is false for every load that will decode: Embed/LMHead are read via .Row() on the host on every backend, so the
// row4-skip trade (a slower CPU fallback for less resident memory) is not this class's call, and quantizeEmbedWM makes
// the same choice. Only a build for a sink passes true (!repackForDecode(sink)): nothing reads the table there.
func StreamEmbedWorkers(sink *StreamWriter) int {
	if sink == nil {
		return 0
	}
	return sink.TestHooks.EmbedWorkers
}

func (h *Hooks) StreamQuantizedEmbed(rows, cols int, mode QuantMode, needCanonical, skipRow4 bool, align, workers int, newReader func() RowReader) (linalg.WeightMat, error) {
	if mode == QuantQ4K {
		mode = QuantInt8I8 // a logit table has no raw Q4_K form here, as in streamQuantizedRow4
	}
	if mode == QuantInt4 && h.fakeQuantScheme() != "" {
		return h.StreamQuantized(rows, cols, mode, newReader())
	}
	switch mode {
	case QuantInt8, QuantInt8I8:
		q8 := make([]int8, rows*cols)
		scales := make([]float32, rows)
		err := ForRowRanges(rows, cols, align, workers, newReader, func(lo, hi int, read RowReader) error {
			scratch := make([]float32, cols)
			for r := lo; r < hi; r++ {
				if err := read(r, scratch); err != nil {
					return err
				}
				scales[r] = linalg.QuantizeRowInt8(scratch, q8[r*cols:(r+1)*cols])
			}
			return nil
		})
		if err != nil {
			return linalg.WeightMat{}, err
		}
		return linalg.WrapInt8(q8, scales, rows, cols, mode == QuantInt8I8), nil
	case QuantInt4:
		const group = Int4GroupSize
		nGroups := (cols + group - 1) / group
		bpr := (cols + 1) / 2
		q4 := make([]byte, rows*bpr)
		q4s := make([]uint16, rows*nGroups)
		err := ForRowRanges(rows, cols, align, workers, newReader, func(lo, hi int, read RowReader) error {
			scratch := make([]float32, cols)
			rowScales := make([]float32, nGroups)
			for r := lo; r < hi; r++ {
				if err := read(r, scratch); err != nil {
					return err
				}
				linalg.QuantizeGroupInt4Row(scratch, cols, group, q4[r*bpr:(r+1)*bpr], rowScales)
				linalg.F32ToF16Slice(q4s[r*nGroups:(r+1)*nGroups], rowScales) // int4 scales are stored as binary16
			}
			return nil
		})
		if err != nil {
			return linalg.WeightMat{}, err
		}
		wm4, _ := linalg.WrapInt4(q4, q4s, rows, cols, group, linalg.LayoutCanonical)
		return h.repackedOnlyOrCanonical(wm4, needCanonical, skipRow4), nil
	default: // QuantNone: keep the full f32
		f32 := make([]float32, rows*cols)
		err := ForRowRanges(rows, cols, align, workers, newReader, func(lo, hi int, read RowReader) error {
			for r := lo; r < hi; r++ {
				if err := read(r, f32[r*cols:(r+1)*cols]); err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil {
			return linalg.WeightMat{}, err
		}
		return linalg.WrapF32(f32, rows, cols), nil
	}
}

// StreamQuantizedBatchedProj is streamQuantizedRepackable for attention Q/K/V and MLP gate/up; the batch-path safety
// argument is quantizeBatchedProjWM's.
func (h *Hooks) StreamQuantizedBatchedProj(rows, cols int, mode QuantMode, needCanonical, skipRow4 bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return h.streamQuantizedRepackable(rows, cols, mode, needCanonical, skipRow4, rowInto)
}

// QuantizeWM returns w streamed to the requested resident precision; the f32 memory is freed by dropping the old
// reference at the call site. No-op for QuantNone or if w is not f32-resident (already quantized, or empty).
func (h *Hooks) QuantizeWM(w linalg.WeightMat, mode QuantMode) linalg.WeightMat {
	return h.quantizeWMRow4(w, mode, true)
}

// QuantizeWMSkipRow4 is quantizeWM with the arm64 row4 side-copy skipped, for a projection reached only through
// quantizeWM's family-specific call sites (o_proj, down_proj, router, MoE experts, shared expert: everything
// isBatchedProjTensor does not cover). Unlike quantizeBatchedProjWM and quantizeEmbedWM it has no needCanonical branch:
// repacked-only is the CPU-batched-dispatch trade repackedOnlyOrCanonical exists for, and none of these tensor classes
// go through that dispatch. It only chooses between canonical plus row4 and canonical alone.
func (h *Hooks) QuantizeWMSkipRow4(w linalg.WeightMat, mode QuantMode) linalg.WeightMat {
	return h.quantizeWMRow4(w, mode, false)
}

// quantizeWMRow4 is the shared body of quantizeWM and quantizeWMSkipRow4: row4 selects repackW4A8IfEligible (canonical
// plus row4) or repackW4A8IfEligibleSkipRow4 (canonical alone) for the int4 case. Every other quant mode ignores row4.
func (h *Hooks) quantizeWMRow4(w linalg.WeightMat, mode QuantMode, row4 bool) linalg.WeightMat {
	f32, ok := w.F32()
	if !ok {
		return w
	}
	if mode == QuantQ4K {
		mode = QuantInt8I8 // no raw Q4_K bytes here: see streamQuantizedRow4
	}
	switch mode {
	case QuantInt8:
		return linalg.QuantizeInt8(f32, w.Rows(), w.Cols(), false)
	case QuantInt8I8:
		return linalg.QuantizeInt8(f32, w.Rows(), w.Cols(), true)
	case QuantBF16:
		return linalg.QuantizeBF16(f32, w.Rows(), w.Cols())
	case QuantF16:
		return linalg.QuantizeF16(f32, w.Rows(), w.Cols())
	case QuantInt4:
		if h.fakeQuantScheme() != "" { // DIAGNOSTIC (default-off, single load-time env read): see fakequant.go
			return h.FakeInt4WM(f32, w.Rows(), w.Cols(), h.fakeQuantScheme())
		}
		canon := linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), Int4GroupSize)
		if row4 {
			return h.repackW4A8IfEligible(canon)
		}
		return h.repackW4A8IfEligibleSkipRow4(canon)
	default:
		return w
	}
}

// RepackW4A8Row4IfEligible opts wm into the arm64 split-half, 4-row-interleaved W4A8 layout by calling
// linalg.WeightMat.RepackInt4Row4: a no-op on non-int4 WeightMats, non-arm64 builds and shapes the repack rejects (rows
// not a multiple of 4, cols not a multiple of the int4 group size), so always safe to call unconditionally.
//
// Only wired into streamQuantized and quantizeWM, the GGUF/safetensors streaming paths, which allocate fresh heap-backed
// q4/q4s. Deliberately not wired into the .giw loader (serialize.go): .giw tensors zero-copy mmap-alias their packed
// bytes, and some (MoE experts under newExpertPager) are later released from RAM via madvise DONTNEED (moepaging.go). A
// heap-resident row4 copy beside a pageable alias would pin that memory permanently, defeating paging for exactly the
// tensors it exists to bound. Streamed int4 tensors are heap-backed and never paged (moepaging.go skips them), so
// repacking here adds no pageability constraint.
//
// Limitation: non-paged .giw tensors get no row4 repack. Extending it needs the decision sequenced after newExpertPager
// picks its experts (docs/code-notes/decoder.md#repackW4A8Row4IfEligible).
func (h *Hooks) RepackW4A8Row4IfEligible(wm linalg.WeightMat) linalg.WeightMat {
	if c := h.row4RepackCalls(); c != nil {
		c.Add(1)
	}
	if h != nil && h.DisableRow4Repack {
		return wm
	}
	wm.RepackInt4Row4()
	return wm
}

// RepackW4A8SplitHalfIfEligible is the amd64 counterpart to repackW4A8Row4IfEligible: it opts wm into the split-half W4A8
// nibble layout (byte i holds weight i's low nibble and weight i+16's high nibble, which removes the AVX2 kernel's two
// per-group VPUNPCK{L,H}BW). A no-op on non-int4 WeightMats, non-amd64 builds, CPUs without AVX2 and any shape the repack
// rejects, so always safe to call unconditionally.
//
// Also a no-op on hosts with AVX-512 VNNI: aikit's canonical W4A8 dot prefers its VNNI tier there and split-half exists
// only at AVX2, so the layout would swap a faster kernel for a slower one. The repack therefore applies to
// AVX2-without-VNNI hosts only.
//
// Wired into the same two call sites as the row4 repack and for the same reason (see that function for why the .giw
// loader is excluded). The repack allocates a second buffer and never writes through the canonical bytes, which for a.
// giw kind=3 tensor are a zero-copy mmap alias of the file; rewriting them in place would silently misdecode every
// existing bundle (aikit's TestWeightMatSplitHalf_canonicalUntouched pins that it does not).
//
// Memory: a second copy of every eligible tensor's nibbles, with canonical kept. The cost and why it is default-off live
// on w4a8SplitHalfRepackEnabled, in one place so the figures cannot drift apart.
func (h *Hooks) RepackW4A8SplitHalfIfEligible(wm linalg.WeightMat) linalg.WeightMat {
	if h == nil || !h.SplitHalfRepackEnabled {
		return wm
	}
	if wm.RepackInt4SplitHalf() {
		if h.SplitHalfRepacked != nil {
			h.SplitHalfRepacked.Add(1)
		}
		// The second copy's actual cost, asked of the layout's owner rather than re-derived from rows x ceil(cols/2): the memory
		// half of the trade must be a computed quantity, and duplicating aikit's arithmetic would let it drift.
		if h.SplitHalfBytes != nil {
			h.SplitHalfBytes.Add(int64(wm.SplitHalfBytes()))
		}
	} else {
		if h.SplitHalfSkipped != nil {
			h.SplitHalfSkipped.Add(1)
		}
	}
	return wm
}

// repackW4A8IfEligible applies whichever ISA-specific W4A8 layout THIS build
// has a kernel for: row4 on arm64, split-half on amd64. Each is a no-op off its
// own architecture, so both are called unconditionally and the two stay
// symmetric — a third layout gets added here and nowhere else.
func (h *Hooks) repackW4A8IfEligible(wm linalg.WeightMat) linalg.WeightMat {
	return h.RepackW4A8SplitHalfIfEligible(h.RepackW4A8Row4IfEligible(wm))
}

// repackW4A8IfEligibleSkipRow4 is repackW4A8IfEligible's twin for a backend committed to GPU residency that will never
// read the arm64 row4 layout (see wantsRow4Fallback): it applies only the amd64 split-half repack and skips row4.
func (h *Hooks) repackW4A8IfEligibleSkipRow4(wm linalg.WeightMat) linalg.WeightMat {
	return h.RepackW4A8SplitHalfIfEligible(wm)
}

// RepackForDecode reports whether an int4 tensor built now also gets this CPU's row4 decode layout in RAM. It is false
// while building for a sink: a transcode decodes nothing, and the writer derives the layout a target's file needs from
// the canonical bytes itself (repackRow4ForEmit). Every row4 decision in a build that can have a sink goes through it:
// buildWeightsFromGGUF and buildWeightsFromSafetensorsTo fold it into skipRow4 before any tensor is built, and the logit
// tables, which never skip row4 otherwise, ask it directly. The amd64 split-half repack is off by default and is not
// covered.
func RepackForDecode(sink *StreamWriter) bool { return sink == nil }

// WantsRow4Fallback reports whether the arm64 row4 repack should be built alongside canonical as a safety net for a CPU
// fallback. Only the CPU's decode kernel reads row4; no GPU backend does. It exists so a model that falls back to CPU
// decode (a declined residency build, LoRA/session paths outside a resident backend's coverage) keeps its speed.
//
// False only for backendName == "metal": Options.Backend is the caller's commitment to that backend (stated, never
// inferred; see wantsCanonicalInt4), and row4 doubles the resident int4 footprint of every projection it applies to, so a
// slower CPU fallback after a declined Metal residency is the price. CUDA and WebGPU residency have not been measured
// against this trade and keep the default. Measurement: docs/code-notes/decoder.md#wantsRow4Fallback.
func WantsRow4Fallback(backendName string) bool {
	return backendName != "metal"
}

// repackedOnlyOrCanonical is the decision shared by quantizeEmbedWM and quantizeBatchedProjWM: given a canonical int4
// WeightMat, build it repacked-only when needCanonical is false and this core and shape can build row4
// (linalg.PreferredInt4Layout (== LayoutRow4)), which avoids the double nibble+scale residency of canonical+row4. Otherwise (needCanonical,
// no row4 on this core, or a shape that does not qualify) it applies repackW4A8IfEligible's policy, what every other int4
// tensor gets from quantizeWM, so it is never worse than the default.
//
// amd64 split-half repacked-only is out of scope: split-half is parked (w4a8SplitHalfRepackEnabled, default off).
//
// skipRow4 is consulted only when needCanonical: the other branch is the CPU-only repacked-only path row4 exists for.
// Callers that must never skip row4 (Embed/LMHead) pass false.
func (h *Hooks) repackedOnlyOrCanonical(canon linalg.WeightMat, needCanonical, skipRow4 bool) linalg.WeightMat {
	if needCanonical {
		if skipRow4 {
			return h.repackW4A8IfEligibleSkipRow4(canon)
		}
		return h.repackW4A8IfEligible(canon)
	}
	q4, q4s, group, ok := canon.Int4F16()
	if !ok || linalg.PreferredInt4Layout(canon.Rows(), canon.Cols(), group) != linalg.LayoutRow4 {
		return h.repackW4A8IfEligible(canon) // this core/shape can't do row4 at all — existing policy
	}
	if repacked, repOK := linalg.RepackInt4InPlace(q4, q4s, canon.Rows(), canon.Cols(), group, linalg.LayoutRow4); repOK {
		return repacked
	}
	return h.repackW4A8IfEligible(canon) // PreferredInt4Layout said yes, so this shouldn't miss — no silent data loss either way
}

// QuantizeEmbedWM is quantizeWM's Embed/LMHead twin; the policy is repackedOnlyOrCanonical. needCanonical is computed once
// per Load (wantsCanonicalInt4) and threaded down. Embed/LMHead are read via WeightMat.Row() (layout-independent) and
// driven through matmul/matmulInto as the LM head (single-op dispatch gated on IsInt4), never through the batched q/k/v
// or gate/up dispatch. It stays separate from quantizeBatchedProjWM although the bodies match: the two safety arguments
// are independent, and a change to one dispatch shape must not silently start covering the other tensor class.
func (h *Hooks) QuantizeEmbedWM(w linalg.WeightMat, mode QuantMode, needCanonical bool) linalg.WeightMat {
	if mode != QuantInt4 || h.fakeQuantScheme() != "" {
		return h.QuantizeWM(w, mode)
	}
	f32, ok := w.F32()
	if !ok {
		return h.QuantizeWM(w, mode) // already quantized, or empty — quantizeWM's own no-op path
	}
	canon := linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), Int4GroupSize)
	// skipRow4 is false: Embed/LMHead are read via .Row() on the host on every backend, so the row4-skip trade is not this
	// class's call (streamQuantizedEmbed makes the same choice).
	return h.repackedOnlyOrCanonical(canon, needCanonical, false)
}

// QuantizeBatchedProjWM is quantizeWM's twin for attention Q/K/V and MLP gate/up, the tensors reached through the
// batched W4A8 dispatch (matmulW4A8Batch, wmW4A8Op, isW4A8). The fused kernel serves a repacked-only op only at M=1
// over whole quads; linalg.MatmulBTBatchInto runs any other such batch per op instead (M>1, or a row count not a
// multiple of 4), so it is correct everywhere and fused where the conditions below hold.
//
// It is safe for the per-token walker's two batch call sites, projQKVToken and gatedMLPAct: both are decode-only and call matmulW4A8Batch
// with a literal M=1, while prefill uses matmul/matmulInto per projection, which are safe at any M. wmW4A8Op builds the
// right op shape for a repacked-only tensor and isW4A8's IsInt4 gate lets it through. The N%4==0 condition is
// linalg.PreferredInt4Layout (== LayoutRow4)'s rows%4 check, applied per tensor by repackedOnlyOrCanonical, so a violating shape is never
// built repacked-only.
//
// Down-proj, the router and MoE expert weights are not this class and take the same policy through quantizeLayerWM. The
// one batched dispatch an expert reaches, gpt-oss's gate and up (Model.gptOssExpertBatch), is decode-only at M=1 and
// per-tensor rows%4 as above, so the same argument covers it. Expert weights read through a read-only mmap span
// (paged .giw loading) have no load-time repack, so a .giw keeps them canonical.
func (h *Hooks) QuantizeBatchedProjWM(w linalg.WeightMat, mode QuantMode, needCanonical, skipRow4 bool) linalg.WeightMat {
	if mode != QuantInt4 || h.fakeQuantScheme() != "" {
		return h.QuantizeWM(w, mode)
	}
	f32, ok := w.F32()
	if !ok {
		return h.QuantizeWM(w, mode)
	}
	canon := linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), Int4GroupSize)
	return h.repackedOnlyOrCanonical(canon, needCanonical, skipRow4)
}

// QuantizeLayerWM is quantizeWM for an int4 layer tensor outside the embedding tables and the batched Q/K/V/gate/up
// class (o_proj, down_proj, routed and shared experts, the router and the family-specific layer tensors in
// quantizeWMSkipRow4's class), under repackedOnlyOrCanonical's policy: a load that needs no canonical bytes
// (wantsCanonicalInt4 false: Backend "cpu") keeps one row4 copy where the shape allows it, instead of canonical plus row4.
//
// Safe for every CPU reader: on arm64, linalg.WeightMat.MatmulBTW4A8Into and MatmulBTW4A8PreInto dispatch to a present
// row4 layout at every M, grouped activations included, so a canonical copy beside it is never read on the CPU; a
// batched W4A8 call takes a repacked-only op only at M=1 (quantizeBatchedProjWM), and the multi-row fused call requires
// canonical bytes (w4a8FusedOps). The .giw writer needs canonical bytes and refuses a repacked-only tensor
// (giwWriter.weightMatKind). Separate from quantizeBatchedProjWM although the bodies match: the safety arguments differ.
func (h *Hooks) QuantizeLayerWM(w linalg.WeightMat, mode QuantMode, needCanonical, skipRow4 bool) linalg.WeightMat {
	if mode != QuantInt4 || h.fakeQuantScheme() != "" {
		if skipRow4 {
			return h.QuantizeWMSkipRow4(w, mode)
		}
		return h.QuantizeWM(w, mode)
	}
	f32, ok := w.F32()
	if !ok {
		return h.QuantizeWM(w, mode)
	}
	canon := linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), Int4GroupSize)
	return h.repackedOnlyOrCanonical(canon, needCanonical, skipRow4)
}

// StreamQuantizedLayer is quantizeLayerWM's streaming twin (the GGUF row dequantizer, safetensors expert stacks).
func (h *Hooks) StreamQuantizedLayer(rows, cols int, mode QuantMode, needCanonical, skipRow4 bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	if mode != QuantInt4 || h.fakeQuantScheme() != "" {
		if skipRow4 {
			return h.StreamQuantizedSkipRow4(rows, cols, mode, rowInto)
		}
		return h.StreamQuantized(rows, cols, mode, rowInto)
	}
	return h.streamQuantizedRepackable(rows, cols, mode, needCanonical, skipRow4, rowInto)
}

// IsBatchedProjTensor reports whether name (llama.cpp's GGUF tensor-name convention) is attention Q/K/V or MLP gate/up,
// the groups quantizeBatchedProjWM covers. It suffix-matches the exact standard names, not a substring, so an MoE expert
// ("ffn_gate_exps.weight") or router ("ffn_gate_inp.weight") never matches, and a family-specific split projection (an
// MLA q_a_proj/q_b_proj pair) stays on the quantizeWM/streamQuantized path with no per-family audit.
func IsBatchedProjTensor(name string) bool {
	for _, suf := range [...]string{"attn_q.weight", "attn_k.weight", "attn_v.weight", "ffn_gate.weight", "ffn_up.weight"} {
		if strings.HasSuffix(name, suf) {
			return true
		}
	}
	return false
}

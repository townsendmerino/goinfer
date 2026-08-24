package decoder

import (
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// Weight matrices are linalg.WeightMat (aikit): one type that hides f32 / per-row
// int8 / group-wise int4 storage behind uniform accessors + the linalg kernels.
// goinfer keeps the model POLICY here — which table gets which precision
// (quantMode), the int4 group size, and the matmul backend routing (the staged
// GPU hooks below) — while the storage wrapper, quantize primitives, and Row
// dequant live in linalg. (Consolidates the wrapper formerly open-coded as
// decoder.weightMat; see aikit linalg.WeightMat.)

// quantMode selects the resident weight precision the loader streams into (see
// loadWeights). The f32 path keeps the widened weights; int8 is per-row symmetric
// (¼ f32); int4 is group-wise symmetric (~⅛ f32).
type quantMode uint8

const (
	quantNone quantMode = iota
	quantInt8
	quantInt4
	quantInt8I8 // weights int8 (as quantInt8) but matmul is full int8×int8 (W8A8)
	// quantInt4Mix is a per-tensor mixed mode (idea #5): attention (+ embed/head/
	// router) at int8 where a calibration spike found the int4→int8 quality loss
	// concentrated, the FFN bulk (gate/up/down/experts) at int4. It is a LOAD-TIME
	// policy only — matmulQuant resolves it to int8/int4 per tensor, so the resident
	// weights and the .giw never carry quantInt4Mix itself. GGUF load path only.
	quantInt4Mix
)

// matmulQuant resolves a matmul tensor's resident precision under the base quant.
// Uniform for every mode except the mixed mode (quantInt4Mix), which keeps the
// (cheap, sensitive) attention tensors at int8 and the (large, int4-tolerant) FFN
// tensors at int4 — keyed off llama.cpp's tensor names (ffn_* vs attn_*).
func matmulQuant(base quantMode, name string) quantMode {
	if base != quantInt4Mix {
		return base
	}
	if strings.Contains(name, "ffn_") {
		return quantInt4
	}
	return quantInt8
}

// embedding returns the precision to use for the token-embedding table (and the
// LM head, tied or not). These are logit-critical: at int4 they flip the argmax
// and tank the cosine (the tied head dots every logit against them), so int4
// mode keeps them at int8 — mirroring how GGUF Q4_K_M keeps token_embd/output at
// Q6_K while the projections go 4-bit. int8 and f32 modes use themselves.
func (q quantMode) embedding() quantMode {
	if q == quantInt4 || q == quantInt4Mix {
		return quantInt8
	}
	return q
}

// embeddingWith resolves the embed/head precision allowing the int8 pin to be
// relaxed to int4 (Options.EmbedInt4): in int4 mode the table goes int4 too,
// halving what is the single largest resident tensor on a big-vocab small model.
// Lossy and opt-in — a 1.5B Q4_K_M spike measured ~2.3 pts top-1 vs the pin (≈0 on
// frequent tokens, ~3 on rare). Off (the pin) is the default and the bit-exact path.
func (q quantMode) embeddingWith(embedInt4 bool) quantMode {
	if embedInt4 && q == quantInt4 {
		return quantInt4
	}
	return q.embedding()
}

// int4GroupSize is the number of consecutive input features that share one f32
// scale in the int4 path. 32 matches GGUF Q4_K's sub-block granularity — small
// enough to keep 4-bit accuracy, large enough that the per-group scale overhead
// stays ~0.125 byte/element.
const int4GroupSize = 32

// int4ParThreshold lowers the fan-out threshold for the int4 (W4A8) matmul below aikit's
// default (parThreshold = 1<<24 = 16.78M MACs) so the small int4 DECODE matmuls parallelize.
// At decode (M=1) every Gemma-4 int4 matmul is small — expert gate‖up 3.96M, down 1.98M,
// dense ~5.9M, attention ~11.5M MACs — so ALL of them fell under aikit's default and ran
// SERIAL, while only the int8 LM head (738M) parallelized. That serial fast-path (NOT a
// barrier) capped 8-core scaling at 1.61× and decode at ~2.3 tok/s (profiled on the real
// gemma4-26b int4 .giw). 1<<20 ≈ 1.05M sits below the 1.98M smallest decode matmul, so all of
// them fan out, while truly tiny ops (<1M) stay serial. Byte-identical (aikit partitions
// output columns in 8-wide groups — the width-invariant contract), measured ~2.3× decode
// (2.3→5.3 tok/s) + TTFT 7.3→3.2s. Only widens fan-out (never narrows it), so prefill's
// already-parallel large-M matmuls are unaffected. See docs/task-gemma4-moe.md.
//
// PROVENANCE: 1<<20 was **Ryzen 7 3700X (8-core) measured**. On that rig it does not regress
// the small end — a 0.5B int4 decodes 1.9× faster than serial at this value (26.8 vs 13.9
// tok/s, 4 cores) because it parallelizes the ~4M-MAC matmuls while leaving truly tiny (<1M)
// ops serial (thr=0, which fans out everything, was *slower* there — over-parallelizes).
//
// M1 PRO SWEEP (6P+2E, BenchmarkInt4ParThresholdSweep): the value TRANSFERS — it sits in the
// flat-optimal region. All four gemma4-26b decode shapes are ≥1.98M, so 1<<20 (1.05M)
// parallelizes every one, capturing 1.46× (down 1.98M), 1.84× (gate_up 3.96M), 1.9× (dense
// 5.9M), 2.56× (attn 11.5M) vs serial. And UNLIKE the Ryzen, thr=0 shows NO over-parallelize
// penalty on M1 Pro (thr=0 ties the low thresholds), so the Ryzen value is if anything slightly
// conservative here but lands squarely in the optimum for every real decode op. No per-platform
// split warranted. Re-run the benchmark if the core topology or aikit's kernel changes.
const int4ParThreshold = 1 << 20

// streamQuantized builds a [rows, cols] linalg.WeightMat in the target precision
// by dequantizing each row through rowInto (into a reused cols-wide scratch) and
// quantizing it straight into the resident arrays — never materializing the whole
// [rows*cols] f32. This is the load-time memory-bandwidth win: a big GGUF tensor's
// f32 intermediate stays one row wide (in cache) instead of streaming to DRAM and
// back. Bit-identical to quantizeWM(WrapF32(fullF32), mode): the per-row primitives
// here are exactly the ones linalg.Quantize{Rows,Groups} call internally, just
// driven one row at a time; the result is WrapInt8/WrapInt4/WrapF32'd (no copy).
func streamQuantized(rows, cols int, mode quantMode, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	scratch := make([]float32, cols)
	switch mode {
	case quantInt8, quantInt8I8:
		q8 := make([]int8, rows*cols)
		scales := make([]float32, rows)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			scales[r] = linalg.QuantizeRowInt8(scratch, q8[r*cols:(r+1)*cols])
		}
		return linalg.WrapInt8(q8, scales, rows, cols, mode == quantInt8I8), nil
	case quantInt4:
		const group = int4GroupSize
		nGroups := (cols + group - 1) / group
		bpr := (cols + 1) / 2
		q4 := make([]byte, rows*bpr)
		q4s := make([]float32, rows*nGroups)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			linalg.QuantizeGroupInt4Row(scratch, cols, group, q4[r*bpr:(r+1)*bpr], q4s[r*nGroups:(r+1)*nGroups])
		}
		return repackW4A8Row4IfEligible(maybeF16RoundInt4Scales(linalg.WrapInt4(q4, q4s, rows, cols, group))), nil
	default: // quantNone — no quant target, keep the full f32
		f32 := make([]float32, rows*cols)
		for r := range rows {
			if err := rowInto(r, f32[r*cols:(r+1)*cols]); err != nil {
				return linalg.WeightMat{}, err
			}
		}
		return linalg.WrapF32(f32, rows, cols), nil
	}
}

// quantizeWM returns w streamed to the requested resident precision — the
// immutable-WeightMat replacement for the old in-place weightMat.quantize (the
// freed-f32 memory win comes from dropping the old f32 reference at the call site).
// No-op for quantNone or if w isn't f32-resident (already quantized / empty).
func quantizeWM(w linalg.WeightMat, mode quantMode) linalg.WeightMat {
	f32, ok := w.F32()
	if !ok {
		return w
	}
	switch mode {
	case quantInt8:
		return linalg.QuantizeInt8(f32, w.Rows(), w.Cols(), false)
	case quantInt8I8:
		return linalg.QuantizeInt8(f32, w.Rows(), w.Cols(), true)
	case quantInt4:
		if fakeQuantScheme != "" { // DIAGNOSTIC (default-off, single load-time env read): see fakequant.go
			return fakeInt4WM(f32, w.Rows(), w.Cols(), fakeQuantScheme)
		}
		return repackW4A8Row4IfEligible(maybeF16RoundInt4Scales(linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize))) // GOINFER_INT4_F16_SCALES diagnostic
	default:
		return w
	}
}

// repackW4A8Row4IfEligible opts wm into the arm64 split-half + 4-row-interleaved
// W4A8 layout (docs/task-w4a8-neon-bandwidth.md's item-3+4 harness, GO
// 2026-08-23/24) by calling linalg.WeightMat.RepackInt4Row4 — a no-op on
// non-int4 WeightMats, non-arm64 builds, and any shape the repack rejects
// (rows not a multiple of 4, cols not a multiple of the int4 group size), so
// always safe to call unconditionally.
//
// ONLY wired into streamQuantized/quantizeWM — the GGUF/safetensors streaming
// paths, which always allocate fresh heap-backed q4/q4s. Deliberately NOT
// wired into the .giw loader (decoder/serialize.go): .giw tensors zero-copy
// mmap-alias their packed bytes, and SOME of those (MoE experts, when
// newExpertPager's paging is active) are later released from RAM on demand
// via madvise DONTNEED (moepaging.go) — a heap-resident row4 copy sitting
// alongside a pageable mmap alias would pin that memory permanently, defeating
// paging's whole point for exactly the tensors it exists to bound. Every
// GGUF/safetensors-streamed int4 tensor is heap-backed regardless (never
// paged — moepaging.go's own MappedSpan check silently skips heap-backed
// weights), so repacking here adds no new pageability constraint. Extending
// this to non-paged .giw tensors is a real follow-up, out of scope for this
// pass: it needs the repack decision sequenced after newExpertPager decides
// which specific experts it's managing, not made at load time before that
// decision exists.
func repackW4A8Row4IfEligible(wm linalg.WeightMat) linalg.WeightMat {
	if !w4a8Row4RepackEnabled {
		return wm
	}
	wm.RepackInt4Row4()
	return wm
}

// w4a8Row4RepackEnabled is a load-time-measurement toggle ONLY — production
// code never sets it, so it stays true always in a real build. A test
// measuring the repack's load-time/resident-memory delta (the two numbers
// the parked .giw-kind decision is waiting on, docs/task-w4a8-neon-
// bandwidth.md) flips it off to get an apples-to-apples "without repack"
// baseline from the exact same load path, rather than comparing against a
// differently-built binary.
var w4a8Row4RepackEnabled = true

// isW8A8 reports whether w uses the int8×int8 (W8A8) path — the only one with a
// zero-alloc Workspace + batched-dispatch kernel.
func isW8A8(w *linalg.WeightMat) bool {
	_, _, w8a8, ok := w.Int8()
	return ok && w8a8
}

// wmInt8 / wmScales pull the int8 codes / per-row scales out of a WeightMat for the
// batched W8A8 ops (the forward's only sites that read the raw arrays directly, to
// fuse several matrices into one matmulW8A8Batch dispatch). Both assume isW8A8(w).
func wmInt8(w *linalg.WeightMat) []int8      { q8, _, _, _ := w.Int8(); return q8 }
func wmScales(w *linalg.WeightMat) []float32 { _, s, _, _ := w.Int8(); return s }

// matmulWSPool recycles the Workspace matmul() falls back to when the caller has no
// decodeScratch to hand in (dflash/dspark/eagle, and every forward_*.go family that
// hasn't been threaded onto matmulInto). A fresh `var ws linalg.Workspace` per call
// starts with nil i8/f32 scratch, so Into() reallocates BOTH the Workspace and its
// internal quant buffers on every single matmul (P8-class allocation, same shape as
// moeMLP's — see scratch.go). Pooling reuses the grown i8/f32 backing arrays across
// calls; sync.Pool's own GC-driven eviction keeps a workspace that briefly saw one
// huge call (e.g. the vocab-sized LM head) from pinning that size forever.
var matmulWSPool = sync.Pool{New: func() any { return new(linalg.Workspace) }}

// matmul computes dst[M, rows] = a[M, cols] · wᵀ, dispatching on w's precision
// with goinfer's backend routing: the f32 and W8A8 paths can run on a GPU backend
// (be.MatmulBT / QuantBackend.MatmulW8A8); int4 (W4A8) and weight-only int8 (Q8)
// stay CPU. (The old weightMat.matmul, now a free function over linalg.WeightMat.)
func matmul(be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if _, _, _, ok := w.Int4(); ok {
		// int4 weights run the int8-activation W4A8 integer kernel at EVERY M (decode
		// AND prefill): it stays integer (int4 weight × int8 activation) and benchmarks
		// faster than the dequant-to-f32 Q4 path at every M, and its per-output result
		// is M-independent so batched prefill is bit-identical to sequential decode.
		//
		// The pooled Workspace lowers the fan-out threshold below aikit's default so the
		// small int4 DECODE matmuls parallelize instead of running serial — see
		// int4ParThreshold. Each Get is exclusive to this call (Put deferred until
		// return), so concurrent decode streams never share one — same race-freedom as
		// the old per-call ws, just with its buffers surviving between calls.
		//
		// w.MatmulBTW4A8Into (not the raw linalg.MatmulBTW4A8Into free function) so the
		// arm64 split-half + 4-row-interleave repack (docs/task-w4a8-neon-bandwidth.md),
		// when RepackInt4Row4 populated it at load time, actually gets used here — the
		// repack alone does nothing without this call using it.
		ws := matmulWSPool.Get().(*linalg.Workspace)
		defer matmulWSPool.Put(ws)
		ws.SetThreshold(int4ParThreshold)
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	if q8, scales, w8a8, ok := w.Int8(); ok {
		if w8a8 {
			if qb, ok := be.(QuantBackend); ok && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
				return
			}
			// Pooled Workspace with the int8 decode threshold — the free-matmul path
			// (e.g. gemma4's own forward) has no scratch Workspace, so without this its
			// W8A8 decode matmuls would run at aikit's conservative 16.78M default. Same
			// mechanism as the int4 branch above. matmulInto() gets this via the
			// decodeScratch Workspace instead. Threshold differs from int4's (300K vs
			// 1<<20) — the crossover is kernel- and model-specific, measured separately;
			// see DefaultDecodeParallelThreshold + int4ParThreshold.
			ws := matmulWSPool.Get().(*linalg.Workspace)
			defer matmulWSPool.Put(ws)
			ws.SetThreshold(DefaultDecodeParallelThreshold)
			linalg.MatmulBTW8A8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
			return
		}
		linalg.MatmulBTQ8(a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	f32, _ := w.F32()
	be.MatmulBT(a, f32, dst, M, w.Cols(), w.Rows())
}

// matmulInto is matmul using the caller's Workspace, so steady-state decode quantizes the activation
// once into reusable scratch instead of allocating per call.
//
// P7 — this used to dispatch on `isW8A8(w)` and send EVERYTHING ELSE to matmul, which allocates a
// fresh Workspace. So W4A8 never reached the per-stream Workspace its six call sites already hand
// in, purely because the dispatch named one quantization instead of asking the question it meant.
// That is the DISPATCH form of sibling drift (see parity-coverage-policy.md): a check that names one
// member fails to CATCH divergences, a dispatch that names one member CREATES them.
//
// The question it meant is "does this weight have an Into form that takes a Workspace", so that is
// what it asks now. Adding a third such quantization needs a case here and nothing else.
//
// Race-freedom is unchanged and does not need a new argument: `ws` is the per-stream Workspace on
// decodeScratch, and "a cache is one generation stream, so the buffers are never shared
// concurrently" (decoder/scratch.go). The per-call Workspace in matmul stays exactly as it was, for
// callers that have no scratch at all.
func matmulInto(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if isW8A8(w) {
		q8, scales, _, _ := w.Int8()
		if qb, ok := be.(QuantBackend); ok && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
			return
		}
		linalg.MatmulBTW8A8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	if _, _, _, ok := w.Int4(); ok {
		// Same threshold matmul's fresh Workspace sets — the point of the reuse is to stop
		// allocating one per projection per token, not to change how the work is fanned out.
		// w.MatmulBTW4A8Into, not the raw free function — see matmul's own comment above.
		ws.SetThreshold(int4ParThreshold)
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	matmul(be, w, a, dst, M)
}

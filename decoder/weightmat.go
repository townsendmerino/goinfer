package decoder

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// Weight matrices are linalg.WeightMat (aikit): one type that hides f32, per-row int8 and group-wise int4 storage behind
// uniform accessors and the linalg kernels. goinfer keeps the model policy here (which table gets which precision, the
// int4 group size, the matmul backend routing) while the storage wrapper, quantize primitives and Row dequant live in
// linalg.

// quantMode selects the resident weight precision the loader streams into (see
// loadWeights). The f32 path keeps the widened weights; int8 is per-row symmetric
// (¼ f32); int4 is group-wise symmetric (~⅛ f32).
type quantMode uint8

const (
	quantNone quantMode = iota
	quantInt8
	quantInt4
	quantInt8I8 // weights int8 (as quantInt8) but matmul is full int8×int8 (W8A8)
	// quantInt4Mix is a per-tensor mixed mode: attention (and embed/head/router) at int8, the FFN bulk
	// (gate/up/down/experts) at int4. It is a load-time policy only: matmulQuant resolves it per tensor, so the resident
	// weights and the .giw never carry quantInt4Mix itself. GGUF load path only.
	quantInt4Mix
	// quantQ4K keeps every tensor the GGUF stores as Q4_K in its own super-block layout
	// (linalg.WrapQ4K, exact, 4.5 bpw) and puts every other layer matmul at int8 W8A8, with per-32
	// activations throughout (docs/tasks/task-int4-weight-quality-2026-09.md: the re-quantization
	// of Q4_K into symmetric int4 is the int4 quality loss). Like quantInt4Mix it is a LOAD-TIME
	// policy resolved per tensor at the GGUF load sites. GGUF only; no .giw form yet.
	quantQ4K
)

// matmulQuant resolves a matmul tensor's resident precision under the base quant.
// Uniform for every mode except the mixed mode (quantInt4Mix), which keeps the
// (cheap, sensitive) attention tensors at int8 and the (large, int4-tolerant) FFN
// tensors at int4 — keyed off llama.cpp's tensor names (ffn_* vs attn_*).
func matmulQuant(base quantMode, name string) quantMode {
	if base == quantQ4K && strings.Contains(name, "ffn_gate_inp") {
		return quantInt8I8 // the router, as in int4mix below: never the Q4_K kind
	}
	if base != quantInt4Mix {
		return base
	}
	// The router (ffn_gate_inp) must stay out of the "ffn_" bulk though its name contains that prefix: top-k selection is
	// discrete, so quantizing it flips which experts win rather than adding rounding noise. Current call sites send the router
	// through streamMat(..., quantNone, ...) and never reach matmulQuant; this guards a future family that adds a router
	// through the generic mat() helper, which would then get int8 rather than int4.
	if strings.Contains(name, "ffn_gate_inp") {
		return quantInt8
	}
	if strings.Contains(name, "ffn_") {
		return quantInt4
	}
	return quantInt8
}

// embedding returns the precision for the token-embedding table and the LM head, tied or not. In the int4 modes it is
// full W8A8 (int8 weights and int8 activations), not weight-only Q8: the LM head on weight-only Q8 was the largest
// per-token cost in int4 decode. It is not int4 either: int4-weight quantizing these tensors flips the argmax and tanks
// the cosine (GGUF Q4_K_M likewise keeps token_embd/output above 4 bits). The pin is the unconditional int4-mode default.
// int8 and f32 modes use themselves. Evidence: docs/code-notes/decoder.md#quantMode.embedding.
func (q quantMode) embedding() quantMode {
	if q == quantInt4 || q == quantInt4Mix || q == quantQ4K {
		return quantInt8I8
	}
	return q
}

// embeddingWith resolves the embed/head precision, allowing the int8 pin to be relaxed to int4 (Options.EmbedInt4): in
// int4 mode the table goes int4 too, halving the largest resident tensor on a big-vocab small model. Lossy and opt-in;
// off (the pin) is the default and the bit-exact path.
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

// int4ParThreshold sets the fan-out threshold for the int4 (W4A8) matmul below aikit's default (parThreshold = 1<<24 MACs)
// so the small int4 decode matmuls parallelize. At M=1 every Gemma-4 int4 matmul (expert gate||up 3.96M, down 1.98M,
// dense ~5.9M, attention ~11.5M MACs) falls under the aikit default and would run serial. 1<<20 sits below the smallest
// decode matmul (1.98M), so all of them fan out, while tiny ops (<1M) stay serial; threshold 0, which fans out everything,
// over-parallelizes on the 8-core Ryzen.
//
// Output is byte-identical (aikit partitions output columns in 8-wide groups, the width-invariant contract), and it only
// widens fan-out, so prefill's already-parallel large-M matmuls are unaffected. Re-run BenchmarkInt4ParThresholdSweep if
// the core topology or aikit's kernel changes. Measurements on both rigs: docs/code-notes/decoder.md#int4ParThreshold.
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
	return streamQuantizedRow4(rows, cols, mode, true, rowInto)
}

// streamQuantizedSkipRow4 is streamQuantized with the arm64 row4 side-copy skipped, for a tensor reached only through
// streamQuantized's non-batched callers (o_proj/down_proj/router via streamMat, MoE experts via stackedExperts, gpt-oss's
// expert gate/up/down, fused-tensor splits via fusedSplit). It is the GGUF/streaming twin of quantizeWMSkipRow4, with the
// same scope: canonical alone, no needCanonical branch.
func streamQuantizedSkipRow4(rows, cols int, mode quantMode, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return streamQuantizedRow4(rows, cols, mode, false, rowInto)
}

// streamQuantizedRow4 is streamQuantized/streamQuantizedSkipRow4's shared body; row4 selects
// which of repackW4A8IfEligible (canonical+row4, today's default) or
// repackW4A8IfEligibleSkipRow4 (canonical alone) runs the int4 case — every other quant mode is
// row4-independent (row4 is int4-only).
func streamQuantizedRow4(rows, cols int, mode quantMode, row4 bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	scratch := make([]float32, cols)
	if mode == quantQ4K {
		// Reached only where a load site has no raw Q4_K bytes to wrap (a non-Q4_K tensor, or a site
		// q4kMat does not serve): the rest of quantQ4K is int8 W8A8.
		mode = quantInt8I8
	}
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
		q4s := make([]uint16, rows*nGroups)
		rowScales := make([]float32, nGroups)
		for r := range rows {
			if err := rowInto(r, scratch); err != nil {
				return linalg.WeightMat{}, err
			}
			linalg.QuantizeGroupInt4Row(scratch, cols, group, q4[r*bpr:(r+1)*bpr], rowScales)
			linalg.F32ToF16Slice(q4s[r*nGroups:(r+1)*nGroups], rowScales) // int4 scales are stored as binary16
		}
		canon := linalg.WrapInt4F16(q4, q4s, rows, cols, group)
		if row4 {
			return repackW4A8IfEligible(canon), nil
		}
		return repackW4A8IfEligibleSkipRow4(canon), nil
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

// streamQuantizedRepackable is streamQuantized's twin for the GGUF streaming path, shared by streamQuantizedEmbed and
// streamQuantizedBatchedProj; the policy and each caller's safety argument are on quantizeEmbedWM,
// quantizeBatchedProjWM and repackedOnlyOrCanonical. The row-quantize loop is the same whatever the final layout; the
// twin differs only in the last step for quantInt4, where the repack decision goes through repackedOnlyOrCanonical
// (row4-only, or canonical optionally row4-skipped) instead of streamQuantized's fixed canonical plus row4.
func streamQuantizedRepackable(rows, cols int, mode quantMode, needCanonical, skipRow4 bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	if mode != quantInt4 || fakeQuantScheme != "" {
		return streamQuantized(rows, cols, mode, rowInto)
	}
	scratch := make([]float32, cols)
	const group = int4GroupSize
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
	canon := linalg.WrapInt4F16(q4, q4s, rows, cols, group)
	return repackedOnlyOrCanonical(canon, needCanonical, skipRow4), nil
}

// streamQuantizedEmbed is streamQuantizedRepackable for Embed/LMHead (the safety argument is quantizeEmbedWM's). skipRow4
// is always false: Embed/LMHead are read via .Row() on the host on every backend, so the row4-skip trade (a slower CPU
// fallback for less resident memory) is not this class's call; quantizeEmbedWM makes the same choice.
func streamQuantizedEmbed(rows, cols int, mode quantMode, needCanonical bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return streamQuantizedRepackable(rows, cols, mode, needCanonical, false, rowInto)
}

// streamQuantizedBatchedProj is streamQuantizedRepackable for attention Q/K/V and MLP gate/up; the batch-path safety
// argument is quantizeBatchedProjWM's.
func streamQuantizedBatchedProj(rows, cols int, mode quantMode, needCanonical, skipRow4 bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return streamQuantizedRepackable(rows, cols, mode, needCanonical, skipRow4, rowInto)
}

// quantizeWM returns w streamed to the requested resident precision; the f32 memory is freed by dropping the old
// reference at the call site. No-op for quantNone or if w is not f32-resident (already quantized, or empty).
func quantizeWM(w linalg.WeightMat, mode quantMode) linalg.WeightMat {
	return quantizeWMRow4(w, mode, true)
}

// quantizeWMSkipRow4 is quantizeWM with the arm64 row4 side-copy skipped, for a projection reached only through
// quantizeWM's family-specific call sites (o_proj, down_proj, router, MoE experts, shared expert: everything
// isBatchedProjTensor does not cover). Unlike quantizeBatchedProjWM and quantizeEmbedWM it has no needCanonical branch:
// repacked-only is the CPU-batched-dispatch trade repackedOnlyOrCanonical exists for, and none of these tensor classes
// go through that dispatch. It only chooses between canonical plus row4 and canonical alone.
func quantizeWMSkipRow4(w linalg.WeightMat, mode quantMode) linalg.WeightMat {
	return quantizeWMRow4(w, mode, false)
}

// quantizeWMRow4 is the shared body of quantizeWM and quantizeWMSkipRow4: row4 selects repackW4A8IfEligible (canonical
// plus row4) or repackW4A8IfEligibleSkipRow4 (canonical alone) for the int4 case. Every other quant mode ignores row4.
func quantizeWMRow4(w linalg.WeightMat, mode quantMode, row4 bool) linalg.WeightMat {
	f32, ok := w.F32()
	if !ok {
		return w
	}
	if mode == quantQ4K {
		mode = quantInt8I8 // no raw Q4_K bytes here: see streamQuantizedRow4
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
		canon := linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize)
		if row4 {
			return repackW4A8IfEligible(canon)
		}
		return repackW4A8IfEligibleSkipRow4(canon)
	default:
		return w
	}
}

// repackW4A8Row4IfEligible opts wm into the arm64 split-half, 4-row-interleaved W4A8 layout by calling
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
func repackW4A8Row4IfEligible(wm linalg.WeightMat) linalg.WeightMat {
	if !w4a8Row4RepackEnabled {
		return wm
	}
	wm.RepackInt4Row4()
	return wm
}

// w4a8Row4RepackEnabled is a load-time-measurement toggle only: production never sets it, so it stays true in a real
// build. A test measuring the repack's load-time and resident-memory delta flips it off to get a baseline from the same
// load path.
var w4a8Row4RepackEnabled = true

// repackW4A8SplitHalfIfEligible is the amd64 counterpart to repackW4A8Row4IfEligible: it opts wm into the split-half W4A8
// nibble layout (byte i holds weight i's low nibble and weight i+16's high nibble, which removes the AVX2 kernel's two
// per-group VPUNPCK{L,H}BW). A no-op on non-int4 WeightMats, non-amd64 builds, CPUs without AVX2 and any shape the repack
// rejects, so always safe to call unconditionally.
//
// Also a no-op on hosts with AVX-512 VNNI: aikit's canonical W4A8 dot prefers its VNNI tier there and split-half exists
// only at AVX2, so the layout would swap a faster kernel for a slower one. The repack therefore applies to
// AVX2-without-VNNI hosts only.
//
// Wired into the same two call sites as the row4 repack and for the same reason (see that function for why the .giw
// loader is excluded). The repack allocates a second buffer and never writes through the canonical bytes, which for a
// .giw kind=3 tensor are a zero-copy mmap alias of the file; rewriting them in place would silently misdecode every
// existing bundle (aikit's TestWeightMatSplitHalf_canonicalUntouched pins that it does not).
//
// Memory: a second copy of every eligible tensor's nibbles, with canonical kept. The cost and why it is default-off live
// on w4a8SplitHalfRepackEnabled, in one place so the figures cannot drift apart.
func repackW4A8SplitHalfIfEligible(wm linalg.WeightMat) linalg.WeightMat {
	if !w4a8SplitHalfRepackEnabled {
		return wm
	}
	if wm.RepackInt4SplitHalf() {
		w4a8SplitHalfRepacked.Add(1)
		// The second copy's actual cost, asked of the layout's owner rather than re-derived from rows x ceil(cols/2): the memory
		// half of the trade must be a computed quantity, and duplicating aikit's arithmetic would let it drift.
		w4a8SplitHalfBytes.Add(int64(wm.SplitHalfBytes()))
	} else {
		w4a8SplitHalfSkipped.Add(1)
	}
	return wm
}

// w4a8SplitHalfRepacked, w4a8SplitHalfSkipped and w4a8SplitHalfBytes count what the repack did, at load only (one atomic
// add per weight tensor). They exist because the repack is otherwise silent: it returns a bool nobody reads, and a wiring
// that repacked nothing (wrong quant, wrong load path, a shape rule rejecting every tensor) would yield a benchmark that
// measures no difference and gets recorded as flat. An A/B against this repack must read these first and confirm the
// repacked count is non-zero.
var w4a8SplitHalfRepacked, w4a8SplitHalfSkipped, w4a8SplitHalfBytes atomic.Int64

// w4a8SplitHalfRepackEnabled is default-off by decision, not caution; GOINFER_W4A8_SPLITHALF=1 opts in. The repack is a
// second copy of every eligible tensor's nibbles (+0.5 byte per weight on top of an int4 tensor's 0.625, about +80% int4
// weight bytes), and canonical is never dropped (M>1 prefill and every non-AVX2 path read it). Its decode gain landed in
// the band the pre-registration named in advance as ambiguous, so it is parked with the code and wiring intact.
//
// Turning it on is defensible where decode latency outranks resident memory, on amd64 with AVX2 and no AVX-512 VNNI
// (aikit declines on VNNI hosts). It is not defensible as a default. Re-open the decision if the kernel gets faster or
// canonical can be dropped for a decode-only build. The A/B and the figures:
// docs/measurements/w4a8-splithalf-decode-ab-PREREGISTERED.md and docs/code-notes/decoder.md#w4a8SplitHalfRepackEnabled.
var w4a8SplitHalfRepackEnabled = os.Getenv("GOINFER_W4A8_SPLITHALF") != ""

// w4a8BatchEnabled runs a layer's q/k/v (and gate/up, where the fused gate+up does not take them) as one W4A8 fork/join
// instead of one per projection (aikit MatmulBTW4A8Batch). Bit-identical: every output column is the same dot product
// either way. The default is per architecture (w4a8BatchDefault, cpu_tuning_{arm64,other}.go); GOINFER_W4A8_BATCH=0
// opts out and =1 forces it on. Why the defaults differ: docs/code-notes/decoder.md#w4a8BatchEnabled.
var w4a8BatchEnabled = envBoolDefault("GOINFER_W4A8_BATCH", w4a8BatchDefault)

// repackW4A8IfEligible applies whichever ISA-specific W4A8 layout THIS build
// has a kernel for: row4 on arm64, split-half on amd64. Each is a no-op off its
// own architecture, so both are called unconditionally and the two stay
// symmetric — a third layout gets added here and nowhere else.
func repackW4A8IfEligible(wm linalg.WeightMat) linalg.WeightMat {
	return repackW4A8SplitHalfIfEligible(repackW4A8Row4IfEligible(wm))
}

// repackW4A8IfEligibleSkipRow4 is repackW4A8IfEligible's twin for a backend committed to GPU residency that will never
// read the arm64 row4 layout (see wantsRow4Fallback): it applies only the amd64 split-half repack and skips row4.
func repackW4A8IfEligibleSkipRow4(wm linalg.WeightMat) linalg.WeightMat {
	return repackW4A8SplitHalfIfEligible(wm)
}

// wantsCanonicalInt4 reports whether this load might need canonical int4 bytes (packed nibbles + scales) for a tensor,
// that is, whether the repacked-only layout is unsafe. Two consumers read them directly rather than through WeightMat's
// layout-agnostic methods:
//
//   - The staged per-token GPU consult (QuantBackend4.MatmulW4A8, QuantBatchBackend4.MatmulW4A8Batch in matmul,
//     matmulInto, matmulW4A8Batch) hands q4/q4s to the backend on every call. A repacked-only tensor has nothing to hand
//     it, and rebuilding canonical per token is the wrong shape for a per-call path.
//   - A resident GPU build (ResidencyBackend.BuildResident) reads w.Int4() directly to upload once at build time.
//
// backendName is Options.Backend as the caller wrote it, not be's resolved capabilities. Repacked-only is a promise that
// no GPU backend will ever touch this *Model, resident or staged, and a promise must be stated, never inferred from an
// omission: only the literal "cpu" states it. An empty name promises nothing, because a generic load may later be handed
// to any backend's resident-build machinery outside decoder.Load's dispatch (metal's tests do exactly this), and keying
// on be's interfaces alone sees the plain CPU backend, which implements none of them, and calls it safe.
//
// The interface assertions are a second guard for the "cpu" case itself: a "cpu" backend that implemented one of these
// would still decline repacked-only. Every other backendName returns true, so canonical(+row4) stays the default.
func wantsCanonicalInt4(backendName string, be Backend) bool {
	if backendName != "cpu" {
		return true
	}
	if _, ok := be.(QuantBackend4); ok {
		return true
	}
	if _, ok := be.(QuantBatchBackend4); ok {
		return true
	}
	_, ok := be.(ResidencyBackend)
	return ok
}

// wantsRow4Fallback reports whether the arm64 row4 repack should be built alongside canonical as a safety net for a CPU
// fallback. Only the CPU's decode kernel reads row4; no GPU backend does. It exists so a model that falls back to CPU
// decode (a declined residency build, LoRA/session paths outside a resident backend's coverage) keeps its speed.
//
// False only for backendName == "metal": Options.Backend is the caller's commitment to that backend (stated, never
// inferred; see wantsCanonicalInt4), and row4 doubles the resident int4 footprint of every projection it applies to, so a
// slower CPU fallback after a declined Metal residency is the price. CUDA and WebGPU residency have not been measured
// against this trade and keep the default. Measurement: docs/code-notes/decoder.md#wantsRow4Fallback.
func wantsRow4Fallback(backendName string) bool {
	return backendName != "metal"
}

// repackedOnlyOrCanonical is the decision shared by quantizeEmbedWM and quantizeBatchedProjWM: given a canonical int4
// WeightMat, build it repacked-only when needCanonical is false and this core and shape can build row4
// (linalg.Int4Row4Usable), which avoids the double nibble+scale residency of canonical+row4. Otherwise (needCanonical,
// no row4 on this core, or a shape that does not qualify) it applies repackW4A8IfEligible's policy, what every other int4
// tensor gets from quantizeWM, so it is never worse than the default.
//
// amd64 split-half repacked-only is out of scope: split-half is parked (w4a8SplitHalfRepackEnabled, default off).
//
// skipRow4 is consulted only when needCanonical: the other branch is the CPU-only repacked-only path row4 exists for.
// Callers that must never skip row4 (Embed/LMHead) pass false.
func repackedOnlyOrCanonical(canon linalg.WeightMat, needCanonical, skipRow4 bool) linalg.WeightMat {
	if needCanonical {
		if skipRow4 {
			return repackW4A8IfEligibleSkipRow4(canon)
		}
		return repackW4A8IfEligible(canon)
	}
	q4, q4s, group, ok := canon.Int4F16()
	if !ok || !linalg.Int4Row4Usable(canon.Rows(), canon.Cols(), group) {
		return repackW4A8IfEligible(canon) // this core/shape can't do row4 at all — existing policy
	}
	if repacked, repOK := linalg.RepackInt4Row4InPlaceF16(q4, q4s, canon.Rows(), canon.Cols(), group); repOK {
		return repacked
	}
	return repackW4A8IfEligible(canon) // Int4Row4Usable said yes, so this shouldn't miss — no silent data loss either way
}

// quantizeEmbedWM is quantizeWM's Embed/LMHead twin; the policy is repackedOnlyOrCanonical. needCanonical is computed once
// per Load (wantsCanonicalInt4) and threaded down. Embed/LMHead are read via WeightMat.Row() (layout-independent) and
// driven through matmul/matmulInto as the LM head (single-op dispatch gated on IsInt4), never through the batched q/k/v
// or gate/up dispatch. It stays separate from quantizeBatchedProjWM although the bodies match: the two safety arguments
// are independent, and a change to one dispatch shape must not silently start covering the other tensor class.
func quantizeEmbedWM(w linalg.WeightMat, mode quantMode, needCanonical bool) linalg.WeightMat {
	if mode != quantInt4 || fakeQuantScheme != "" {
		return quantizeWM(w, mode)
	}
	f32, ok := w.F32()
	if !ok {
		return quantizeWM(w, mode) // already quantized, or empty — quantizeWM's own no-op path
	}
	canon := linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize)
	// skipRow4 is false: Embed/LMHead are read via .Row() on the host on every backend, so the row4-skip trade is not this
	// class's call (streamQuantizedEmbed makes the same choice).
	return repackedOnlyOrCanonical(canon, needCanonical, false)
}

// quantizeBatchedProjWM is quantizeWM's twin for attention Q/K/V and MLP gate/up, the tensors reached through the
// batched W4A8 dispatch (matmulW4A8Batch, wmW4A8Op, isW4A8). That dispatch is unsafe for a repacked-only op in general:
// MatmulBTW4A8Batch has no row4 tile, so a repacked-only op panics if it is reached at M>1 or when a fan-out shard
// boundary splits one of its quads (N%4 != 0 at the boundary).
//
// It is safe for the two batch call sites, causalAttention and gatedMLP: both are decode-only and call matmulW4A8Batch
// with a literal M=1, while prefill uses matmul/matmulInto per projection, which are safe at any M. wmW4A8Op builds the
// right op shape for a repacked-only tensor and isW4A8's IsInt4 gate lets it through. The N%4==0 condition is
// linalg.Int4Row4Usable's rows%4 check, applied per tensor by repackedOnlyOrCanonical, so a violating shape is never
// built repacked-only.
//
// Down-proj, the router and MoE expert weights are deliberately not covered: down-proj has not been audited against the
// batch-path constraint, and expert weights read through a read-only mmap span (paged .giw loading) have no load-time
// repack, so they stay canonical-only. skipRow4 is scoped to this class too; those other projections take the row4 skip
// through quantizeWMSkipRow4.
func quantizeBatchedProjWM(w linalg.WeightMat, mode quantMode, needCanonical, skipRow4 bool) linalg.WeightMat {
	if mode != quantInt4 || fakeQuantScheme != "" {
		return quantizeWM(w, mode)
	}
	f32, ok := w.F32()
	if !ok {
		return quantizeWM(w, mode)
	}
	canon := linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize)
	return repackedOnlyOrCanonical(canon, needCanonical, skipRow4)
}

// isBatchedProjTensor reports whether name (llama.cpp's GGUF tensor-name convention) is attention Q/K/V or MLP gate/up,
// the groups quantizeBatchedProjWM covers. It suffix-matches the exact standard names, not a substring, so an MoE expert
// ("ffn_gate_exps.weight") or router ("ffn_gate_inp.weight") never matches, and a family-specific split projection (an
// MLA q_a_proj/q_b_proj pair) stays on the quantizeWM/streamQuantized path with no per-family audit.
func isBatchedProjTensor(name string) bool {
	for _, suf := range [...]string{"attn_q.weight", "attn_k.weight", "attn_v.weight", "ffn_gate.weight", "ffn_up.weight"} {
		if strings.HasSuffix(name, suf) {
			return true
		}
	}
	return false
}

// GIWTarget names the single consumer a .giw bundle (or one cmd/prequant run) is built for, so the writer can choose the
// one on-disk int4 layout that consumer reads. GIWTargetNone ("") means unknown or multi-consumer and keeps every int4
// tensor canonical (kind 3), the safe default: a bundle nobody has promised to a single reader must stay portable, as
// wantsCanonicalInt4 never infers a promise from an omission.
type GIWTarget string

const (
	GIWTargetNone     GIWTarget = ""
	GIWTargetCPUArm64 GIWTarget = "cpu-arm64"
	GIWTargetCPUAmd64 GIWTarget = "cpu-amd64" // defined for L5; does not yet change what's written
	GIWTargetMetal    GIWTarget = "metal"
	GIWTargetCUDA     GIWTarget = "cuda"
	GIWTargetWebGPU   GIWTarget = "webgpu"
)

// GIWTargetForBackend derives the GIWTarget for a caller naming backendName as the
// ONE consumer about to read a .giw bundle being built now (EnsureCachedGIW,
// cmd/prequant's -target default). Mirrors wantsCanonicalInt4's literal-"cpu"-is-
// a-promise rule: only backendName == "cpu" unlocks an arch-specific repacked-only
// target, and only for THIS process's own GOARCH — the box building the bundle now
// is the box that will read a same-machine cache, never a claim about some other
// reader. Any GPU name maps to its own always-canonical target (visible in the
// cache key even though it does not yet change which kind gets written); an
// unrecognized name returns GIWTargetNone, the safe default.
func GIWTargetForBackend(backendName string) GIWTarget {
	switch backendName {
	case "cpu":
		switch runtime.GOARCH {
		case "arm64":
			// A kind-5 (row4-only) file promises that this core can read it, and aikit's row4 kernels need DotProd
			// (Int4Row4Usable). A core without it (a Raspberry Pi 4, Windows on ARM, where aikit assumes none) would refuse to load
			// its own cpu-arm64 file and rebuild it on every start, so it builds canonical bundles, which any core loads. A
			// cpu-arm64 bundle can still be built for a DotProd reader by naming the target (-target cpu-arm64).
			if linalg.Int4Row4Usable(4, 32, 32) {
				return GIWTargetCPUArm64
			}
			return GIWTargetNone
		case "amd64":
			return GIWTargetCPUAmd64
		default:
			return GIWTargetNone
		}
	case "metal":
		return GIWTargetMetal
	case "cuda":
		return GIWTargetCUDA
	case "webgpu":
		return GIWTargetWebGPU
	default:
		return GIWTargetNone
	}
}

// ParseGIWTarget parses a -target flag value: one of the five named GIWTarget
// values, "cpu" (resolved via this process's own GOARCH, like
// GIWTargetForBackend("cpu")), "canonical" (GIWTargetNone, an explicit "no
// target" for a bundle more than one consumer/arch will read — e.g. one .giw
// embedded into several cross-compiled release binaries, where no single
// arch-specific repacked-only layout is safe for all of them), or "" (defaults
// the same as "cpu": cmd/prequant's traditional single-box use).
func ParseGIWTarget(s string) (GIWTarget, error) {
	switch GIWTarget(s) {
	case GIWTargetCPUArm64, GIWTargetCPUAmd64, GIWTargetMetal, GIWTargetCUDA, GIWTargetWebGPU:
		return GIWTarget(s), nil
	}
	switch s {
	case "", "cpu":
		return GIWTargetForBackend("cpu"), nil
	case "canonical":
		return GIWTargetNone, nil
	}
	return GIWTargetNone, fmt.Errorf("unknown .giw target %q (want cpu-arm64 | cpu-amd64 | metal | cuda | webgpu | cpu | canonical)", s)
}

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

// isW4A8 reports whether w is int4-resident, the only precision with a batched dispatch on the W4A8 path. It uses
// IsInt4(), not Int4()'s narrower "canonical bytes present": a repacked-only WeightMat is still int4 and routes here, and
// wmW4A8Op builds the matching op shape for it.
func isW4A8(w *linalg.WeightMat) bool {
	return w.IsInt4()
}

// Int4F32 returns w's canonical int4 nibbles and its binary16 group scales widened, exactly, to f32 in a NEW
// slice — for load-time consumers written against f32 scales (GPU uploads, the .giw writer). ok and group are
// linalg.WeightMat.Int4F16's. It allocates, so it is not for a per-token path; decode reads Int4F16.
func Int4F32(w *linalg.WeightMat) (q4 []byte, scales []float32, group int, ok bool) {
	q4, s16, group, ok := w.Int4F16()
	if s16 != nil {
		scales = make([]float32, len(s16))
		linalg.F16ToF32Slice(scales, s16)
	}
	return q4, scales, group, ok
}

// wmW4A8Op builds one linalg.W4A8Op for the batched W4A8 dispatch. Canonical nibbles and scales are present unless the
// tensor is repacked-only (then W4 is nil and Row4 carries the data). Row4/Row4Scales are populated only after
// RepackInt4Row4 has run (arm64, heap-backed weights; see repackW4A8Row4IfEligible) and are nil otherwise, which
// linalg.MatmulBTW4A8Batch reads as "run canonical for this op". group is shared across the batch, as in
// MatmulBTW4A8Into: every op in one call comes from the same layer's quant config.
func wmW4A8Op(w *linalg.WeightMat, dst []float32) (op linalg.W4A8Op, group int) {
	q4, q4s, group, _ := w.Int4F16()
	row4, row4s, _ := w.Int4Row4F16()
	return linalg.W4A8Op{W4: q4, ScalesF16: q4s, Row4: row4, Row4ScalesF16: row4s, Dst: dst, N: w.Rows()}, group
}

// matmulWSPool recycles the Workspace matmul falls back to when the caller has no decodeScratch. A fresh Workspace per
// call would reallocate its quant buffers on every matmul. sync.Pool's GC-driven eviction keeps a workspace that saw one
// huge call (the vocab-sized LM head) from pinning that size forever.
var matmulWSPool = sync.Pool{New: func() any { return new(linalg.Workspace) }}

// w4a8PreOff (tests only) makes prepare leave every block unset, so each matmul quantizes its own input, for the gate
// that compares the two paths and the A/B that times them. w4a8PreCalls counts the matmuls that ran on a shared block, so
// a test can see the path was taken.
var (
	w4a8PreOff   = false
	w4a8PreCalls atomic.Int64
)

// w4a8Act is one activation quantized once for several CPU W4A8 matmuls over it: q, k and v of one normed row, gate and
// up, every expert of a MoE layer. Quantization is deterministic, so repeating it per call was dead work. prepare
// quantizes exactly as matmul's CPU W4A8 path would, and matmulPre runs aikit's Pre entry on it, so the result is the bits
// matmul gives. A weight the block does not fit (another K, another activation group, a backend that may take the call)
// runs matmul instead.
type w4a8Act struct {
	q        linalg.ActQ
	set      bool
	group, M int
}

// cpuW4A8 reports whether matmul runs w on the CPU W4A8 path for certain: an int4 weight no QuantBackend4 can claim
// (matmul offers a per-row-scale int4 matmul to one first).
func cpuW4A8(be Backend, w *linalg.WeightMat) bool {
	if !w.IsInt4() {
		return false
	}
	if _, ok := be.(QuantBackend4); ok && w.ActQuantGroup() == 0 {
		return false
	}
	return true
}

// prepare quantizes a (M rows of w.Cols()) for w and every weight sharing its K and activation group, or clears the
// block when w does not run the CPU W4A8 path.
func (p *w4a8Act) prepare(be Backend, w *linalg.WeightMat, a []float32, M int) {
	p.set = false
	if w4a8PreOff || !cpuW4A8(be, w) {
		return
	}
	ws := matmulWSPool.Get().(*linalg.Workspace)
	defer matmulWSPool.Put(ws)
	ws.SetActQuantGroup(w.ActQuantGroup())
	w.QuantizeActW4A8(ws, a, M, &p.q)
	p.set, p.group, p.M = true, w.ActQuantGroup(), M
}

// matmulPre is matmul(be, w, a, dst, M), using p's block when it was quantized for an input w can take.
func matmulPre(be Backend, w *linalg.WeightMat, p *w4a8Act, a, dst []float32, M int) {
	if p == nil || !p.set || p.M != M || p.q.K != w.Cols() || p.group != w.ActQuantGroup() || !cpuW4A8(be, w) {
		matmul(be, w, a, dst, M)
		return
	}
	ws := matmulWSPool.Get().(*linalg.Workspace)
	defer matmulWSPool.Put(ws)
	ws.SetThreshold(int4ParThreshold)
	ws.SetActQuantGroup(w.ActQuantGroup())
	w.MatmulBTW4A8PreInto(ws, &p.q, dst, M)
	w4a8PreCalls.Add(1)
}

// matmul computes dst[M, rows] = a[M, cols] . w^T, dispatching on w's precision with goinfer's backend routing: the f32,
// W8A8 and W4A8 paths can run on a GPU backend (be.MatmulBT, QuantBackend.MatmulW8A8, QuantBackend4.MatmulW4A8);
// weight-only int8 (Q8) stays on the CPU.
func matmul(be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if w.IsInt4() {
		// Staged int4 backend consult, mirrored in matmulInto and the W8A8 branch below. Nested under Int4()'s ok (canonical
		// bytes present), not IsInt4(): the consult hands q4/q4s to the backend on every call, so a repacked-only tensor has
		// nothing to hand it, and rebuilding canonical per token would allocate on a per-call path. It falls through to
		// w.MatmulBTW4A8Into, which dispatches on whichever layout is present. A per-group activation weight (ActQuantGroup)
		// skips the consult: no staged GPU kernel reads per-group activation scales.
		if q4, q4s, group, ok := w.Int4F16(); ok && w.ActQuantGroup() == 0 {
			if qb, ok := be.(QuantBackend4); ok && qb.MatmulW4A8(a, q4, q4s, group, dst, M, w.Cols(), w.Rows()) {
				return
			}
		}
		// int4 weights run the int8-activation W4A8 integer kernel at every M this CPU path reaches. Its per-output result is
		// M-independent, so batched prefill matches sequential decode on this kernel. That does not make the whole matmul call
		// M-independent: the QuantBackend4 consult above takes M=1 on a staged webgpu backend (its MatmulW4A8 declines any
		// M != 1) through a different kernel, so decode and verify run two kernels there. Model.SpecDecodeConflict refuses
		// speculative decoding for that combination; read it before assuming this path covers a device-backed one.
		//
		// The pooled Workspace lowers the fan-out threshold (see int4ParThreshold). Each Get is exclusive to this call, so
		// concurrent decode streams never share one.
		//
		// Use w.MatmulBTW4A8Into, not the raw linalg free function: aikit picks the repacked layout (arm64 row4, amd64
		// split-half) inside it, at M=1 only, and the load-time repacks do nothing without this call.
		ws := matmulWSPool.Get().(*linalg.Workspace)
		defer matmulWSPool.Put(ws)
		ws.SetThreshold(int4ParThreshold)
		ws.SetActQuantGroup(w.ActQuantGroup()) // pooled: always set, so a previous model's group never leaks
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	if q8, scales, w8a8, ok := w.Int8(); ok {
		if w8a8 {
			if qb, ok := be.(QuantBackend); ok && w.ActQuantGroup() == 0 && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
				return
			}
			// Pooled Workspace with the int8 decode threshold: the free-matmul path (e.g. gemma4's own forward) has no scratch
			// Workspace, and without this its W8A8 decode matmuls would run at aikit's conservative default. matmulInto gets the same
			// through decodeScratch. The threshold differs from int4's (see DefaultDecodeParallelThreshold and int4ParThreshold).
			ws := matmulWSPool.Get().(*linalg.Workspace)
			defer matmulWSPool.Put(ws)
			ws.SetThreshold(DefaultDecodeParallelThreshold)
			ws.SetActQuantGroup(w.ActQuantGroup())
			linalg.MatmulBTW8A8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
			return
		}
		// Pooled Workspace, as in the W8A8 case: the bare linalg.MatmulBTQ8 wrapper builds a fresh, non-pooled Workspace every
		// call, and the weight-only Q8 LM head is the largest per-token cost on this path.
		ws := matmulWSPool.Get().(*linalg.Workspace)
		defer matmulWSPool.Put(ws)
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		linalg.MatmulBTQ8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	if w.Kind() == "q4k" {
		// Native Q4_K (quantQ4K) runs on the CPU here: there is no staged-backend consult for it. Per-32 activations are built
		// into the kind.
		ws := matmulWSPool.Get().(*linalg.Workspace)
		defer matmulWSPool.Put(ws)
		ws.SetThreshold(int4ParThreshold)
		w.MatmulBTInto(ws, a, dst, M)
		return
	}
	f32, _ := w.F32()
	be.MatmulBT(a, f32, dst, M, w.Cols(), w.Rows())
}

// matmulIntoPre is matmulInto(ws, be, w, a, dst, M), using p's block when it was quantized for an input w can take (see
// w4a8Act).
func matmulIntoPre(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, p *w4a8Act, a, dst []float32, M int) {
	if p == nil || !p.set || p.M != M || p.q.K != w.Cols() || p.group != w.ActQuantGroup() || !cpuW4A8(be, w) {
		matmulInto(ws, be, w, a, dst, M)
		return
	}
	ws.SetActQuantGroup(w.ActQuantGroup())
	ws.SetThreshold(int4ParThreshold)
	w.MatmulBTW4A8PreInto(ws, &p.q, dst, M)
	w4a8PreCalls.Add(1)
}

// matmulInto is matmul using the caller's Workspace, so steady-state decode quantizes the activation once into reusable
// scratch instead of allocating per call. It dispatches on the question "does this weight have an Into form that takes a
// Workspace", not on one quantization's name (a dispatch that names one member creates sibling drift, see
// docs/parity-coverage-policy.md); adding a third such quantization needs a case here and nothing else.
//
// Race-freedom: ws is the per-stream Workspace on decodeScratch, and a cache is one generation stream, so the buffers are
// never shared concurrently (decoder/scratch.go). matmul's per-call Workspace stays for callers with no scratch at all.
func matmulInto(ws *linalg.Workspace, be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if isW8A8(w) {
		q8, scales, _, _ := w.Int8()
		if qb, ok := be.(QuantBackend); ok && w.ActQuantGroup() == 0 && qb.MatmulW8A8(a, q8, scales, dst, M, w.Cols(), w.Rows()) {
			return
		}
		ws.SetActQuantGroup(w.ActQuantGroup())
		linalg.MatmulBTW8A8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	if q8, scales, w8a8, ok := w.Int8(); ok && !w8a8 {
		// Weight-only Q8 (the int8-pinned LM head in int4 mode): thread the caller's own scratch Workspace rather than falling
		// through to matmul's pool round-trip.
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		linalg.MatmulBTQ8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	if w.IsInt4() {
		// Staged int4 backend consult, as in matmul: nested under Int4()'s ok (canonical bytes present), not IsInt4(), because a
		// repacked-only tensor has no canonical bytes to hand a staged backend.
		if q4, q4s, group, ok := w.Int4F16(); ok && w.ActQuantGroup() == 0 {
			if qb, ok := be.(QuantBackend4); ok && qb.MatmulW4A8(a, q4, q4s, group, dst, M, w.Cols(), w.Rows()) {
				return
			}
		}
		ws.SetActQuantGroup(w.ActQuantGroup())
		// Same threshold as matmul's fresh Workspace: reusing one only stops a per-projection, per-token allocation and does not
		// change how the work fans out. w.MatmulBTW4A8Into, not the raw free function (see matmul).
		ws.SetThreshold(int4ParThreshold)
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	matmul(be, w, a, dst, M)
}

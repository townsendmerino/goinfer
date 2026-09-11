package decoder

import (
	"os"
	"strings"
	"sync"
	"sync/atomic"

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
// LM head, tied or not). Full W8A8 (int8 weights AND int8 activations), not
// weight-only Q8 — changed 2026-08-24 (docs/prompts/lmhead-workspace-fix.md Step
// 2, after the W4A8 plumbing phase found the LM head running weight-only Q8 was
// the single largest per-token cost in int4 decode, achieving only 11-13 GB/s
// against W8A8's 97.12 GB/s at the same shape — 7.7x). Precision measured before
// switching, teacher-forced real-continuation comparison against the old
// weight-only-Q8 pin, both real model sizes: 1.5% argmax flip rate, mean cosine
// 0.9998+ (200 positions each, docs/task-w4a8-neon-bandwidth.md). Small and real,
// nowhere near int4-weight quantizing these same tensors ("flips the argmax and
// tanks the cosine" — mirrors why GGUF Q4_K_M keeps token_embd/output at Q6_K
// while the projections go 4-bit) — kept as the unconditional int4-mode default,
// not opt-in, given the size of the win against the size of the cost. int8 and
// f32 modes use themselves, unaffected.
func (q quantMode) embedding() quantMode {
	if q == quantInt4 || q == quantInt4Mix {
		return quantInt8I8
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
		return repackW4A8IfEligible(maybeF16RoundInt4Scales(linalg.WrapInt4(q4, q4s, rows, cols, group))), nil
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

// streamQuantizedRepackable is streamQuantized's twin for the GGUF streaming path, shared by
// streamQuantizedEmbed (Embed/LMHead) and streamQuantizedBatchedProj (attention Q/K/V, MLP
// gate/up) — see decoder/weightmat.go's quantizeEmbedWM/quantizeBatchedProjWM/
// repackedOnlyOrCanonical for the full policy and each caller's own safety argument. The
// row-quantize loop is identical regardless of final layout — the quantized BYTES are the same —
// so this only differs from streamQuantized in the last step for quantInt4: when needCanonical
// is false and the shape/core qualify, it repacks the just-quantized bytes in place
// (linalg.RepackInt4Row4InPlace) instead of wrapping them canonical+row4 "both".
func streamQuantizedRepackable(rows, cols int, mode quantMode, needCanonical bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	if mode != quantInt4 || fakeQuantScheme != "" || needCanonical {
		return streamQuantized(rows, cols, mode, rowInto)
	}
	scratch := make([]float32, cols)
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
	canon := maybeF16RoundInt4Scales(linalg.WrapInt4(q4, q4s, rows, cols, group))
	return repackedOnlyOrCanonical(canon, needCanonical), nil
}

// streamQuantizedEmbed is streamQuantizedRepackable applied to Embed/LMHead — see
// quantizeEmbedWM's own doc comment for why this tensor class is safe.
func streamQuantizedEmbed(rows, cols int, mode quantMode, needCanonical bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return streamQuantizedRepackable(rows, cols, mode, needCanonical, rowInto)
}

// streamQuantizedBatchedProj is streamQuantizedRepackable applied to the attention Q/K/V and MLP
// gate/up projections — see quantizeBatchedProjWM's own doc comment for the batch-path safety
// argument this shares.
func streamQuantizedBatchedProj(rows, cols int, mode quantMode, needCanonical bool, rowInto func(r int, dst []float32) error) (linalg.WeightMat, error) {
	return streamQuantizedRepackable(rows, cols, mode, needCanonical, rowInto)
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
		return repackW4A8IfEligible(maybeF16RoundInt4Scales(linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize))) // GOINFER_INT4_F16_SCALES diagnostic
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

// repackW4A8SplitHalfIfEligible is the amd64 counterpart to
// repackW4A8Row4IfEligible: it opts wm into the split-half W4A8 nibble layout
// (byte i holds weight i's low nibble and weight i+16's high nibble, so the
// AVX2 kernel's two per-group VPUNPCK{L,H}BW disappear — docs/queue-
// performance.md P14 item 3, measured 1.12x hot AND cold on Zen 2). A no-op on
// non-int4 WeightMats, on non-amd64 builds, on CPUs without AVX2, and on any
// shape the repack rejects, so it is always safe to call unconditionally.
//
// ALSO a no-op on hosts WITH AVX-512 VNNI, which is the surprising one: aikit's
// canonical W4A8 dot prefers its VNNI tier there and split-half exists only at
// AVX2, so the layout would swap a faster kernel for a slower one. aikit
// declines rather than pessimize. The consequence here is that this repack —
// and the +2.10% below — applies to AVX2-WITHOUT-VNNI hosts only, which is a
// narrower audience than "amd64".
//
// Wired into exactly the same two call sites as the row4 repack and for the
// same reason — see that function's comment for why the .giw loader is
// deliberately excluded. The constraint is identical here: the repack
// ALLOCATES a second buffer and never writes through the canonical bytes,
// which for a .giw kind=3 tensor are a zero-copy mmap alias of the file.
// Rewriting them in place would silently misdecode every existing bundle, with
// no error and wrong numbers; aikit's TestWeightMatSplitHalf_canonicalUntouched
// pins that it does not.
//
// MEMORY: this is a second copy of every eligible tensor's nibbles, and
// canonical is NOT dropped. The cost, the measurement that priced it, and why
// it is default-off live on w4a8SplitHalfRepackEnabled below — deliberately in
// ONE place, so the figures cannot drift apart from each other.
func repackW4A8SplitHalfIfEligible(wm linalg.WeightMat) linalg.WeightMat {
	if !w4a8SplitHalfRepackEnabled {
		return wm
	}
	if wm.RepackInt4SplitHalf() {
		w4a8SplitHalfRepacked.Add(1)
		// What the second copy actually cost, asked of the layout's owner rather
		// than re-derived from rows x ceil(cols/2) out here — the memory half of
		// this trade has to be a quantity we compute, not one we estimate, and
		// duplicating aikit's arithmetic is how it would drift from the truth.
		w4a8SplitHalfBytes.Add(int64(wm.SplitHalfBytes()))
	} else {
		w4a8SplitHalfSkipped.Add(1)
	}
	return wm
}

// w4a8SplitHalfRepacked / w4a8SplitHalfSkipped count what the repack actually
// did, at LOAD only (one atomic add per weight tensor, never in a hot loop).
//
// They exist because the repack is otherwise SILENT: it returns a bool nobody
// reads, and a wiring that quietly repacked nothing — wrong quant, wrong load
// path, a shape rule that rejects every tensor — would produce a benchmark that
// confidently measures no difference and gets written down as "flat". That is
// the same failure mode loadBenchModel's own quant comment warns about, one
// layer down. An A/B against this repack MUST read these first and confirm the
// repacked count is non-zero, or its result means nothing.
var w4a8SplitHalfRepacked, w4a8SplitHalfSkipped, w4a8SplitHalfBytes atomic.Int64

// w4a8SplitHalfRepackEnabled is DEFAULT-OFF, and that is a measured decision,
// not caution. Set GOINFER_W4A8_SPLITHALF=1 to opt in.
//
// The A/B is recorded in docs/measurements/w4a8-splithalf-decode-ab-
// PREREGISTERED.md: on Qwen2.5-Coder-1.5B at int4, Ryzen 7 3700X, interleaved,
// same binary both arms, the repack is worth **+2.10% decode tok/s** — real
// (floor 0.75%, and the two arms' sample ranges do not overlap at all), but
// short of the +4% that was pre-registered as the bar for accepting its memory
// cost. It landed in the band the pre-registration named in advance as
// AMBIGUOUS -> PARKED, so it parks, with the code and the wiring kept intact.
//
// The cost it is short against: a second copy of every eligible tensor's
// nibbles, +0.5 bytes/weight on top of the 0.625 an int4 tensor already pays,
// so int4 weight bytes grow ~80%. MEASURED on that 1.5B model, not estimated:
// 196 tensors repacked, **+624.8 MiB** of duplicate nibbles, taking its int4
// weights from 781 MiB to 1.37 GiB.
// Canonical is never dropped — M>1 prefill and every non-AVX2 path read it.
//
// Turning this on is defensible where decode latency outranks resident memory
// and the machine is amd64 with AVX2 and NO AVX-512 VNNI (aikit declines on
// VNNI hosts — see above). It is not defensible as a default, which is why it
// is not one. Re-open the decision if the kernel gets faster than
// 1.12x, or if canonical can be dropped for a build that only ever decodes.
var w4a8SplitHalfRepackEnabled = os.Getenv("GOINFER_W4A8_SPLITHALF") != ""

// w4a8BatchEnabled is DEFAULT-OFF, a measured decision (audit R-06). Set
// GOINFER_W4A8_BATCH=1 to opt in.
//
// aikit's MatmulBTW4A8Batch fuses q/k/v (and separately gate/up) into one
// fork/join, amortizing the goroutine-wake stagger S-02 measured as the real
// decode-fan-out cost (docs/task-simd-audit.md). Paired, interleaved,
// run1-discarded-as-warm-up measurement on this box (qwen2.5-coder-1.5b-int4,
// BenchmarkDecode, 10 pairs): mean before 46.122 tok/s, mean after 49.915,
// **1.08x** (per-pair ratios 0.984-1.143). S-02's own pre-registered rule is
// ship at >=1.15x, park below 1.05x; 1.08x lands in neither band. Per this
// repo's own measurement discipline ("pre-register... an explicit ambiguous ->
// parked band... the zone just below the threshold is where motivated
// reasoning lives"), that ambiguous reading parks rather than ships by
// default — the code is correctness-proven (TestInt4_forwardParity,
// mutation-checked) and kept, shovel-ready, not rebuilt from zero if a
// different box or workload clears the bar.
//
// Re-open the decision with a fresh paired measurement — ideally on a
// different day/machine state, per this same repo's own "a single-machine
// result needs to reproduce before a remedy gets built against it" rule
// (docs/task-zeno-compare.md's R-05 saga) — before flipping this default.
var w4a8BatchEnabled = os.Getenv("GOINFER_W4A8_BATCH") != ""

// repackW4A8IfEligible applies whichever ISA-specific W4A8 layout THIS build
// has a kernel for: row4 on arm64, split-half on amd64. Each is a no-op off its
// own architecture, so both are called unconditionally and the two stay
// symmetric — a third layout gets added here and nowhere else.
func repackW4A8IfEligible(wm linalg.WeightMat) linalg.WeightMat {
	return repackW4A8SplitHalfIfEligible(repackW4A8Row4IfEligible(wm))
}

// wantsCanonicalInt4 reports whether this load might need canonical int4 bytes (packed nibbles +
// scales) for a tensor somewhere in its lifetime — the two consumers that read them directly
// rather than through WeightMat's own layout-agnostic methods:
//
//   - The staged per-token GPU consult (QuantBackend4.MatmulW4A8 / QuantBatchBackend4.
//     MatmulW4A8Batch in matmul()/matmulInto()/matmulW4A8Batch): hands q4/q4s to the backend on
//     EVERY call, no persistent upload. A repacked-only tensor has nothing to hand it —
//     reconstructing canonical on demand there would cost a canonical-sized allocation per
//     token, the wrong shape for a per-call path, not merely a deferred optimization.
//   - A resident GPU build (ResidencyBackend.BuildResident — cuda/resident.go, metal/model.go,
//     gpu/residency.go all read w.Int4() directly to upload a tensor once at build time).
//
// backendName is Options.Backend AS THE CALLER WROTE IT, not be's resolved capabilities — this
// is the load of the decision: repacked-only is a PROMISE ("no GPU backend will ever touch this
// *Model, resident or staged") that must be STATED, never INFERRED from an omission. Only the
// literal "cpu" states it. Empty/unspecified is not a promise of anything — it is simply what a
// caller wrote before deciding, or a generic load a DIFFERENT package's code may later hand to
// ANY backend's own resident-build machinery outside decoder.Load's own dispatch entirely (this
// is not hypothetical: metal's own test suite does exactly this at ~87 call sites — load
// generically, then call metal.buildResident on the result directly, which decoder.Load has no
// way to see coming). Getting this wrong previously (keying on be's interfaces alone, which are
// only known for the backend the CALLER already named) broke exactly that pattern: a model
// loaded with Backend unset resolves to the plain CPU backend, which implements none of the
// interfaces below, so the interface-only check said "safe" — and a later, out-of-band Metal
// residency attempt on that same model then found no canonical bytes to upload.
//
// The interface assertions stay as a SECOND, belt-and-braces guard for the "cpu" case itself:
// if some future build ever registers a "cpu" name whose Backend value also happens to
// implement one of these (should never happen — NewBackend("cpu") always returns the plain CPU
// backend, in every configuration), this still declines to repacked-only rather than trust the
// name alone. Every other backendName (a real GPU name, or empty/unspecified) returns true
// unconditionally: canonical(+row4) stays today's default, never worse than before this policy
// existed.
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

// repackedOnlyOrCanonical is the shared decision behind quantizeEmbedWM and
// quantizeBatchedProjWM: given an already-quantized CANONICAL int4 WeightMat, build it
// repacked-only (aikit audit M-22) when needCanonical is false and this core/shape can build
// row4 (linalg.Int4Row4Usable) — closing the double nibble+scale residency the canonical+row4
// "both" repackW4A8IfEligible policy otherwise pays (aikit's own audit M-22, cross-referenced
// from goinfer's audit-2026-09-10.md's own M-22, an unrelated finding sharing the same label by
// coincidence of two repos' independent numbering). Falls back to repackW4A8IfEligible's
// existing policy whenever needCanonical is true, this core can't build row4 at all (non-arm64,
// no dotprod), or the shape doesn't qualify — identical to what every other int4 tensor already
// gets from quantizeWM, so this never produces a WORSE outcome than today's default.
//
// amd64 split-half repacked-only is out of scope for both callers: split-half repacking itself
// is already a separate, measured-marginal, parked feature (w4a8SplitHalfRepackEnabled, default
// off), so there is no default-on amd64 path this closes yet.
func repackedOnlyOrCanonical(canon linalg.WeightMat, needCanonical bool) linalg.WeightMat {
	if needCanonical {
		return repackW4A8IfEligible(canon)
	}
	q4, q4s, group, ok := canon.Int4()
	if !ok || !linalg.Int4Row4Usable(canon.Rows(), canon.Cols(), group) {
		return repackW4A8IfEligible(canon) // this core/shape can't do row4 at all — existing policy
	}
	if repacked, repOK := linalg.RepackInt4Row4InPlace(q4, q4s, canon.Rows(), canon.Cols(), group); repOK {
		return repacked
	}
	return repackW4A8IfEligible(canon) // Int4Row4Usable said yes, so this shouldn't miss — no silent data loss either way
}

// quantizeEmbedWM is quantizeWM's Embed/LMHead-specific twin (see repackedOnlyOrCanonical for
// the shared repacked-only-or-canonical policy). needCanonical is computed ONCE per Load call
// (wantsCanonicalInt4(be)) and threaded down rather than a *Backend* itself, since the weight
// builders that call this are already several calls removed from where be is constructed.
//
// Embed/LMHead: read via WeightMat.Row() (per-token embed lookup, layout-independent per
// aikit's own TestWeightMatRow_repackedMatchesCanonical) and driven through matmul()/
// matmulInto() as the LM head (single-op dispatch, gated on IsInt4() — see those functions' own
// comments) — never through the batched q/k/v or gate/up W4A8 dispatch this same policy is ALSO
// safe for now (quantizeBatchedProjWM, below), so this function stays even though the two now
// share the identical repackedOnlyOrCanonical body: Embed/LMHead's safety argument (Row() +
// single-op dispatch) is independent of the batched path's own (M=1-only, N%4==0), and a future
// change to either dispatch shape should not silently start covering the other tensor class too.
func quantizeEmbedWM(w linalg.WeightMat, mode quantMode, needCanonical bool) linalg.WeightMat {
	if mode != quantInt4 || fakeQuantScheme != "" {
		return quantizeWM(w, mode)
	}
	f32, ok := w.F32()
	if !ok {
		return quantizeWM(w, mode) // already quantized, or empty — quantizeWM's own no-op path
	}
	canon := maybeF16RoundInt4Scales(linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize))
	return repackedOnlyOrCanonical(canon, needCanonical)
}

// quantizeBatchedProjWM is quantizeWM's twin for the attention Q/K/V and MLP gate/up
// projections — the tensors reached through the batched W4A8 dispatch (matmulW4A8Batch /
// wmW4A8Op / isW4A8), which aikit's own audit M-22 note flags as unsafe for a repacked-only op
// in general: MatmulBTW4A8Batch has no row4 TILE (unlike WeightMat.MatmulBTW4A8Into's own M>1
// case), so a repacked-only op PANICS if it is ever reached at M>1, or when a fan-out shard
// boundary splits one of its quads (N%4 != 0 at the boundary).
//
// Verified safe for THIS codebase's actual two batch call sites (decoder/attention.go's
// causalAttention, decoder/mlp.go's gatedMLP): both are decode-only functions that call
// matmulW4A8Batch with a hardcoded M=1 literal — never reached from the batched-M prefill path,
// which uses matmul()/matmulInto() per projection instead (safe at any M, since those dispatch
// through WeightMat.MatmulBTW4A8Into directly, not the batch entry point). wmW4A8Op itself needs
// no change: it already builds the correct W4:nil/Row4:.../Row4Scales:... op shape for a
// repacked-only tensor (aikit's audit confirms this), and isW4A8's IsInt4() gate (already fixed,
// this same audit item) is what makes such a tensor reach it at all. The remaining condition —
// N%4==0 — is exactly linalg.Int4Row4Usable's own rows%4==0 check, applied per tensor by
// repackedOnlyOrCanonical below, so a shape that would violate the quad-boundary constraint
// never gets built repacked-only in the first place.
//
// Down-proj, the router, and MoE expert weights are deliberately NOT covered here: down-proj is
// unverified against this same batch-path constraint (it is never one of the two batched
// tensors today, but has not been separately audited), and expert/layer weights read through a
// read-only mmap span (paged .giw loading) are explicitly excluded by aikit's own note — paging
// has no load-time repack step, so they stay canonical-only regardless of this policy.
func quantizeBatchedProjWM(w linalg.WeightMat, mode quantMode, needCanonical bool) linalg.WeightMat {
	if mode != quantInt4 || fakeQuantScheme != "" {
		return quantizeWM(w, mode)
	}
	f32, ok := w.F32()
	if !ok {
		return quantizeWM(w, mode)
	}
	canon := maybeF16RoundInt4Scales(linalg.QuantizeInt4(f32, w.Rows(), w.Cols(), int4GroupSize))
	return repackedOnlyOrCanonical(canon, needCanonical)
}

// isBatchedProjTensor reports whether name — llama.cpp's own GGUF tensor-name convention — is
// one of the two batched-W4A8-dispatch projection groups quantizeBatchedProjWM covers: attention
// Q/K/V or MLP gate/up. Suffix-matched against the EXACT standard names (not a substring check),
// so an MoE expert tensor ("ffn_gate_exps.weight") or router ("ffn_gate_inp.weight") never
// matches — both are already excluded from this policy for their own reasons (see
// quantizeBatchedProjWM's doc comment) — and a family-specific split projection (e.g. an MLA
// q_a_proj/q_b_proj pair) simply stays on the existing quantizeWM/streamQuantized path
// automatically, with no per-family audit needed: only names this repo/llama.cpp's own
// convention actually produces for a plain dense Q/K/V/gate/up ever match.
func isBatchedProjTensor(name string) bool {
	for _, suf := range [...]string{"attn_q.weight", "attn_k.weight", "attn_v.weight", "ffn_gate.weight", "ffn_up.weight"} {
		if strings.HasSuffix(name, suf) {
			return true
		}
	}
	return false
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

// isW4A8 reports whether w is int4-resident, the only precision with a batched
// dispatch on the W4A8 path (audit R-06). IsInt4(), not Int4()'s narrower "canonical
// bytes present" ok — a repacked-only WeightMat (aikit audit M-22) is still int4 and
// still routes here; wmW4A8Op already builds the correct W4:nil/Row4:... op shape for
// it, once this gate stops excluding it.
func isW4A8(w *linalg.WeightMat) bool {
	return w.IsInt4()
}

// wmW4A8Op builds one linalg.W4A8Op for the batched W4A8 dispatch: canonical
// nibbles/scales are always present when isW4A8(w), and Row4/Row4Scales are
// populated only when RepackInt4Row4 has already run for this tensor (arm64,
// heap-backed weights only — see repackW4A8Row4IfEligible) — nil otherwise, which
// linalg.MatmulBTW4A8Batch reads as "run canonical for this op". group is read
// separately (assumed shared across the batch, exactly like MatmulBTW4A8Into's own
// single-scalar signature) since every op in one call comes from the same layer's
// quant config.
func wmW4A8Op(w *linalg.WeightMat, dst []float32) (op linalg.W4A8Op, group int) {
	q4, q4s, group, _ := w.Int4()
	row4, row4s, _ := w.Int4Row4()
	return linalg.W4A8Op{W4: q4, Scales: q4s, Row4: row4, Row4Scales: row4s, Dst: dst, N: w.Rows()}, group
}

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
// with goinfer's backend routing: the f32, W8A8 and W4A8 paths can run on a GPU backend
// (be.MatmulBT / QuantBackend.MatmulW8A8 / QuantBackend4.MatmulW4A8, the last a G6
// docs/task-gpu-paths-2026-09.md addition); weight-only int8 (Q8) stays CPU. (The old
// weightMat.matmul, now a free function over linalg.WeightMat.)
func matmul(be Backend, w *linalg.WeightMat, a, dst []float32, M int) {
	if w.IsInt4() {
		// G6 (docs/task-gpu-paths-2026-09.md): staged int4 backend consult, mirroring the W8A8
		// branch below — matmulInto's own int4 branch gets the same fix, for the same reason.
		//
		// Nested under Int4()'s narrower ok (canonical bytes present), not the outer IsInt4():
		// the staged consult hands q4/q4s to the GPU backend on EVERY call (no persistent
		// upload), so a repacked-only tensor (no canonical bytes, aikit audit M-22) has nothing
		// to hand it here — reconstructing canonical on demand would cost a canonical-sized
		// allocation per token, the wrong shape for a per-call path. It falls through to
		// w.MatmulBTW4A8Into below instead, which dispatches on whichever layout is actually
		// present (canonical, row4, or split-half) via aikit's own per-arch method.
		if q4, q4s, group, ok := w.Int4(); ok {
			if qb, ok := be.(QuantBackend4); ok && qb.MatmulW4A8(a, q4, q4s, group, dst, M, w.Cols(), w.Rows()) {
				return
			}
		}
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
		// load-time layout repacks actually get used here — the repack alone does
		// nothing without this call using it. BOTH arches depend on this one line:
		// arm64's split-half + 4-row-interleave (RepackInt4Row4, docs/task-w4a8-neon-
		// bandwidth.md) and amd64's split-half (RepackInt4SplitHalf, queue-performance
		// P14 item 3). Neither has a dispatch of its own — aikit picks the layout
		// inside this method, at M=1 only, whenever the repack populated it.
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
		// Pooled Workspace, same reason as the W8A8 case just above — the bare
		// linalg.MatmulBTQ8 wrapper builds a fresh, non-pooled Workspace every call
		// (docs/prompts/lmhead-workspace-fix.md, Step 1: the LM head's own weight-only
		// Q8 path, the largest per-token cost measured in the W4A8 plumbing phase).
		ws := matmulWSPool.Get().(*linalg.Workspace)
		defer matmulWSPool.Put(ws)
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		linalg.MatmulBTQ8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
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
	if q8, scales, w8a8, ok := w.Int8(); ok && !w8a8 {
		// Weight-only Q8 (the int8-pinned LM head, in int4 mode): thread the
		// caller's own scratch Workspace directly, the same as the branches
		// above/below, instead of falling through to matmul()'s pool round-trip.
		// docs/prompts/lmhead-workspace-fix.md Step 1.
		ws.SetThreshold(DefaultDecodeParallelThreshold)
		linalg.MatmulBTQ8Into(ws, a, q8, scales, dst, M, w.Cols(), w.Rows())
		return
	}
	if w.IsInt4() {
		// G6 (docs/task-gpu-paths-2026-09.md): the staged int4 backend consult this branch
		// never had, mirroring the isW8A8 branch's QuantBackend check above.
		//
		// Nested under Int4()'s ok, not the outer IsInt4() — see matmul()'s own comment on
		// this same shape for why a repacked-only tensor cannot serve the staged consult.
		if q4, q4s, group, ok := w.Int4(); ok {
			if qb, ok := be.(QuantBackend4); ok && qb.MatmulW4A8(a, q4, q4s, group, dst, M, w.Cols(), w.Rows()) {
				return
			}
		}
		// Same threshold matmul's fresh Workspace sets — the point of the reuse is to stop
		// allocating one per projection per token, not to change how the work is fanned out.
		// w.MatmulBTW4A8Into, not the raw free function — see matmul's own comment above.
		ws.SetThreshold(int4ParThreshold)
		w.MatmulBTW4A8Into(ws, a, dst, M)
		return
	}
	matmul(be, w, a, dst, M)
}

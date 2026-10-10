package decoder

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// The load-time fit guard: refuse a model that cannot fit in RAM BEFORE allocating it, naming the numbers and the flag that fixes it (an
// unpinned context that does not fit is pinned smaller instead: see guardFit). It prices and refuses; choosing a placement is Plan's job
// (fitplan.go) and whether to retry with -stream-weights is the caller's (FitDeclineError.DenseStreamable). The failure it prevents, a
// 35B-A3B on a 16 GB Mac paging into swap in seconds with no message, is docs/measurements/cold-user-2026-09-06.md scenario D.
//
// EVERY UNKNOWN PROCEEDS. An unreadable RAM figure, an unsupported source format, a zero-byte estimate: each returns "don't know" and the
// load continues. The guard's failure mode must be letting a doomed load through, never refusing one that would have run.

// fitMemFraction is the share of the fit check's base memory figure (currently available memory) the WEIGHTS alone may occupy. The rest is
// not slack: KV, scratch, the tokenizer and the operating system live there too. Same figure as metal/backend.go's residentMemFraction,
// from one measured thrash failure, so a threshold rather than a swept curve.
const fitMemFraction = WeightsMemFraction

// fitWarnRatio is how close to the budget the load has to come before the banner prints the
// arithmetic unasked. At 0.75 the message appears within 25% of the refusal, so the user sees the
// cliff on the run BEFORE the one that steps off it.
const fitWarnRatio = 0.75

// hostRAM is indirected so a test can inject a machine's worth of RAM instead of needing one.
var hostRAM = HostRAMBytes

// hostRAMAvailable is the same indirection for CURRENTLY AVAILABLE memory (prefill_budget.go): a live figure, never cached, unlike
// hostRAM's total-RAM snapshot.
var hostRAMAvailable = HostRAMAvailableBytes

// ctxFloor is the smallest context this guard will auto-pin down to when the caller did not pin one and the model's own maximum does not
// fit; below it a context is not useful enough to hand a user silently, so refuse instead. A product choice, not a measured floor, and
// separate from Plan's ctxPlanFloor.
const ctxFloor = 2048

// fitCheck is the arithmetic, separated from every source of it so it can be driven with the
// numbers from a measurement rather than a 21 GB checkpoint.
type fitCheck struct {
	noGuard     bool   // GOINFER_NO_FIT_GUARD (or Options.Knobs) at Load: skip the guard entirely
	name        string // what to call the model in the message
	quant       string // the requested quant, named because it moves the weight term the most
	weightBytes int64  // estimated resident weight bytes AT THAT QUANT
	kvBytes     int64  // KV at effCtx (see below) — always priced now, not only when pinned
	availBytes  int64  // CURRENTLY AVAILABLE memory at check time, 0 when unknown

	// srcFileBytes is the on-disk size of a plain (non-streamed) .gguf SOURCE file, priced as an additional transient term beside
	// weightBytes+kvBytes: the mapping (embed.OpenGGUFMmap) stays open for the whole build and the per-row dequantizer touches essentially
	// every page, so for most of the load the whole source file is resident at the same time as the resident weight set built from it. That is
	// source and destination coexisting, not double-buffering: nothing releases the source pages as each tensor is consumed (that would need
	// per-tensor madvise inside aikit/embed).
	//
	// Only meaningful for a plain resident .gguf load (isGGUF && !streamWeights). A .giw load mmaps its weight blob directly and never reaches
	// fitCheckFor (model.go's .giw branch: its pages are file-backed and droppable under pressure, so there is no allocation peak to price),
	// and a safetensors directory's loader has its own accounting. Zero when not applicable, so a fitCheck literal built elsewhere is
	// unaffected. The measurement: docs/code-notes/decoder.md#fitCheck.srcFileBytes.
	srcFileBytes int64

	// cudaBuildBytes is the HOST peak of building a CUDA C' expert cache (--backend cuda --moe-cache-experts) from a plain .gguf:
	// 2*weightBytes + expertBytes (the canonical weights on the Go heap, a second host-packed copy, and a pinned host copy of the experts).
	// Zero when that path is not in play. It REPLACES the weights+KV+srcFileBytes total when it is larger rather than adding to it, because
	// the phases do not overlap: decoder.Load unmaps the source file before cuda.BuildResident starts, and on the resident path KV lives in
	// VRAM. Go heap slack (about 9% over this estimate on the measured model) is not priced. The measurement:
	// docs/code-notes/decoder.md#fitCheck.cudaBuildBytes.
	cudaBuildBytes int64
	expertBytes    int64

	// cfg, effCtx, pinned, kvF16, kvI8 carry what re-pricing needs to recompute KV at a SMALLER context when the load does not fit and the
	// caller never pinned one (see smallerFittingContext). cfg is nil when the GGUF's config could not be read: kvBytes stays 0 and unknown
	// proceeds.
	cfg    *Config
	effCtx int  // the context KV was priced at: opts.ResidentContext if pinned, else cfg.MaxPositions
	pinned bool // true when the caller explicitly requested effCtx (an explicit request that
	// cannot be honoured is refused, never silently downgraded)
	kvF16 bool
	kvI8  bool

	// denseStreamable is true when a -stream-weights retry after this refusal would engage
	// decoder/layerpaging.go's windowed dense pager — see denseStreamable(cfg)'s own doc comment.
	// Carried into declineErr()'s *FitDeclineError so a caller can decide whether an automatic
	// retry is sound without re-deriving this from the arch registry itself.
	denseStreamable bool

	// isGGUF is true when the source that was priced is a .gguf FILE, false for a safetensors DIRECTORY; remedy() needs it because
	// -stream-weights only has anything to do for a .gguf source.
	isGGUF bool
}

func (f fitCheck) need() int64 {
	n := max(f.cudaBuildBytes, f.weightBytes+f.kvBytes+f.srcFileBytes)
	return n
}
func (f fitCheck) budget() int64 { return int64(float64(f.availBytes) * fitMemFraction) }

// known reports whether both sides of the comparison are real numbers. Anything else proceeds.
func (f fitCheck) known() bool { return f.availBytes > 0 && f.weightBytes > 0 }

func (f fitCheck) fits() bool { return !f.known() || f.need() <= f.budget() }

// ratio is need/budget — 1.0 is exactly at the refusal.
func (f fitCheck) ratio() float64 {
	if !f.known() || f.budget() <= 0 {
		return 0
	}
	return float64(f.need()) / float64(f.budget())
}

const fitGB = 1 << 30

// arithmetic is the one line both the warning and the refusal share, so the two can never quote
// different numbers for the same load.
func (f fitCheck) arithmetic() string {
	var b strings.Builder
	if f.cudaBuildBytes > f.weightBytes+f.kvBytes+f.srcFileBytes {
		fmt.Fprintf(&b, "%s needs ~%.1f GB of host memory while building the CUDA expert cache (%.1f GB weights + %.1f GB packed copy + %.1f GB pinned copy of the experts)",
			f.name, float64(f.cudaBuildBytes)/fitGB, float64(f.weightBytes)/fitGB, float64(f.weightBytes)/fitGB, float64(f.expertBytes)/fitGB)
		fmt.Fprintf(&b, "; this machine currently has %.1f GB of memory available (budget %.1f GB = %.0f%% of that)",
			float64(f.availBytes)/fitGB, float64(f.budget())/fitGB, fitMemFraction*100)
		return b.String()
	}
	fmt.Fprintf(&b, "%s needs ~%.1f GB resident at quant %s", f.name, float64(f.weightBytes)/fitGB, f.quant)
	if f.kvBytes > 0 {
		fmt.Fprintf(&b, " + %.1f GB KV", float64(f.kvBytes)/fitGB)
	}
	if f.srcFileBytes > 0 {
		// MEASURED (see srcFileBytes's own doc comment): the on-disk .gguf stays mmap-resident for
		// the whole load, alongside the resident weights being built from it — named explicitly so
		// this doesn't read as a doubled weight estimate.
		fmt.Fprintf(&b, " + %.1f GB reading the checkpoint (the .gguf stays mapped resident for the whole load)", float64(f.srcFileBytes)/fitGB)
	}
	if f.kvBytes > 0 || f.srcFileBytes > 0 {
		fmt.Fprintf(&b, " = %.1f GB", float64(f.need())/fitGB)
	}
	fmt.Fprintf(&b, "; this machine currently has %.1f GB of memory available (budget %.1f GB = %.0f%% of that)",
		float64(f.availBytes)/fitGB, float64(f.budget())/fitGB, fitMemFraction*100)
	return b.String()
}

// warning is what the banner prints when the load fits but only just.
func (f fitCheck) warning() string {
	return fmt.Sprintf("decoder: fit is tight — %s (%.0f%% of budget). %s",
		f.arithmetic(), f.ratio()*100, f.remedy())
}

// remedy names the way out and what it will do, because the user who reads this message is by definition the one who did not know the
// option existed: -stream-weights for a .gguf source. For a safetensors directory -stream-weights does nothing (serve's gates are
// .gguf-suffix-only and decoder.Load ignores Options.StreamWeights for a directory), and recommending it gives an identical refusal after
// the user did as told; the remedy there is a one-time .giw build with cmd/prequant, which streams one layer at a time
// (StreamTranscodeDir) and no longer hits this guard. GOINFER_NO_FIT_GUARD=1 stays in that text for the case the 70% margin is being
// conservative about.
func (f fitCheck) remedy() string {
	if f.isGGUF {
		return "Re-run goinfer-serve with -stream-weights: it caches the model as a sidecar .giw once, " +
			"then pages weights out of it on demand instead of holding them all resident."
	}
	return "This is a safetensors checkpoint — -stream-weights only helps a .gguf source. Build a .giw " +
		"from it once with cmd/prequant (-quant <q> -o <name>.giw <dir>), which streams one layer at a time " +
		"and does not need this load's memory, then load the .giw (goinfer-serve and goinfer-chat do this " +
		"for you unless --direct-load). If you believe this machine can hold it as it is (this guard's 70% " +
		"margin is deliberately conservative), GOINFER_NO_FIT_GUARD=1 loads it anyway."
}

// ErrWontFitResident is wrapped into every load-time fit-guard refusal (declineErr), so a caller
// can detect the refusal with errors.Is/errors.As instead of parsing the message text — e.g. to
// decide whether an automatic -stream-weights retry is worth attempting (see FitDeclineError).
var ErrWontFitResident = errors.New("decoder: model will not fit resident RAM")

// FitDeclineError is what declineErr returns instead of a bare error.
type FitDeclineError struct {
	msg string

	// DenseStreamable is true when a -stream-weights retry after this refusal would engage decoder/layerpaging.go's windowed dense pager, the
	// path measured as sound for an automatic retry (docs/tasks/task-fit-to-hardware.md, docs/tasks/task-gpu-paths-2026-09.md). It is false for
	// MoE models and own-forward families: MoE CPU weight streaming is a documented, measured failure (docs/benchmarks.md "M35/M26 on the
	// Mac": a real 20 GB MoE through the CPU-staged path ran for hours with no completions, re-reading weights from disk on essentially every
	// token), so there the retry stays a manual, explicit choice and is never something the guard does on the caller's behalf.
	DenseStreamable bool
}

func (e *FitDeclineError) Error() string { return e.msg }
func (e *FitDeclineError) Unwrap() error { return ErrWontFitResident }

// denseStreamable mirrors newLayerPager's own exclusions (decoder/layerpaging.go) exactly, so the fit guard's retry decision and the
// pager's decision to build one cannot disagree: an MoE model uses expertPager instead (see FitDeclineError.DenseStreamable), and an
// own-forward family (resolved through arch.ownForward(), not a hand-written list) runs a layer loop that never calls enterLayer, so
// layerPager would never engage for it. cfg == nil (header unreadable) answers false: don't know must not attempt a retry the guard cannot
// vouch for.
func denseStreamable(cfg *Config) bool {
	if cfg == nil {
		return false
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil || arch == nil || arch.MoE != nil {
		return false
	}
	_, own := arch.ownForward()
	return !own
}

// declineErr is the refusal. It is an error, not a warning, because the measured alternative is a
// machine that stops responding: the user cannot read a warning on a box that is thrashing.
func (f fitCheck) declineErr() *FitDeclineError {
	// Suggesting a smaller quant to someone already at int4 is noise, and noise in a refusal is how the useful line gets skipped. And int4 is
	// not always the smaller one: where the loader keeps a repacked second copy of the nibbles (arm64 with dotprod, AVX2 without VNNI) int4
	// costs more resident memory than int8, so the line is derived from the same measured bytes-per-element as the arithmetic
	// (quantBytesPerElem) rather than from the nominal bit width.
	alt := "  Or run a smaller model.\n"
	if f.quant != "int4" && f.quant != "int4mix" &&
		quantBytesPerElem(quantInt4) < quantBytesPerElem(quantInt8) {
		alt = "  Or run a smaller model, or --quant int4, which is smaller on this machine.\n"
	}
	msg := fmt.Sprintf("decoder: %s.\n"+
		"  Loading it would page to swap rather than run, so it was NOT loaded.\n"+
		"  %s\n"+
		"%s"+
		"  Set GOINFER_NO_FIT_GUARD=1 to load anyway if this machine really fits it",
		f.arithmetic(), f.remedy(), alt)
	return &FitDeclineError{msg: msg, DenseStreamable: f.denseStreamable}
}

// guardFit runs the check. It returns the context to PIN (0 meaning leave the caller's request alone, nonzero meaning the guard chose this
// smaller one, apply it) and the refusal, or (0, nil) to proceed unchanged. It prints the arithmetic to stderr when the load is within
// fitWarnRatio of refusing, or when it auto-pins.
//
// Three outcomes, in order: an explicit pin that does not fit is refused (an explicit request that cannot be honoured is refused, not
// silently downgraded); an unpinned load that does not fit at the model's maximum but does fit at some smaller context >= ctxFloor is
// auto-pinned to it, reported, and proceeds; an unpinned load that does not fit even at ctxFloor is refused. KV is always priced, at the
// model's own maximum when nothing is pinned, because nothing else bounds the CPU cache's growth.
func guardFit(f fitCheck) (int, error) {
	if f.noGuard {
		return 0, nil
	}
	if !f.known() {
		return 0, nil
	}
	if !f.fits() {
		if ctx, ok := f.smallerFittingContext(); ok {
			fmt.Fprintln(os.Stderr, f.capNote(ctx))
			return ctx, nil
		}
		return 0, f.declineErr()
	}
	if f.ratio() >= fitWarnRatio {
		fmt.Fprintln(os.Stderr, f.warning())
	}
	return 0, nil
}

// giwMemMargin is the flat margin guardGIWFit leaves against the live availability probe (docs/tasks/task-never-swap-2026-09.md, item 1).
// A .giw load's weights are file-backed (zero-copy aliases, evictable under memory pressure), so it deliberately does not reuse
// fitMemFraction's 70% conservatism, which is sized for a load that commits anonymous memory for its weights; only KV and scratch are real
// anonymous cost here.
const giwMemMargin = 1 << 30 // 1 GB

// guardGIWFit is the load-time check for a .giw load: refuse, or auto-pin to a smaller context, when KV + scratch would exceed CURRENTLY
// AVAILABLE memory minus giwMemMargin. Same return shape as guardFit (0 = no pin needed). Every unknown proceeds, as in guardFit: no
// cfg, no live probe reading, or no usable context all mean cannot price, so do not refuse.
//
// Scratch is priced at prefillAttnScratchBudget alone, the fixed attention-scratch cap every prefill enforces; prefillScratchBytes' other
// term (gate/up MLP activations) scales with a prompt's length, which load time does not know.
//
// Limitation: Metal's per-projection host buffer copies (metal/model.go's int4Buf/int4Concat) are not priced, so on Metal this under-counts.
// decoder cannot import metal, so pricing it needs its own hook, which is not built; CPU is this function's only backend. Recorded in
// docs/tasks/task-never-swap-2026-09.md ("Not priced").
func guardGIWFit(cfg *Config, opts Options) (pinnedCtx int, err error) {
	if loadKnob(opts, knobNoFitGuard) != "" {
		return 0, nil
	}
	if cfg == nil {
		return 0, nil
	}
	avail := hostRAMAvailable()
	if avail <= 0 {
		return 0, nil
	}
	budget := max(avail-giwMemMargin, 0)
	p, ok := kvPricingFor(cfg, opts) // what the load will allocate: Metal's f16 at its default context, or the CPU's ceiling
	if !ok {
		return 0, nil // no pin and the model's own max is unknown ⇒ can't price KV; proceed
	}
	pinned, effCtx, kvF16, kvI8 := p.pinned, p.ctx, p.kvF16, p.kvI8
	needAt := func(ctx int) int64 { return estimateKVBytes(cfg, ctx, kvF16, kvI8) + prefillAttnScratchBudget }
	need := needAt(effCtx)
	if need <= budget {
		return 0, nil
	}
	if pinned {
		return 0, fmt.Errorf(
			"decoder: pinned context %d needs ~%.2f GB (KV + scratch) but only %.2f GB of this "+
				"machine's %.2f GB currently-available memory would be left as a %.0f GB safety "+
				"margin — pass a smaller -ctx, or set GOINFER_NO_FIT_GUARD=1 to allow it anyway",
			effCtx, float64(need)/fitGB, float64(budget)/fitGB, float64(avail)/fitGB, float64(giwMemMargin)/fitGB)
	}
	if floorNeed := needAt(ctxFloor); floorNeed > budget {
		return 0, fmt.Errorf(
			"decoder: this model needs ~%.2f GB (KV + scratch) even at the %d-token floor, but only "+
				"%.2f GB of this machine's %.2f GB currently-available memory would be left as a "+
				"%.0f GB safety margin — set GOINFER_NO_FIT_GUARD=1 to allow it anyway",
			float64(floorNeed)/fitGB, ctxFloor, float64(budget)/fitGB, float64(avail)/fitGB, float64(giwMemMargin)/fitGB)
	}
	lo, hi := ctxFloor, effCtx
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if needAt(mid) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	fmt.Fprintf(os.Stderr,
		"decoder: .giw context capped at %d (model allows %d) so that KV+scratch fits: needs ~%.2f GB "+
			"of %.2f GB available (%.0f GB margin) — pass -ctx to choose, or free memory to lift\n",
		lo, effCtx, float64(needAt(lo))/fitGB, float64(avail)/fitGB, float64(giwMemMargin)/fitGB)
	return lo, nil
}

// resolveWeightCacheBudget resolves WeightCacheBytes's 0 "auto" default from this platform's live probe (hostRAMAvailable: vm_stat on
// darwin, /proc/meminfo on linux): half of available memory, the same math aikit's mmap.AutoBudget() uses, but AutoBudget is Linux-only and
// falls back to a fixed 8 GB elsewhere, a number with no relation to what the machine has free. An explicit request (> 0) passes through
// unchanged. When this platform's probe is unavailable too it returns 0 (aikit's own AutoBudget) and logs that, so the fallback is visible.
func resolveWeightCacheBudget(requested int64) int64 {
	if requested > 0 {
		return requested
	}
	if avail := hostRAMAvailable(); avail > 0 {
		return avail / 2
	}
	fmt.Fprintln(os.Stderr, "decoder: no live memory probe on this platform — falling back to aikit's default weight-cache budget (Linux /proc probe, or a fixed 8 GB elsewhere)")
	return 0
}

// quantBytesPerElem is the resident cost of one weight ELEMENT of a 2-D matmul matrix under each quant mode, MEASURED by running a probe
// matrix through the loader's own quantization, not derived from the nominal bit width. The loader repacks (repackW4A8Row4IfEligible on
// arm64, repackW4A8SplitHalfIfEligible on AVX2-without-VNNI amd64) and keeps the canonical nibbles beside the repacked buffer, so an int4
// weight can cost about twice its encoding; wmBytes counts both. A constant tuned on one box is wrong exactly where the guard matters
// (Apple Silicon); measuring through the real path tracks the arch, the CPU features and any repack added later.
// TestFitEstimate_agreesWithResidentWeightBytes pins the estimate against the loader's own accounting.
//
// The residual, stated rather than hidden: the probe is 256x256, which the repacks accept (rows a multiple of 4, cols a multiple of
// int4GroupSize). A model built entirely from matrices the repack rejects would be over-priced, the direction that can refuse a model
// that would have fit. Real transformer matrices are multiples of 4 and 32, the 70% budget has slack of its own, and
// GOINFER_NO_FIT_GUARD is named in the refusal: taken deliberately, because the alternative under-reports by ~2x on the platform the guard
// was written for.
func quantBytesPerElem(q quantMode) float64 {
	bpeMu.Lock()
	defer bpeMu.Unlock()
	if v, ok := bpeCache[q]; ok {
		return v
	}
	v := measureBytesPerElem(q)
	bpeCache[q] = v
	return v
}

var (
	bpeMu    sync.Mutex
	bpeCache = map[quantMode]float64{}
)

// measureBytesPerElem quantizes a probe matrix through quantizeWM — the loader's own path,
// repacks included — and asks wmBytes what it costs. Sub-millisecond, and cached per mode.
func measureBytesPerElem(q quantMode) float64 {
	const n = 256 // rows and cols: a multiple of 4 and of int4GroupSize, so the repacks apply
	f32 := make([]float32, n*n)
	for i := range f32 {
		// Non-degenerate values: byte counts do not depend on them, but a matrix of zeros is the
		// kind of probe that quietly stops exercising a scale path if one is ever added.
		f32[i] = float32(i%251) - 125
	}
	// quantInt4Mix is a LOAD-TIME POLICY, not a resident precision — quantizeWM does not handle it
	// and would hand back the untouched f32, pricing the model at 4 bytes/element and refusing
	// models that fit six times over. Measure its dominant class instead: the FFN bulk it puts at
	// int4. Attention stays int8, so this stays a lower bound, which is the safe direction.
	probe := q
	if probe == quantInt4Mix || probe == quantQ4K {
		// q4k: Q4_K tensors are 4.5 bpw like int4, the rest int8 — int4 is its lower bound too.
		probe = quantInt4
	}
	wm := quantizeWM(linalg.WrapF32(f32, n, n), probe)
	if b := wmBytes(&wm); b > 0 {
		return float64(b) / float64(n*n)
	}
	// quantizeWM is a no-op for quantNone and anything it does not handle; f32 is the answer then.
	return 4
}

// f32PinnedTensorName reports whether a checkpoint tensor name is one the loaders ALWAYS keep resident at f32 regardless of the requested
// quant: a Mamba-2 mixer (Granite-4.0-H's "mamba.*" prefix, GGUF "ssm_*") or an MLA attention projection (GGUF
// "attn_q_a"/"attn_q_b"/"attn_kv_a_mqa"/"attn_kv_b", safetensors "q_a_proj"/"q_b_proj"/"kv_a_proj"/"kv_b_proj").
//
// DELIBERATELY NOT a bare "mixer."/"in_proj"/"out_proj" match: Nemotron-H shares the "mixer." prefix across its Mamba-2 (f32), attention,
// MLP and MoE layer types, and bare "in_proj"/"out_proj" collide with LFM2's conv mixer and qwen3.5/qwen3_next's DeltaNet hybrid, where
// SOME "linear_attn.in_proj_*" sub-tensors are f32 and OTHERS are genuinely quantized, so a name-only classifier cannot tell them apart.
// Nemotron-H's Mamba-2 in_proj/out_proj/conv1d ARE matched, via the full "mixer.in_proj"/"mixer.out_proj"/"mixer.conv1d" compound
// (never bare "mixer." or bare "in_proj"), which no other Nemotron block kind contains.
//
// KNOWN RESIDUAL GAP, left unfixed rather than over-matched: MLA's output projection (self_attn.o_proj.weight / Bailing's dense.weight) is
// ALSO f32-pinned (loadDeepseekAttn/loadDeepseekAttnGGUF), but "o_proj"/"dense.weight" are ordinary, widely used quantizable tensor names
// in every non-MLA family, so matching them bare would misclassify most of the registry. The result is a small, honest under-estimate
// (o_proj is a minority of MLA's attention weight: TestFitEstimate_f32PinnedMixersAgreeWithResident).
func f32PinnedTensorName(name string) bool {
	for _, s := range []string{
		"ssm_", "mamba.",
		"mixer.in_proj", "mixer.out_proj", "mixer.conv1d", // Nemotron-H Mamba-2 only (see doc comment)
		"attn_q_a", "attn_q_b", "attn_kv_a", "attn_kv_b",
		"q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
	} {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// visionTowerTensorName reports whether name belongs to a bundled multimodal vision tower or projector rather than the text decoder.
// decoder/weights.go's text loader never requests these names, but this estimator walks EVERY tensor in the checkpoint's metadata, so it
// must exclude them explicitly or they are swept into the text model's per-element quant price.
func visionTowerTensorName(name string) bool {
	return strings.Contains(name, "vision_tower.") || strings.Contains(name, "multi_modal_projector.")
}

// gptqAWQAuxSuffix reports whether name is a GPTQ/AWQ auxiliary tensor (qzeros/g_idx/scales, decoder/gptq.go, decoder/awq.go) that is fully
// consumed during the one-time reconstruct-then-requantize step and never itself kept resident: pricing it as an ordinary weight matrix at
// the target quant's rate would double-count storage the real load path discards.
func gptqAWQAuxSuffix(name string) bool {
	for _, s := range []string{".qzeros", ".g_idx", ".scales"} {
		if strings.HasSuffix(name, s) {
			return true
		}
	}
	return false
}

// gptqAWQPackFactor is how many 4-bit codes GPTQ/AWQ pack into each element along qweight's packed dimension (decoder/gptq.go: 8 4-bit codes
// per int32), so qweight's ON-DISK shape under-counts the logical [in,out] matrix by exactly this factor. The checkpoint is fully unpacked
// and re-quantized to the requested resident quant at load time (decoder/weights.go's loadProj), so the estimator must correct for it or
// lands ~8x low.
const gptqAWQPackFactor = 8

// estimateGGUFWeightBytes sums every tensor's element count from the GGUF's metadata and prices
// it at the target quant. Metadata only: no tensor data is touched, so this costs a header parse
// on a model that is about to be read in full anyway.
//
// 1-D tensors (norms, biases) stay f32 whatever the quant, so they are priced as f32. They round
// to nothing beside the matrices, which is exactly the accounting ResidentWeightBytes documents.
func estimateGGUFWeightBytes(path string, q quantMode) int64 {
	total, _ := estimateGGUFWeightBreakdown(path, q)
	return total
}

// estimateGGUFWeightBreakdown is estimateGGUFWeightBytes plus the share of it that is routed-expert
// weight (GGUF names carry "_exps"; the small per-expert bias tables are left out).
func estimateGGUFWeightBreakdown(path string, q quantMode) (total, experts int64) {
	g, err := OpenGGUFMmap(path)
	if err != nil {
		return 0, 0 // unknown ⇒ proceed
	}
	defer g.Close()
	var sum, exp float64
	for _, name := range g.Names() {
		dims, ok := g.Dims(name)
		if !ok {
			continue
		}
		n := 1
		for _, d := range dims {
			if d <= 0 {
				n = 0
				break
			}
			n *= d
		}
		if n == 0 {
			continue
		}
		if len(dims) < 2 {
			sum += 4 * float64(n)
			continue
		}
		// Mamba-2/MLA tensors stay f32 regardless of the requested quant: price them at what they cost, not the ambient rate.
		if f32PinnedTensorName(name) {
			sum += 4 * float64(n)
			continue
		}
		b := quantBytesPerElem(q) * float64(n)
		sum += b
		if strings.Contains(name, "_exps") && !strings.HasSuffix(name, ".bias") {
			exp += b
		}
	}
	return int64(sum), int64(exp)
}

// estimateSafetensorsWeightBytes is estimateGGUFWeightBytes's safetensors twin: shape-only, quant-independent accounting (sum every
// tensor's element count, price 2-D+ tensors at the target quant, 1-D norms/biases at f32), reusing openCheckpointMmap so single-file and
// sharded (model.safetensors.index.json) checkpoints are handled as a real load handles them.
//
// WHY SHAPE-ONLY, NOT ON-DISK FILE SIZE: a safetensors checkpoint is usually f32 or bf16 on disk and shrinks several-fold once quantized on
// load, so pricing the on-disk bytes would refuse loads that fit comfortably. That is wrong in the refusing direction, which this guard's
// design treats as worse than not pricing at all. Element counts do not depend on the on-disk dtype, and are priced at the REQUESTED load
// quant, as estimateGGUFWeightBytes does.
func estimateSafetensorsWeightBytes(dir string, q quantMode) int64 {
	st, err := openCheckpointMmap(dir)
	if err != nil {
		return 0 // unknown ⇒ proceed
	}
	defer st.Close()
	var total float64
	for _, name := range st.Names() {
		t, err := st.Tensor(name)
		if err != nil {
			continue
		}
		n := 1
		for _, d := range t.Shape {
			if d <= 0 {
				n = 0
				break
			}
			n *= d
		}
		if n == 0 {
			continue
		}
		// A GPTQ/AWQ auxiliary tensor is fully consumed during reconstruction and never kept resident: price it at zero, not as a weight matrix.
		if gptqAWQAuxSuffix(name) {
			continue
		}
		// A bundled vision tower/projector loads at its own precision, independent of the text
		// model's requested quant (aikit/vision.LoadEncoder's own -vision-quant knob, invisible
		// here) — price it at f32, the default the text loader never touches it at, rather than
		// blending it into the text quant rate.
		if visionTowerTensorName(name) {
			total += 4 * float64(n)
			continue
		}
		if len(t.Shape) < 2 {
			total += 4 * float64(n)
			continue
		}
		// Mamba-2/MLA tensors stay f32 regardless of the requested quant.
		if f32PinnedTensorName(name) {
			total += 4 * float64(n)
			continue
		}
		// A GPTQ/AWQ qweight tensor's ON-DISK shape is the packed [in/8, out] layout, not the
		// logical [in, out] matrix it is unpacked and re-quantized to at load time — correct for
		// the pack factor before pricing at the target quant's rate, or this under-counts ~8x.
		if strings.HasSuffix(name, ".qweight") {
			n *= gptqAWQPackFactor
		}
		// The token embedding and LM head are held at q.embedding() (int8 under int4), not at q: pricing
		// them at q under-counted every big-vocabulary small model, and binary16 int4 scales (aikit
		// v1.50.0) pushed the tiny fixtures' estimate below the 0.85 band.
		if embeddingTensorName(name) {
			total += quantBytesPerElem(q.embedding()) * float64(n)
			continue
		}
		total += quantBytesPerElem(q) * float64(n)
	}
	return int64(total)
}

// embeddingTensorName reports a checkpoint tensor the loaders keep at the embedding precision
// (quantMode.embedding()): a token-embedding table or an untied LM head, under the names the supported
// safetensors families use.
func embeddingTensorName(name string) bool {
	if name == "output.weight" { // InternLM2's LM head
		return true
	}
	for _, s := range []string{"embed_tokens.weight", "lm_head.weight", "tok_embeddings.weight", "wte.weight", "word_embeddings.weight", "embed_in.weight", "embed_out.weight"} {
		if name == s || strings.HasSuffix(name, "."+s) {
			return true
		}
	}
	return false
}

// kvBytesPerPosition is the KV cost of ONE position, so estimateKVBytes and smallerFittingContext share the identical per-position rate.
func kvBytesPerPosition(cfg *Config, kvF16, kvI8 bool) int64 {
	if cfg == nil || cfg.NumLayers <= 0 || cfg.NumKVHeads <= 0 {
		return 0
	}
	perElem := 4.0
	switch {
	case kvI8:
		perElem = 1.125 // int8 payload + per-row f32 scale
	case kvF16:
		perElem = 2
	}
	kvDim := cfg.NumKVHeads * cfg.headDim()
	// ×2 for K and V.
	return int64(2 * perElem * float64(kvDim) * float64(cfg.NumLayers))
}

// estimateKVBytes is the KV cache at ctx positions. ctx is the caller's job to choose (fitCheckFor picks opts.ResidentContext when pinned,
// else cfg.MaxPositions): "no context pinned" does not mean no KV is ever allocated, it means the ceiling is the model's own maximum,
// because nothing else bounds the CPU/Metal-staged KV cache's growth, and a real agent request (system prompt plus tool schema, tens of
// thousands of tokens) reaches it. This function just prices what it is given.
func estimateKVBytes(cfg *Config, ctx int, kvF16, kvI8 bool) int64 {
	if ctx <= 0 {
		return 0
	}
	// Resolve the real per-layer geometry when possible (kvBytesForCtx, decoder/arch.go): the flat kvBytesPerPosition formula overprices
	// hybrid (DeltaNet/conv/Mamba), sliding-window and MLA models. Fall back to the flat formula only when the architecture cannot be
	// resolved at all: not a real load, but a synthetic Config with no registered model_type, which several unit tests construct.
	if arch, _, err := resolveArchitecture(cfg); err == nil && arch != nil {
		return kvBytesForCtx(arch, ctx, kvF16, kvI8)
	}
	return kvBytesPerPosition(cfg, kvF16, kvI8) * int64(ctx)
}

// fitCheckFor assembles the check for a load that has not happened yet. It prices a .gguf and a safetensors directory the same way:
// shape-only, quant-priced element counts (estimateGGUFWeightBytes / estimateSafetensorsWeightBytes), never on-disk file size, because a
// safetensors checkpoint shrinks several-fold on quantization and an estimate that is wrong in the refusing direction is worse than none.
// Anything neither format resolves (a bare .giw path, an unreadable config, a directory in flux) is unknown, so proceed.
//
// PRICED AGAINST CURRENTLY-AVAILABLE MEMORY, NOT TOTAL RAM: a fixed fraction of total RAM assumes nothing else on a shared machine ever
// needs more than the rest. Weights are still subtracted here (unlike prefill_budget.go's request-time check) because guardFit runs before
// loadWeights (decoder/model.go): the weights this call is about to allocate are not yet resident, so the availability figure does not yet
// reflect their cost.
func fitCheckFor(path, quantName string, quant quantMode, opts Options) fitCheck {
	if quantName == "" {
		quantName = "f32"
	}
	f := fitCheck{
		name:       filepath.Base(path),
		quant:      quantName,
		availBytes: hostRAMAvailable(),
		noGuard:    loadKnob(opts, knobNoFitGuard) != "",
	}
	f.kvF16 = opts.KVPrecision == "f16"
	f.kvI8 = opts.KVQuant == "i8"
	if strings.HasSuffix(path, ".gguf") {
		f.isGGUF = true
		var expertBytes int64
		f.weightBytes, expertBytes = estimateGGUFWeightBreakdown(path, quant)
		if !opts.StreamWeights && expertBytes > 0 && opts.Backend == "cuda" &&
			(opts.MoECacheExperts || loadKnob(opts, knobMoECacheExperts) != "") {
			f.expertBytes = expertBytes
			f.cudaBuildBytes = 2*f.weightBytes + expertBytes
		}
		// srcFileBytes prices the TRANSIENT peak (see its doc comment), a term separate from the final resident estimate above; it does not
		// contradict the never-from-on-disk-size rule for weightBytes, which is about the final size shrinking on quantize. Only priced for a plain
		// resident load: StreamWeights means this path is transcoded to a .giw and re-loaded by a different Load call that never reaches
		// fitCheckFor.
		if !opts.StreamWeights {
			if n, ok := GGUFFileBytes(path); ok { // a split set's whole size, not its first shard's
				f.srcFileBytes = n
			}
		}
		g, err := OpenGGUFMmap(path)
		if err != nil {
			return f // unknown ⇒ proceed, same as always
		}
		defer g.Close()
		cfg, cerr := ggufConfig(g)
		if cerr != nil {
			return f
		}
		return f.priceCtxAndKV(cfg, opts)
	}
	// A safetensors directory (or any other non-.gguf path — .giw streamed-weights included):
	// openCheckpointMmap/loadConfig fail cleanly on anything that is not a plain safetensors
	// checkpoint, which is "unknown ⇒ proceed", same as every other unreadable source here.
	f.weightBytes = estimateSafetensorsWeightBytes(path, quant)
	cfg, cerr := loadConfig(os.DirFS(path), "config.json")
	if cerr != nil {
		return f
	}
	return f.priceCtxAndKV(cfg, opts)
}

// FitDescribe returns the same priced-terms sentence the fit guard's own refusal and warning messages use (fitCheck.arithmetic()), for a
// caller that wants to name resident-weight and mapped-source bytes in its own message without re-deriving decoder's pricing. It is the
// source of the "reason/pricing detail" Options.LoadAbort's doc promises a caller once ErrLoadAborted comes back.
func FitDescribe(path string, opts Options) (string, error) {
	opts = opts.withAutoBackend()
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		return "", err
	}
	return fitCheckFor(path, opts.Quant, quant, opts).arithmetic(), nil
}

// priceCtxAndKV fills in effCtx/pinned/kvBytes from a resolved Config: the ctx-pricing logic fitCheckFor's .gguf and safetensors branches
// share. Price at opts.ResidentContext when pinned, else the model's own MaxPositions, the worst case a real request can reach.
func (f fitCheck) priceCtxAndKV(cfg *Config, opts Options) fitCheck {
	f.cfg = cfg
	f.denseStreamable = denseStreamable(cfg)
	p, ok := kvPricingFor(cfg, opts)
	f.pinned = p.pinned
	if !ok {
		return f // no pin and the model's own max is unknown ⇒ can't price KV; proceed as before
	}
	f.effCtx, f.kvF16, f.kvI8 = p.ctx, p.kvF16, p.kvI8
	f.kvBytes = estimateKVBytes(cfg, f.effCtx, f.kvF16, f.kvI8)
	return f
}

// kvPrice is the KV a host fit guard prices: the context, whether the caller pinned it, and the precision.
type kvPrice struct {
	ctx         int
	pinned      bool
	kvF16, kvI8 bool
	metalSizing bool // priced as the Metal resident will allocate, not at the CPU's per-request ceiling
}

// kvPricingFor is what both host guards (priceCtxAndKV for .gguf/safetensors, guardGIWFit for .giw) price KV at: what the load will
// actually allocate.
//
//   - A load that will be Metal-resident (Metal compiled in, the resolved backend, and the architecture inside Metal's feature gate) holds
//     f16 KV, the only KV Metal ships, for MetalCtxDefault positions unless the caller pinned a context, clamped to the model's window.
//     Pricing it at Options.KVPrecision over the whole window would refuse or pin down a load that fits.
//   - Every other load: the CPU allocates KV per request, at Options.KVPrecision, up to the window, and that ceiling is what a long
//     request reaches.
//
// A Metal resident that then declines for memory falls back to the CPU, whose per-request KV this no longer prices at load. The weights
// term is unchanged by that fallback, and serve's -require-backend refuses rather than fall back.
//
// ok is false when nothing is pinned and the model's own maximum is unknown: nothing to price.
func kvPricingFor(cfg *Config, opts Options) (p kvPrice, ok bool) {
	p.pinned = opts.ResidentContext > 0
	p.kvF16, p.kvI8 = opts.KVPrecision == "f16", opts.KVQuant == "i8"
	if metalWillBeResident(cfg, opts) {
		p.metalSizing, p.kvF16 = true, true
		p.ctx = MetalCtxDefault
		if p.pinned {
			p.ctx = opts.ResidentContext
		}
		if cfg.MaxPositions > 0 && p.ctx > cfg.MaxPositions {
			p.ctx = cfg.MaxPositions
		}
		return p, true
	}
	switch {
	case p.pinned:
		p.ctx = opts.ResidentContext
		if cfg.MaxPositions > 0 && p.ctx > cfg.MaxPositions {
			p.ctx = cfg.MaxPositions // a pin past the model's own window prices no higher than the window
		}
	case cfg.MaxPositions > 0:
		p.ctx = cfg.MaxPositions // the worst case a real request can reach, unpinned
	default:
		return p, false
	}
	return p, true
}

// metalWillBeResident reports whether this load will build a Metal resident, as far as load time can tell: Metal is
// compiled into this binary, it is the resolved backend, and the architecture passes Metal's feature gate (the check
// metal/backend.go applies first). Resolved on a copy of cfg: resolveArchitecture may backfill fields.
func metalWillBeResident(cfg *Config, opts Options) bool {
	if opts.Backend != "metal" || !slices.Contains(CompiledBackends(), "metal") {
		return false
	}
	c := *cfg
	arch, _, err := resolveArchitecture(&c)
	if err != nil {
		return false
	}
	return len(missingFeatures(arch.residentFeatures(), residentBackendFeatures["metal"])) == 0
}

// smallerFittingContext solves for the largest context <= f.effCtx whose weights+KV fit the budget, floored at ctxFloor. Only meaningful
// when the caller did not pin a context: a pin is an explicit request and is refused outright (see guardFit). estimateKVBytes is not
// exactly linear in ctx once a sliding-window layer's cost flattens past its window, but it is monotonic non-decreasing, so a binary
// search finds the largest fitting ctx exactly.
func (f fitCheck) smallerFittingContext() (int, bool) {
	if f.pinned || f.cfg == nil {
		return 0, false
	}
	// The CUDA build peak has no KV in it, so a smaller context cannot bring it under the budget.
	if f.cudaBuildBytes > f.budget() {
		return 0, false
	}
	// srcFileBytes is a fixed term like weightBytes: it does not shrink with ctx, so it comes out of the budget too, or a load whose
	// weights+file already exceed the budget would be offered a smaller-context "fit" that only re-prices KV.
	available := f.budget() - f.weightBytes - f.srcFileBytes
	if available <= 0 {
		return 0, false
	}
	fits := func(ctx int) bool { return estimateKVBytes(f.cfg, ctx, f.kvF16, f.kvI8) <= available }
	if !fits(ctxFloor) {
		return 0, false
	}
	if fits(f.effCtx) {
		// KV genuinely costs nothing extra all the way to effCtx (e.g. an all-recurrent model with
		// no attention layers at all) — the caller only reaches this function when the ORIGINAL
		// fits() was false, so this is a defensive fallback, not the expected common case.
		return f.effCtx, true
	}
	lo, hi := ctxFloor, f.effCtx
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if fits(mid) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return lo, true
}

// capNote is the banner line printed when the guard auto-pins a smaller context — the default
// outcome for an unpinned load that would not fit at the model's own maximum, because it leaves
// the user with a working server and a visible limit rather than a refusal on the model the
// README told them to pull.
func (f fitCheck) capNote(ctx int) string {
	kv := estimateKVBytes(f.cfg, ctx, f.kvF16, f.kvI8)
	maxCtx := 0
	if f.cfg != nil {
		maxCtx = f.cfg.MaxPositions
	}
	return fmt.Sprintf(
		"decoder: context capped at %d (model allows %d) so that KV fits: weights %.1f GB + KV %.1f GB of budget %.1f GB — pass -ctx to choose, or -stream-weights to lift",
		ctx, maxCtx, float64(f.weightBytes)/fitGB, float64(kv)/fitGB, float64(f.budget())/fitGB)
}

// weightAllocs counts entries into loadWeights — the call that turns a checkpoint into resident
// heap. It exists so "the guard refused BEFORE allocating" is an assertion rather than a
// deduction from the error text; a guard placed one line too late produces the same error and the
// same swap storm.
var weightAllocs atomic.Int64

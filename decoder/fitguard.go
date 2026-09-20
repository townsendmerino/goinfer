package decoder

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// The load-time fit guard: refuse a model that cannot fit in RAM BEFORE allocating it, naming
// the numbers and the flag that fixes it.
//
// WHY. Cold-user run 2026-09-06 scenario D (docs/measurements/cold-user-2026-09-06.md): a 21 GB
// 35B-A3B on a 16 GB Mac drove the box +7,819 MB into swap in FIVE SECONDS with no message —
// "the tool never told me it would not fit, the machine told me". `serve --help` already names
// -stream-weights for exactly that model and that RAM, and with it RSS capped at 8.95 GB with
// zero swapouts. The engine does the right thing; nothing told the user it existed.
//
// SCOPE, deliberately. This is Phase 0 of docs/tasks/task-fit-to-hardware.md and nothing else: a
// refusal with arithmetic. It does NOT plan a configuration and it does NOT flip -stream-weights
// on for you; both are that doc's later phases, and choosing for the user is a bigger change than
// telling them.
//
// EVERY UNKNOWN PROCEEDS. An unreadable RAM figure, an unsupported source format, a zero-byte
// estimate — each returns "don't know" and the load continues. The guard's failure mode must be
// letting a doomed load through (the status quo), never refusing one that would have run.

// fitMemFraction is the share of the fit check's base memory figure the WEIGHTS alone may occupy.
// Same figure and same provenance as metal/backend.go's residentMemFraction: ONE measured failure
// (11.28 GB of 16 GB = 70.5% thrashed to swap exhaustion), so a threshold rather than a swept
// curve. The rest is not slack — KV, scratch, the tokenizer, and the operating system live there
// too. Originally fractioned against TOTAL physical RAM; fitCheckFor now fractions it against
// CURRENTLY AVAILABLE memory instead (R13-follow-on) — the threshold itself is unchanged, only
// what it is a fraction OF.
const fitMemFraction = 0.70

// fitWarnRatio is how close to the budget the load has to come before the banner prints the
// arithmetic unasked. At 0.75 the message appears within 25% of the refusal, so the user sees the
// cliff on the run BEFORE the one that steps off it.
const fitWarnRatio = 0.75

// hostRAM is indirected so a test can inject a machine's worth of RAM instead of needing one.
var hostRAM = HostRAMBytes

// hostRAMAvailable is the same indirection for CURRENTLY AVAILABLE memory (prefill_budget.go,
// R13-follow-on) — a live figure, never cached, unlike hostRAM's total-RAM snapshot.
var hostRAMAvailable = HostRAMAvailableBytes

// ctxFloor is the smallest context this guard will auto-pin down to when the caller did not pin
// one and the model's own maximum does not fit. Below this a context is not useful enough to hand
// a user silently — refuse instead, the way R3 already does for the rest of the model. Named,
// not measured: R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md) did not measure a
// real floor, and 2048 is stated as a product choice pending a real one.
const ctxFloor = 2048

// fitCheck is the arithmetic, separated from every source of it so it can be driven with the
// numbers from a measurement rather than a 21 GB checkpoint.
type fitCheck struct {
	name        string // what to call the model in the message
	quant       string // the requested quant, named because it moves the weight term the most
	weightBytes int64  // estimated resident weight bytes AT THAT QUANT
	kvBytes     int64  // KV at effCtx (see below) — always priced now, not only when pinned
	availBytes  int64  // CURRENTLY AVAILABLE memory at check time (R13-follow-on), 0 when unknown

	// srcFileBytes is the on-disk size of a plain (non-streamed) .gguf SOURCE file, priced as an
	// ADDITIONAL transient term alongside weightBytes+kvBytes.
	//
	// MEASURED, 2026-09-18, docs/measurements/cold-user-2026-09-18-nobara-pc.md Scenario D,
	// reproduced directly on nobara-pc (the same box) with a heap profile + /proc RSS sampling
	// around a real `decoder.Load` of gpt-oss-20b (12.11 GB MXFP4 GGUF, int4 resident weights
	// 12.58 GB): peak RSS reached ~24.5 GB — matching weightBytes+fileSize (12.58+12.11=24.69 GB)
	// to within 2%, NOT the ~12.58 GB this guard priced before this field existed. Once
	// decoder.Load returns and the mmap is closed, RSS drops back to ~13.0 GB, confirming the
	// extra ~12 GB was the mmap'd SOURCE file, not a second copy of the resident weights.
	//
	// WHY: loadGGUFWeights's own comment ("mmap, not heap-read: the raw quantized bytes stay in
	// reclaimable page cache") is true but incomplete — those pages are reclaimable in principle,
	// but the mapping (embed.OpenGGUFMmap) is held open for the ENTIRE build (buildWeightsFromGGUF
	// runs parallelLayers across every layer before the deferred g.Close() in loadGGUFWeights
	// finally runs), and RowDequantizer's per-row reads touch essentially every page of the file
	// by the time the model is fully quantized — so for most of the load, the WHOLE source file is
	// resident in RAM at the same time as the (also whole, by the end) resident weight set. This is
	// not double-buffering of the SAME data — it is source-plus-destination coexisting because
	// nothing releases the source pages incrementally as each tensor is consumed. That release
	// would need per-tensor madvise inside aikit/embed (a separate module, out of scope here); this
	// guard fixes what goinfer controls — pricing the real peak instead of only the final size.
	//
	// Only meaningful for a plain resident `.gguf` load (isGGUF && !streamWeights): a `.giw` load
	// mmaps its own weight blob directly (no separate dequant-and-copy pass) and a safetensors
	// directory's loader has its own accounting; StreamWeights (once transcoded to .giw) also
	// leaves this repo through a different Load branch entirely (see model.go's ".giw" branch,
	// which never reaches fitCheckFor at all — a separate, pre-existing gap this change does not
	// touch). Zero when not applicable, so an existing fitCheck literal built by a test or another
	// caller is unaffected.
	srcFileBytes int64

	// cudaBuildBytes is the HOST peak of building a CUDA C' expert cache (--backend cuda
	// --moe-cache-experts) from a plain .gguf: 2*weightBytes + expertBytes. Zero when that path is
	// not in play. It REPLACES the weights+KV+srcFileBytes total when it is larger rather than adding
	// to it, because the phases do not overlap: decoder.Load unmaps the source file before
	// cuda.BuildResident starts, and on the resident path KV lives in VRAM.
	//
	// MEASURED 2026-09-19 on the real gpt-oss-20b (docs/measurements/cold-user-2026-09-18-nobara-pc.md
	// follow-up), GC-traced with a per-region /proc breakdown: after Load the canonical weights sit on
	// the Go heap (~13 GB live); BuildResident then host-packs every layer (a second, packed copy of
	// the same size) and cacheWQ copies each expert stack into pinned host memory (~10 GB, counted
	// as neither anon nor file). Peak RSS 39.1 GB against this model's 2*12.2 + 11.1 = 35.5 GB — the
	// ~9% remainder is Go heap slack, not priced here. Before cuda.packWeightStack stopped regrowing
	// its slices the same load reached 50 GB+ and was killed unfinished.
	cudaBuildBytes int64
	expertBytes    int64

	// cfg, effCtx, pinned, kvF16, kvI8 carry what R13's re-pricing needs to recompute KV at a
	// SMALLER context when the load does not fit and the caller never pinned one — see
	// smallerFittingContext. cfg is nil when the GGUF's config could not be read (proceeds as
	// before: kvBytes stays 0, "unknown" wins as it always has).
	cfg    *Config
	effCtx int  // the context KV was priced at: opts.ResidentContext if pinned, else cfg.MaxPositions
	pinned bool // true when the caller explicitly requested effCtx (an explicit request that
	// cannot be honoured is refused, never silently downgraded — the G-07 principle)
	kvF16 bool
	kvI8  bool

	// denseStreamable is true when a -stream-weights retry after this refusal would engage
	// decoder/layerpaging.go's windowed dense pager — see denseStreamable(cfg)'s own doc comment.
	// Carried into declineErr()'s *FitDeclineError so a caller can decide whether an automatic
	// retry is sound without re-deriving this from the arch registry itself.
	denseStreamable bool

	// isGGUF is true when the source that was priced is a .gguf FILE, false for a safetensors
	// DIRECTORY (M-30, docs/audit-2026-09-10.md) — remedy() needs this because -stream-weights
	// only has anything to do for a .gguf source; see that function's own doc comment.
	isGGUF bool
}

func (f fitCheck) need() int64 {
	n := f.weightBytes + f.kvBytes + f.srcFileBytes
	if f.cudaBuildBytes > n {
		n = f.cudaBuildBytes
	}
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

// remedy names -stream-weights (a .gguf source) or GOINFER_NO_FIT_GUARD=1 (a safetensors
// directory) and says what it will do, because the user who reads this message is by definition
// the one who did not know the option existed.
//
// M-30 (docs/audit-2026-09-10.md): this used to return the -stream-weights text unconditionally,
// on the stale claim that only a .gguf can reach a refusal. P9(b) (b7715ca) made a safetensors
// DIRECTORY reachable here too (it now prices those, correctly — before it they silently always
// "fit"), and -stream-weights genuinely does nothing for one: serve's manual and auto-retry gates
// are both .gguf-suffix-only, and decoder.Load ignores Options.StreamWeights for a directory
// input — so the flag was being recommended as a fix that could not possibly change anything,
// producing an identical refusal after the user did what they were told.
//
// cmd/prequant CAN build a streamable .giw from a directory (transcodeDir) — but it is not
// offered as the remedy here, because transcodeDir loads the checkpoint fully resident to
// serialize it (unlike the .gguf path's true one-layer-at-a-time streaming transcode), so it
// hits this exact guard for the exact same reason and cannot help a checkpoint that genuinely
// does not fit. GOINFER_NO_FIT_GUARD=1 is named directly instead — it is already the correct,
// working escape hatch for the case this guard's own 70% margin is being conservative about (a
// checkpoint that would actually fit), and cmd/prequant becomes a real, valuable one-time step
// only once that variable lets its own internal load through.
func (f fitCheck) remedy() string {
	if f.isGGUF {
		return "Re-run goinfer-serve with -stream-weights: it caches the model as a sidecar .giw once, " +
			"then pages weights out of it on demand instead of holding them all resident."
	}
	return "This is a safetensors checkpoint — -stream-weights only helps a .gguf source. If you " +
		"believe this machine can actually hold it (this guard's 70% margin is deliberately " +
		"conservative), set GOINFER_NO_FIT_GUARD=1 and re-run; cmd/prequant can then build a " +
		"streamable .giw from it once, for every later run."
}

// ErrWontFitResident is wrapped into every load-time fit-guard refusal (declineErr), so a caller
// can detect the refusal with errors.Is/errors.As instead of parsing the message text — e.g. to
// decide whether an automatic -stream-weights retry is worth attempting (see FitDeclineError).
var ErrWontFitResident = errors.New("decoder: model will not fit resident RAM")

// FitDeclineError is what declineErr returns instead of a bare error.
type FitDeclineError struct {
	msg string

	// DenseStreamable is true when a -stream-weights retry after this refusal would engage
	// decoder/layerpaging.go's windowed dense pager — the mechanism tasks/task-fit-to-hardware.md's
	// CPU placement piece measured as sound for an AUTOMATIC retry
	// (docs/tasks/task-gpu-paths-2026-09.md). It is false for MoE models and "own-forward" families
	// (gemma4, nemotron-h-moe, lfm2): MoE CPU weight streaming is a documented, MEASURED failure
	// mode instead — docs/benchmarks.md "M35/M26 on the Mac" ran a real 20 GB MoE checkpoint
	// through the CPU-staged --stream-weights path for 2h10min with ZERO completions (RSS pinned
	// at ~3.2 GB against a 20 GB model — re-reading weights from disk essentially every token, no
	// useful cache retention), and that run is very likely what produced a genuine kernel panic on
	// this machine shortly afterward. An automatic retry into that path would risk repeating the
	// same incident silently, so it stays a manual, explicit choice (-stream-weights typed by
	// hand) rather than something the guard does on the caller's behalf.
	DenseStreamable bool
}

func (e *FitDeclineError) Error() string { return e.msg }
func (e *FitDeclineError) Unwrap() error { return ErrWontFitResident }

// denseStreamable mirrors newLayerPager's own exclusions (decoder/layerpaging.go) exactly, so the
// fit guard's retry decision and the pager's own decision to actually build one can never
// disagree: an MoE model uses expertPager instead (see FitDeclineError.DenseStreamable's doc for
// why that path stays manual), and an "own-forward" family (resolved dynamically via
// arch.ownForward() — not a hand-written list, per the C-02/C-03 lfm2 miss newLayerPager's own
// comment names) runs a layer loop that never calls enterLayer, so layerPager would never engage
// for it either. cfg == nil (header unreadable) answers false: "don't know" must not attempt a
// retry the fit guard cannot vouch for, the same "every unknown proceeds [without acting]"
// discipline this file states at the top for the refusal path itself.
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
	// Suggesting a smaller quant to someone already at int4 is noise, and noise in a refusal is
	// how the useful line gets skipped.
	//
	// AND int4 IS NOT ALWAYS THE SMALLER ONE. On arm64-with-dotprod (and AVX2-without-VNNI) the
	// loader keeps a repacked second copy of the nibbles beside the canonical ones, so int4
	// measures ~1.25 bytes/element against int8's ~1.02 — MORE resident memory, not less
	// (measured in CI on darwin/arm64, docs/tasks/task-first-hour.md). Offering "int4, the smallest"
	// there would send a user who is already out of memory in the wrong direction, so the line is
	// derived from the same measurement the arithmetic above uses rather than from the nominal
	// bit width.
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

// guardFit runs the check. It returns the context to PIN — 0 meaning "leave the caller's request
// alone", nonzero meaning "the guard chose this smaller one, apply it" — and the refusal, or
// (0, nil) to proceed unchanged. It prints the arithmetic to stderr when the load is within
// fitWarnRatio of refusing, or when it auto-pins.
//
// R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): before this, an unpinned load
// that did not fit at its own maximum context simply loaded anyway (kvBytes was 0, so `fits()`
// only ever saw the weight term) — the guard existed and said nothing, because nothing asked it
// the question a real request would ask. Three outcomes now, in order: an explicit pin that does
// not fit is REFUSED (G-07: an explicit request that cannot be honoured is refused, not silently
// downgraded); an unpinned load that does not fit at the model's maximum but DOES fit at some
// smaller context ≥ ctxFloor is auto-pinned to that context, reported, and proceeds; an unpinned
// load that does not fit even at ctxFloor is refused, same as a pinned one.
func guardFit(f fitCheck) (int, error) {
	if os.Getenv("GOINFER_NO_FIT_GUARD") != "" {
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

// quantBytesPerElem is the resident cost of one weight ELEMENT of a 2-D matmul matrix under each
// quant mode — MEASURED by running a probe matrix through the loader's own quantization, not
// derived from the nominal bit width.
//
// WHY IT IS MEASURED. The arithmetic answer ("int4 is 0.5 bytes plus a scale per group of 32, so
// 0.625") is right about the encoding and wrong about the FOOTPRINT, because the loader repacks:
// repackW4A8Row4IfEligible on arm64 and repackW4A8SplitHalfIfEligible on AVX2-without-VNNI amd64
// both ALLOCATE A SECOND BUFFER and keep the canonical nibbles alongside it, so an int4 weight
// really costs about twice its encoding on those hosts. wmBytes counts both, correctly.
//
// Caught by CI, not by reasoning: TestFitEstimate_agreesWithResidentWeightBytes passed on
// linux/amd64 (ratio 0.96) and failed on darwin/arm64 at ratio 0.53 — estimate 104256 against
// 195584 accounted. Apple Silicon is exactly the platform the fit guard exists for
// (docs/measurements/cold-user-2026-09-06.md was a 16 GB M1 Pro), so a constant tuned on the
// developer's box was ~1.8x low precisely where it mattered. Measuring through the real path
// tracks the arch, the CPU features, and any repack added later, none of which a constant can.
//
// THE RESIDUAL, stated rather than hidden. The probe is 256x256, which the repacks accept (rows a
// multiple of 4, cols a multiple of int4GroupSize). A model built entirely from matrices the
// repack REJECTS would be over-priced by up to that factor — the direction that can refuse a
// model which would have fit. Real transformer matrices are multiples of 4 and 32 by
// construction, the 70% budget carries slack of its own, and GOINFER_NO_FIT_GUARD is named in the
// refusal; that is the trade, taken deliberately, because the alternative was a guard that
// under-reports by ~2x on the platform it was written for.
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
	if probe == quantInt4Mix {
		probe = quantInt4
	}
	wm := quantizeWM(linalg.WrapF32(f32, n, n), probe)
	if b := wmBytes(&wm); b > 0 {
		return float64(b) / float64(n*n)
	}
	// quantizeWM is a no-op for quantNone and anything it does not handle; f32 is the answer then.
	return 4
}

// f32PinnedTensorName reports whether a checkpoint tensor name is one the loaders ALWAYS keep
// resident at f32 regardless of the requested quant (M-29, docs/audit-2026-09-10.md): a Mamba-2
// mixer (Granite-4.0-H's "mamba.*" prefix, GGUF "ssm_*") or an MLA attention projection
// (DeepSeek/Bailing — GGUF "attn_q_a"/"attn_q_b"/"attn_kv_a_mqa"/"attn_kv_b", safetensors
// "q_a_proj"/"q_b_proj"/"kv_a_proj"/"kv_b_proj").
//
// DELIBERATELY NOT a bare "mixer."/"in_proj"/"out_proj" match: Nemotron-H shares the "mixer."
// prefix across its Mamba-2 (f32), attention, MLP, and MoE layer types (decoder/weights.go's
// buildNemotronWeights loadLayer — same prefix, different block kinds), and "in_proj"/"out_proj"
// alone collide with LFM2's conv mixer and qwen3.5/qwen3_next's DeltaNet hybrid, where SOME
// "linear_attn.in_proj_*" sub-tensors are f32 and OTHERS (loaded via mkQ two lines away in
// weights.go) are genuinely quantized — a name-only classifier cannot safely tell those apart.
// Nemotron-H's Mamba-2 in_proj/out_proj/conv1d ARE matched below, but via the full
// "mixer.in_proj"/"mixer.out_proj"/"mixer.conv1d" compound (never bare "mixer." or bare
// "in_proj"), which no other Nemotron block-kind tensor name contains.
//
// KNOWN RESIDUAL GAP, left unfixed rather than over-matched: MLA's output projection
// (self_attn.o_proj.weight / Bailing's dense.weight) is ALSO f32-pinned
// (loadDeepseekAttn/loadDeepseekAttnGGUF), but "o_proj"/"dense.weight" are ordinary, WIDELY-used
// quantizable tensor names in every non-MLA family — matching them bare would misclassify most
// of the registry. Left as a small, honest under-estimate (verified in
// TestFitEstimate_f32PinnedMixersAgreeWithResident: o_proj is a minority of MLA's own attention
// weight, not the dominant term) rather than risk a much larger false-positive elsewhere.
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

// visionTowerTensorName reports whether name belongs to a bundled multimodal vision tower or
// projector rather than the text decoder (M-29, docs/audit-2026-09-10.md). decoder/weights.go's
// text loader already never REQUESTS these names (see its own comment on this exact prefix
// pair) — it only reads specific tensor names, so they are silently skipped there. This
// estimator instead walks EVERY tensor in the checkpoint's metadata, so it must recognize and
// exclude them explicitly or they get swept into the text model's per-element quant price.
func visionTowerTensorName(name string) bool {
	return strings.Contains(name, "vision_tower.") || strings.Contains(name, "multi_modal_projector.")
}

// gptqAWQAuxSuffix reports whether name is a GPTQ/AWQ auxiliary tensor (qzeros/g_idx/scales,
// decoder/gptq.go, decoder/awq.go) that is fully consumed during the one-time
// reconstruct-then-requantize step and never itself kept resident — pricing it as an ordinary
// weight matrix at the target quant's rate would double-count storage the real load path
// discards after use. M-29 (docs/audit-2026-09-10.md).
func gptqAWQAuxSuffix(name string) bool {
	for _, s := range []string{".qzeros", ".g_idx", ".scales"} {
		if strings.HasSuffix(name, s) {
			return true
		}
	}
	return false
}

// gptqAWQPackFactor is how many 4-bit codes GPTQ/AWQ pack into each element along qweight's
// packed dimension (decoder/gptq.go: "8 4-bit codes pack into each int32... qweight packs the
// input dim") — so qweight's ON-DISK shape under-counts the LOGICAL [in,out] matrix it
// represents by exactly this factor. M-29 (docs/audit-2026-09-10.md): without correcting for it,
// the estimator priced the packed shape's element count at the target quant's bytes/elem,
// landing ~8x low — the checkpoint is fully unpacked and RE-quantized to the requested resident
// quant at load time (decoder/weights.go's loadProj), not kept in its packed on-disk form.
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
	g, err := embed.OpenGGUFMmap(path)
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
		// M-29 (docs/audit-2026-09-10.md): Mamba-2/MLA tensors stay f32 regardless of the
		// requested quant — price them at what they actually cost, not the ambient rate.
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

// estimateSafetensorsWeightBytes is estimateGGUFWeightBytes's safetensors twin: same
// shape-only, quant-independent accounting (sum every tensor's element count, price 2-D+
// tensors at the target quant, 1-D norms/biases at f32), reusing openCheckpointMmap so single-file
// and sharded (model.safetensors.index.json) checkpoints are handled identically to a real load.
//
// WHY SHAPE-ONLY, NOT ON-DISK FILE SIZE (see fitCheckFor's own header comment on why safetensors
// was excluded before this function existed): a safetensors checkpoint is usually f32 or bf16 on
// disk and shrinks several-fold once quantized on load, so pricing the ON-DISK bytes would refuse
// loads that fit comfortably — wrong in the refusing direction, which this guard's own design
// principle treats as worse than not pricing at all. Reading SHAPES (element counts, which do not
// depend on the on-disk dtype) and pricing them at the REQUESTED load quant is the same technique
// estimateGGUFWeightBytes already uses — a GGUF file is quantized on disk too, and that function
// never reads its byte size either.
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
		// M-29 (docs/audit-2026-09-10.md): a GPTQ/AWQ auxiliary tensor is fully consumed during
		// reconstruction and never itself kept resident — price it at zero, not as an ordinary
		// weight matrix (which would double-count storage the real load path discards).
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
		// M-29: Mamba-2/MLA tensors stay f32 regardless of the requested quant.
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
		total += quantBytesPerElem(q) * float64(n)
	}
	return int64(total)
}

// kvBytesPerPosition is the KV cost of ONE position, so both estimateKVBytes and
// smallerFittingContext (R13) share the identical per-position rate — the cost is exactly linear
// in context, so solving "the largest context that fits" is arithmetic, not a search.
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

// estimateKVBytes is the KV cache at ctx positions.
//
// R13 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): this used to return 0 whenever
// no context was explicitly pinned, reasoning that "the CPU cache grows with the conversation
// rather than being allocated up front, so counting a context nobody asked for would refuse
// models that run fine for short turns." That reasoning is true about short turns and wrong about
// what a user actually sends: a 7B int4 model priced at "79% of budget" (KV priced at 0) reached
// 14 GB RSS and swapped the machine hard on its first real agent request — an opencode system
// prompt plus tool schema, tens of thousands of tokens, well inside the model's own context
// window. "No context pinned" does not mean "no KV ever allocated"; it means the ceiling is
// whatever the model's own maximum context is, because nothing else bounds the CPU/Metal-staged
// KV cache's growth. ctx is now the caller's job to choose correctly (fitCheckFor picks
// opts.ResidentContext when pinned, else cfg.MaxPositions) — this function just prices whatever
// it is given.
func estimateKVBytes(cfg *Config, ctx int, kvF16, kvI8 bool) int64 {
	if ctx <= 0 {
		return 0
	}
	// M-28 (docs/audit-2026-09-10.md): kvBytesPerPosition's flat formula overpriced hybrid
	// (DeltaNet/conv/Mamba), sliding-window, and MLA models 3-7x. Resolve the real per-layer
	// geometry when possible (kvBytesForCtx, decoder/arch.go) and fall back to the flat formula
	// only when the architecture cannot be resolved at all — not a real load (every real GGUF/
	// safetensors config that reaches this point already resolved one further up in
	// fitCheckFor/denseStreamable), but a synthetic Config with no registered model_type, the
	// shape several of this file's own unit tests construct directly.
	if arch, _, err := resolveArchitecture(cfg); err == nil && arch != nil {
		return kvBytesForCtx(arch, ctx, kvF16, kvI8)
	}
	return kvBytesPerPosition(cfg, kvF16, kvI8) * int64(ctx)
}

// fitCheckFor assembles the check for a load that has not happened yet. It prices both a .gguf
// and a safetensors directory (P9b, docs/multimodal.md) the SAME way — shape-only, quant-priced
// element counts (estimateGGUFWeightBytes / estimateSafetensorsWeightBytes) — never from on-disk
// file size: a safetensors checkpoint is usually f32 or bf16 on disk and shrinks several-fold once
// quantized on load, so pricing the on-disk bytes would refuse models that fit comfortably. An
// estimate that is wrong in the refusing direction is worse than none, which is why this waited
// for the shape-based technique rather than shipping the naive (and wrong) file-size one earlier.
// Anything neither format resolves (a bare .giw path, an unreadable config, in-flux directory) is
// "unknown ⇒ proceed", same as always.
//
// PRICED AGAINST CURRENTLY-AVAILABLE MEMORY, NOT TOTAL RAM (R13-follow-on,
// docs/measurements/cold-user-2026-09-07-macbook-arm64.md's SECOND live re-run). The first
// version of this function read hostRAM() — total physical RAM, a fixed number that assumes
// nothing else on the machine ever needs more than the 30% fitMemFraction reserves. The live
// re-run of R13's own fix (which changed prefill_budget.go's request-time check the same way)
// found the load-time guard's version of this bug too: on a real, shared Mac, swap began within
// 15 SECONDS OF LOAD COMPLETING, with the server sitting idle and no request in flight yet — proof
// the "30% of total RAM is always enough for everything else" assumption is what was actually
// wrong, not merely a per-request pricing gap. Weights ARE still subtracted here (unlike
// prefill_budget.go's request-time check): at LOAD time the weights this call is about to allocate
// are NOT YET resident (guardFit runs before loadWeights, decoder/model.go), so the current
// availability figure does not yet reflect their cost the way it does for an already-loaded model.
func fitCheckFor(path, quantName string, quant quantMode, opts Options) fitCheck {
	if quantName == "" {
		quantName = "f32"
	}
	f := fitCheck{
		name:       filepath.Base(path),
		quant:      quantName,
		availBytes: hostRAMAvailable(),
	}
	f.kvF16 = opts.KVPrecision == "f16"
	f.kvI8 = opts.KVQuant == "i8"
	if strings.HasSuffix(path, ".gguf") {
		f.isGGUF = true
		var expertBytes int64
		f.weightBytes, expertBytes = estimateGGUFWeightBreakdown(path, quant)
		if !opts.StreamWeights && expertBytes > 0 && opts.Backend == "cuda" &&
			(opts.MoECacheExperts || os.Getenv("GOINFER_MOE_CACHE_EXPERTS") != "") {
			f.expertBytes = expertBytes
			f.cudaBuildBytes = 2*f.weightBytes + expertBytes
		}
		// srcFileBytes prices the TRANSIENT peak (see its own doc comment), not the final resident
		// weight estimate above — the two are deliberately separate terms and this does not
		// contradict this function's own "never from on-disk file size" rule for weightBytes: that
		// rule is about not mis-estimating the FINAL size from a shrinking-on-quantize file; this is
		// about the SOURCE file staying mapped resident for the whole build, on top of whatever the
		// final size turns out to be. Only priced for a plain resident load: StreamWeights means
		// this exact path is about to be transcoded to a .giw and re-loaded from THAT (a different
		// Load call, a different fitCheckFor invocation, currently not priced at all — see
		// srcFileBytes's doc comment on that pre-existing, separate gap).
		if !opts.StreamWeights {
			if fi, serr := os.Stat(path); serr == nil {
				f.srcFileBytes = fi.Size()
			}
		}
		g, err := embed.OpenGGUFMmap(path)
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

// priceCtxAndKV fills in effCtx/pinned/kvBytes from a resolved Config — the ctx-pricing logic
// fitCheckFor's .gguf and safetensors branches share verbatim (R13: price at opts.ResidentContext
// when pinned, else the model's own MaxPositions, the worst case a real request can reach).
func (f fitCheck) priceCtxAndKV(cfg *Config, opts Options) fitCheck {
	f.cfg = cfg
	f.denseStreamable = denseStreamable(cfg)
	f.pinned = opts.ResidentContext > 0
	switch {
	case f.pinned:
		f.effCtx = opts.ResidentContext
		if cfg.MaxPositions > 0 && f.effCtx > cfg.MaxPositions {
			f.effCtx = cfg.MaxPositions // a pin past the model's own window prices no higher than the window
		}
	case cfg.MaxPositions > 0:
		f.effCtx = cfg.MaxPositions // R13: the worst case a real request can reach, unpinned
	default:
		return f // no pin and the model's own max is unknown ⇒ can't price KV; proceed as before
	}
	f.kvBytes = estimateKVBytes(cfg, f.effCtx, f.kvF16, f.kvI8)
	return f
}

// smallerFittingContext solves for the largest context ≤ f.effCtx whose weights+KV fit the
// budget, floored at ctxFloor. Only meaningful when the caller did not pin a context — a pin is
// an explicit request and is refused outright rather than silently downgraded (see guardFit).
//
// M-28 (docs/audit-2026-09-10.md): this used to divide the budget by a single flat per-position
// rate, exact only because the flat formula priced every position identically. estimateKVBytes
// is no longer exactly linear in ctx once a sliding-window layer's cost flattens past its own
// window — but it IS still monotonic non-decreasing (more context never needs LESS KV), so a
// binary search finds the largest fitting ctx exactly, the same guarantee the division used to
// give for free.
func (f fitCheck) smallerFittingContext() (int, bool) {
	if f.pinned || f.cfg == nil {
		return 0, false
	}
	// srcFileBytes is a FIXED term exactly like weightBytes (see its own doc comment) — it does not
	// shrink when ctx shrinks, so it has to come out of the budget here too, or a load whose
	// weights+file already exceed the budget would still get offered a smaller-context "fit" that
	// only ever re-prices KV.
	// The CUDA build peak has no KV in it, so a smaller context cannot bring it under the budget.
	if f.cudaBuildBytes > f.budget() {
		return 0, false
	}
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

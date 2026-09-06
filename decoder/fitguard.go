package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/townsendmerino/aikit/embed"
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
// SCOPE, deliberately. This is Phase 0 of docs/task-fit-to-hardware.md and nothing else: a
// refusal with arithmetic. It does NOT plan a configuration and it does NOT flip -stream-weights
// on for you; both are that doc's later phases, and choosing for the user is a bigger change than
// telling them.
//
// EVERY UNKNOWN PROCEEDS. An unreadable RAM figure, an unsupported source format, a zero-byte
// estimate — each returns "don't know" and the load continues. The guard's failure mode must be
// letting a doomed load through (the status quo), never refusing one that would have run.

// fitMemFraction is the share of physical RAM the WEIGHTS alone may occupy. Same figure and same
// provenance as metal/backend.go's residentMemFraction: ONE measured failure (11.28 GB of 16 GB
// = 70.5% thrashed to swap exhaustion), so a threshold rather than a swept curve. The rest is not
// slack — KV, scratch, the tokenizer, and the operating system live there too.
const fitMemFraction = 0.70

// fitWarnRatio is how close to the budget the load has to come before the banner prints the
// arithmetic unasked. At 0.75 the message appears within 25% of the refusal, so the user sees the
// cliff on the run BEFORE the one that steps off it.
const fitWarnRatio = 0.75

// hostRAM is indirected so a test can inject a machine's worth of RAM instead of needing one.
var hostRAM = HostRAMBytes

// fitCheck is the arithmetic, separated from every source of it so it can be driven with the
// numbers from a measurement rather than a 21 GB checkpoint.
type fitCheck struct {
	name        string // what to call the model in the message
	quant       string // the requested quant, named because it moves the weight term the most
	weightBytes int64  // estimated resident weight bytes AT THAT QUANT
	kvBytes     int64  // KV at the requested context, or 0 when no context was pinned
	ramBytes    int64  // physical RAM, 0 when unknown
}

func (f fitCheck) need() int64   { return f.weightBytes + f.kvBytes }
func (f fitCheck) budget() int64 { return int64(float64(f.ramBytes) * fitMemFraction) }

// known reports whether both sides of the comparison are real numbers. Anything else proceeds.
func (f fitCheck) known() bool { return f.ramBytes > 0 && f.weightBytes > 0 }

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
	fmt.Fprintf(&b, "%s needs ~%.1f GB resident at quant %s", f.name, float64(f.weightBytes)/fitGB, f.quant)
	if f.kvBytes > 0 {
		fmt.Fprintf(&b, " + %.1f GB KV = %.1f GB", float64(f.kvBytes)/fitGB, float64(f.need())/fitGB)
	}
	fmt.Fprintf(&b, "; this machine has %.1f GB RAM (budget %.1f GB = %.0f%%)",
		float64(f.ramBytes)/fitGB, float64(f.budget())/fitGB, fitMemFraction*100)
	return b.String()
}

// warning is what the banner prints when the load fits but only just.
func (f fitCheck) warning() string {
	return fmt.Sprintf("decoder: fit is tight — %s (%.0f%% of budget). %s",
		f.arithmetic(), f.ratio()*100, f.remedy())
}

// remedy names -stream-weights and says what it will do, because the user who reads this message
// is by definition the one who did not know the flag existed.
//
// Only a .gguf can reach a refusal (fitCheckFor prices nothing else), and goinfer-serve
// transcodes a .gguf to a sidecar .giw on first use — so the flag alone is the whole remedy, with
// no manual prequant step. A .giw is already mmap-backed and pageable, which is why it is never
// refused and why this text does not need a second branch.
func (f fitCheck) remedy() string {
	return "Re-run goinfer-serve with -stream-weights: it caches the model as a sidecar .giw once, " +
		"then pages weights out of it on demand instead of holding them all resident."
}

// declineErr is the refusal. It is an error, not a warning, because the measured alternative is a
// machine that stops responding: the user cannot read a warning on a box that is thrashing.
func (f fitCheck) declineErr() error {
	// Suggesting a smaller quant to someone already at int4 is noise, and noise in a refusal is
	// how the useful line gets skipped.
	alt := "  Or run a smaller model.\n"
	if f.quant != "int4" && f.quant != "int4mix" {
		alt = "  Or run a smaller model, or --quant int4 (the smallest, and roughly half the RAM of int8).\n"
	}
	return fmt.Errorf("decoder: %s.\n"+
		"  Loading it would page to swap rather than run, so it was NOT loaded.\n"+
		"  %s\n"+
		"%s"+
		"  Set GOINFER_NO_FIT_GUARD=1 to load anyway if this machine really fits it",
		f.arithmetic(), f.remedy(), alt)
}

// guardFit runs the check and returns the refusal, or nil to proceed. It prints the arithmetic to
// stderr when the load is within fitWarnRatio of refusing.
func guardFit(f fitCheck) error {
	if os.Getenv("GOINFER_NO_FIT_GUARD") != "" {
		return nil
	}
	if !f.known() {
		return nil
	}
	if !f.fits() {
		return f.declineErr()
	}
	if f.ratio() >= fitWarnRatio {
		fmt.Fprintln(os.Stderr, f.warning())
	}
	return nil
}

// quantBytesPerElem is the resident cost of one weight ELEMENT of a 2-D matmul matrix under each
// quant mode, scales included. int8 carries one f32 scale per ROW and int4 one per group of
// int4GroupSize, so the per-element cost is 1 + 4/cols and 0.5 + 4/32 respectively; the row term
// is bounded by the group term for any real matrix, so 1.125 covers it.
//
// int4mix is deliberately given int4's cost even though it keeps attention at int8: this is a
// LOWER BOUND on the real footprint, which is the safe direction — the guard may fail to refuse a
// marginal model, but it cannot refuse one that would have fit.
func quantBytesPerElem(q quantMode) float64 {
	switch q {
	case quantInt4, quantInt4Mix:
		return 0.5 + 4.0/float64(int4GroupSize)
	case quantInt8, quantInt8I8:
		return 1.125
	default: // quantNone — f32
		return 4
	}
}

// estimateGGUFWeightBytes sums every tensor's element count from the GGUF's metadata and prices
// it at the target quant. Metadata only: no tensor data is touched, so this costs a header parse
// on a model that is about to be read in full anyway.
//
// 1-D tensors (norms, biases) stay f32 whatever the quant, so they are priced as f32. They round
// to nothing beside the matrices, which is exactly the accounting ResidentWeightBytes documents.
func estimateGGUFWeightBytes(path string, q quantMode) int64 {
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return 0 // unknown ⇒ proceed
	}
	defer g.Close()
	var total float64
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
			total += 4 * float64(n)
			continue
		}
		total += quantBytesPerElem(q) * float64(n)
	}
	return int64(total)
}

// estimateKVBytes is the KV cache at a PINNED context. It returns 0 when no context was pinned,
// which is the common case and is honest: the CPU cache grows with the conversation rather than
// being allocated up front, so counting a context nobody asked for would refuse models that run
// fine for short turns. When -resident-context IS set, the allocation is real and up-front.
func estimateKVBytes(cfg *Config, ctx int, kvF16, kvI8 bool) int64 {
	if cfg == nil || ctx <= 0 || cfg.NumLayers <= 0 || cfg.NumKVHeads <= 0 {
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
	return int64(2 * perElem * float64(ctx) * float64(kvDim) * float64(cfg.NumLayers))
}

// fitCheckFor assembles the check for a load that has not happened yet. It knows how to price a
// .gguf; every other source returns a zero weight estimate, which means "unknown" and proceeds.
//
// A safetensors directory is deliberately NOT estimated from its file size: those are usually f32
// or bf16 on disk and shrink 6.4x loading at int4, so the file bytes would refuse models that fit
// comfortably. An estimate that is wrong in the refusing direction is worse than none.
func fitCheckFor(path, quantName string, quant quantMode, opts Options) fitCheck {
	if quantName == "" {
		quantName = "f32"
	}
	f := fitCheck{
		name:     filepath.Base(path),
		quant:    quantName,
		ramBytes: hostRAM(),
	}
	if !strings.HasSuffix(path, ".gguf") {
		return f
	}
	f.weightBytes = estimateGGUFWeightBytes(path, quant)
	// The KV term needs the config, which the same open already parsed on the way past; it is
	// only non-zero when a context was actually pinned, so skip the second open otherwise.
	if opts.ResidentContext > 0 {
		if g, err := embed.OpenGGUFMmap(path); err == nil {
			if cfg, cerr := ggufConfig(g); cerr == nil {
				f.kvBytes = estimateKVBytes(cfg, opts.ResidentContext, opts.KVPrecision == "f16", opts.KVQuant == "i8")
			}
			g.Close()
		}
	}
	return f
}

// weightAllocs counts entries into loadWeights — the call that turns a checkpoint into resident
// heap. It exists so "the guard refused BEFORE allocating" is an assertion rather than a
// deduction from the error text; a guard placed one line too late produces the same error and the
// same swap storm.
var weightAllocs atomic.Int64

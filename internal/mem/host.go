package mem

import (
	"fmt"
	"strings"
)

// HostLoad is what a resident load will put in host memory, as the caller priced it before allocating anything.
type HostLoad struct {
	Name  string // what the lines call the model
	Quant string // the requested weight mode, named because it moves the weight term the most

	WeightBytes int64 // resident weights at Quant; 0 = could not be priced, which proceeds
	KVBytes     int64 // KV at Ctx (0 when the KV is on a device)
	// SourceBytes is a plain .gguf source's size: the mapping stays resident for the whole build, beside the weights built from it.
	SourceBytes int64
	// CUDABuildBytes is the host peak of building a CUDA expert cache from a .gguf (weights, a packed copy, a pinned copy of the
	// experts); it replaces the weights+KV+source total when larger, because the phases do not overlap. ExpertBytes is its last term.
	CUDABuildBytes, ExpertBytes int64
	// PackBytes is what a CUDA-resident build adds beside the weights when there is no source term (one packed layer and staging).
	PackBytes int64
	DeviceKV  bool // the KV is allocated on a device, which a context plan sizes

	Ctx    int     // the context KV was priced at
	MaxCtx int     // the model's own window, 0 when it declares none
	Pinned bool    // the caller asked for Ctx explicitly: refuse rather than shrink
	KV     KVCurve // KV at a smaller context, for the auto-pin; nil when the config could not be read

	NoGuard         bool // the guard is turned off for this load
	GGUF            bool // the source is a .gguf file, not a safetensors directory: it names the remedy
	DenseStreamable bool // a -stream-weights retry would engage the dense layer pager
	Int4Smaller     bool // int4 costs less than int8 on this machine, so a refusal may suggest it
}

// HostCheck is a HostLoad against a machine's available host memory under a policy: the arithmetic every resident-guard
// line and decision reads.
type HostCheck struct {
	HostLoad
	Available int64 // available host memory; 0 = unknown
	Policy    Policy
}

// NewHostCheck is l against m under p.
func NewHostCheck(l HostLoad, m Machine, p Policy) HostCheck {
	return HostCheck{HostLoad: l, Available: m.Available(HostAnon), Policy: p}
}

// Need is the host memory the load takes at its peak.
func (c HostCheck) Need() int64 {
	n := max(c.CUDABuildBytes, c.WeightBytes+c.KVBytes+c.SourceBytes)
	if c.PackBytes > 0 {
		n = max(n, c.WeightBytes+c.PackBytes)
	}
	return n
}

// Budget is the share of available memory the load may take.
func (c HostCheck) Budget() int64 { return int64(float64(c.Available) * c.Policy.WeightsFraction) }

// Known reports whether both sides of the comparison are real numbers. Anything else proceeds.
func (c HostCheck) Known() bool { return c.Available > 0 && c.WeightBytes > 0 }

// Fits reports whether the load proceeds as asked: unknown, or within its budget.
func (c HostCheck) Fits() bool { return !c.Known() || c.Need() <= c.Budget() }

// Ratio is need over budget; 1.0 is exactly at the refusal.
func (c HostCheck) Ratio() float64 {
	if !c.Known() || c.Budget() <= 0 {
		return 0
	}
	return float64(c.Need()) / float64(c.Budget())
}

// Arithmetic is the one line the warning and the refusal share, so the two never quote different numbers for one load.
func (c HostCheck) Arithmetic() string {
	var b strings.Builder
	if c.CUDABuildBytes > c.WeightBytes+c.KVBytes+c.SourceBytes {
		fmt.Fprintf(&b, "%s needs ~%.1f GB of host memory while building the CUDA expert cache (%.1f GB weights + %.1f GB packed copy + %.1f GB pinned copy of the experts)",
			c.Name, float64(c.CUDABuildBytes)/gb, float64(c.WeightBytes)/gb, float64(c.WeightBytes)/gb, float64(c.ExpertBytes)/gb)
		fmt.Fprintf(&b, "; this machine currently has %.1f GB of memory available (budget %.1f GB = %.0f%% of that)",
			float64(c.Available)/gb, float64(c.Budget())/gb, c.Policy.WeightsFraction*100)
		return b.String()
	}
	fmt.Fprintf(&b, "%s needs ~%.1f GB resident at quant %s", c.Name, float64(c.WeightBytes)/gb, c.Quant)
	if c.DeviceKV {
		b.WriteString(" (its KV is in VRAM, which the context plan sizes)")
	}
	if c.KVBytes > 0 {
		fmt.Fprintf(&b, " + %.1f GB KV", float64(c.KVBytes)/gb)
	}
	if c.SourceBytes > 0 {
		// The source stays mapped resident for the whole load beside the weights built from it, named so this does not read as
		// a doubled weight estimate.
		fmt.Fprintf(&b, " + %.1f GB reading the checkpoint (the .gguf stays mapped resident for the whole load)", float64(c.SourceBytes)/gb)
	}
	if c.KVBytes > 0 || c.SourceBytes > 0 {
		fmt.Fprintf(&b, " = %.1f GB", float64(c.Need())/gb)
	}
	fmt.Fprintf(&b, "; this machine currently has %.1f GB of memory available (budget %.1f GB = %.0f%% of that)",
		float64(c.Available)/gb, float64(c.Budget())/gb, c.Policy.WeightsFraction*100)
	return b.String()
}

// Warning is what the banner prints when the load fits but only just.
func (c HostCheck) Warning() string {
	return fmt.Sprintf("decoder: fit is tight — %s (%.0f%% of budget). %s", c.Arithmetic(), c.Ratio()*100, c.Remedy())
}

// Remedy names the way out. -stream-weights helps only a .gguf source; a safetensors directory's way out is a one-time .giw
// build with cmd/prequant, which streams one layer at a time.
func (c HostCheck) Remedy() string {
	if c.GGUF {
		return "Re-run goinfer-serve with -stream-weights: it caches the model as a sidecar .giw once, " +
			"then pages weights out of it on demand instead of holding them all resident."
	}
	return "This is a safetensors checkpoint — -stream-weights only helps a .gguf source. Build a .giw " +
		"from it once with cmd/prequant (-quant <q> -o <name>.giw <dir>), which streams one layer at a time " +
		"and does not need this load's memory, then load the .giw (goinfer-serve and goinfer-chat do this " +
		"for you unless --direct-load). If you believe this machine can hold it as it is (this guard's 70% " +
		"margin is deliberately conservative), GOINFER_NO_FIT_GUARD=1 loads it anyway."
}

// Decline is a refusal: its message, and whether an automatic -stream-weights retry is sound.
type Decline struct {
	Msg             string
	DenseStreamable bool
}

// Decline is the refusal of this load.
func (c HostCheck) Decline() *Decline {
	// A smaller quant is suggested only where it is smaller: where the loader keeps a repacked copy of the nibbles, int4 costs
	// more than int8.
	alt := "  Or run a smaller model.\n"
	if c.Quant != "int4" && c.Quant != "int4mix" && c.Int4Smaller {
		alt = "  Or run a smaller model, or --quant int4, which is smaller on this machine.\n"
	}
	msg := fmt.Sprintf("decoder: %s.\n"+
		"  Loading it would page to swap rather than run, so it was NOT loaded.\n"+
		"  %s\n"+
		"%s"+
		"  Set GOINFER_NO_FIT_GUARD=1 to load anyway if this machine really fits it",
		c.Arithmetic(), c.Remedy(), alt)
	return &Decline{Msg: msg, DenseStreamable: c.DenseStreamable}
}

// SmallerFittingContext is the largest context at most Ctx, and at least the policy's floor, whose weights and KV fit the
// budget. It answers false for a pinned load (refused, never shrunk), for a load with no KV curve, and when even the floor
// does not fit. The curve is non-decreasing, so a binary search finds the largest fitting context exactly.
func (c HostCheck) SmallerFittingContext() (int, bool) {
	if c.Pinned || c.KV == nil {
		return 0, false
	}
	// The CUDA build peak has no KV in it, so a smaller context cannot bring it under the budget.
	if c.CUDABuildBytes > c.Budget() {
		return 0, false
	}
	// The source is a fixed term like the weights: it comes out of the budget too.
	room := c.Budget() - c.WeightBytes - c.SourceBytes
	if room <= 0 {
		return 0, false
	}
	fits := func(ctx int) bool { return c.KV(ctx) <= room }
	floor := c.Policy.CtxFloor
	if !fits(floor) {
		return 0, false
	}
	if fits(c.Ctx) {
		return c.Ctx, true // KV costs nothing more up to Ctx (an all-recurrent model): a defensive answer, not the common case
	}
	lo, hi := floor, c.Ctx
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

// Decide is the guard's decision: the context to pin (0 = leave the request alone), or the refusal.
func (c HostCheck) Decide() (pinCtx int, decline *Decline) {
	if c.NoGuard || !c.Known() || c.Fits() {
		return 0, nil
	}
	if ctx, ok := c.SmallerFittingContext(); ok {
		return ctx, nil
	}
	return 0, c.Decline()
}

// CapNote is the banner line for an auto-pinned context.
func (c HostCheck) CapNote(ctx int) string {
	var kv int64
	if c.KV != nil {
		kv = c.KV(ctx)
	}
	return fmt.Sprintf(
		"decoder: context capped at %d (model allows %d) so that KV fits: weights %.1f GB + KV %.1f GB of budget %.1f GB — pass -ctx to choose, or -stream-weights to lift",
		ctx, c.MaxCtx, float64(c.WeightBytes)/gb, float64(kv)/gb, float64(c.Budget())/gb)
}

// HostFit is SolveHost's answer: the decision, and the line the guard prints (a cap note or a warning; "" for none).
type HostFit struct {
	PinCtx  int
	Decline *Decline
	Line    string
}

// SolveHost is the resident guard on l against m under p.
func SolveHost(l HostLoad, m Machine, p Policy) HostFit {
	c := NewHostCheck(l, m, p)
	pin, decline := c.Decide()
	f := HostFit{PinCtx: pin, Decline: decline}
	switch {
	case decline != nil:
	case pin > 0:
		f.Line = c.CapNote(pin)
	case c.Known() && !c.NoGuard && c.Ratio() >= p.WarnRatio:
		f.Line = c.Warning()
	}
	return f
}

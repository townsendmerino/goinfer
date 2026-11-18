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
	// SnapshotBytes is the prefix-snapshot store's budget (-kv-host-bytes), 0 where the load keeps no store. The store fills after the build, beside what
	// stays on the host once the server is up: the weights and KV of a load that is not device-resident, the pinned expert copy of one that is.
	SnapshotBytes int64

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

// heapSlackDen prices the process's own heap beside the weights it builds, while a load maps its source: 1/heapSlackDen of the weights.
// A margin, on Francis's word; the reading behind it is docs/measurements/memfix-e6-2026-11/README.md (the anonymous peak 2.2-2.8% over
// the weights on two direct .gguf loads).
const heapSlackDen = 100

// HeapSlackBytes is the heap slack the guard prices beside weights bytes of weights for a load mapping source bytes of its source: 0 with
// no source term.
func HeapSlackBytes(weights, source int64) int64 {
	if source <= 0 {
		return 0
	}
	return weights / heapSlackDen
}

// loadTerm is the load's own peak: the weights, the KV, the mapped source and the heap slack beside them (HeapSlackBytes).
func (c HostCheck) loadTerm() int64 {
	return c.WeightBytes + c.KVBytes + c.SourceBytes + HeapSlackBytes(c.WeightBytes, c.SourceBytes)
}

// Need is the host memory the load takes at its peak: the build's, or the serving state's with the snapshot store full, whichever is larger.
func (c HostCheck) Need() int64 { return max(c.buildPeak(), c.servingNeed()) }

// buildPeak is the host memory the build takes at its peak.
func (c HostCheck) buildPeak() int64 {
	n := max(c.CUDABuildBytes, c.loadTerm())
	if c.PackBytes > 0 {
		n = max(n, c.WeightBytes+c.PackBytes)
	}
	return n
}

// servingNeed is the host memory a server holds once it is up with the snapshot store at its budget: that budget beside what stays on the host
// (the weights and KV of a load that is not device-resident, the pinned expert copy of one that is). 0 without a store: the build's peak is then
// the whole need.
func (c HostCheck) servingNeed() int64 {
	if c.SnapshotBytes <= 0 {
		return 0
	}
	if c.DeviceKV {
		return c.ExpertBytes + c.SnapshotBytes
	}
	return c.WeightBytes + c.KVBytes + c.SnapshotBytes
}

// snapshotBinds reports whether the snapshot store, not the build, sets Need: the case whose refusal names -kv-host-bytes.
func (c HostCheck) snapshotBinds() bool { return c.servingNeed() > c.buildPeak() }

// needUnpinned is Need with the CUDA expert cache's pinned copy out of the build peak, as --moe-experts-unpinned prices it: every other
// term Need takes stays.
func (c HostCheck) needUnpinned() int64 {
	n := max(c.CUDABuildBytes-c.ExpertBytes, c.loadTerm())
	if c.PackBytes > 0 {
		n = max(n, c.WeightBytes+c.PackBytes)
	}
	switch {
	case c.SnapshotBytes > 0 && c.DeviceKV:
		n = max(n, c.SnapshotBytes) // the pinned experts are gone from the serving state too
	default:
		n = max(n, c.servingNeed())
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
	if c.snapshotBinds() {
		fmt.Fprintf(&b, "%s needs ~%.1f GB of host memory once it is serving with its prefix-snapshot store full (%s + %.1f GB of snapshots, -kv-host-bytes)",
			c.Name, float64(c.Need())/gb, c.stays(), float64(c.SnapshotBytes)/gb)
		fmt.Fprintf(&b, "; this machine currently has %.1f GB of memory available (budget %.1f GB = %.0f%% of that)",
			float64(c.Available)/gb, float64(c.Budget())/gb, c.Policy.WeightsFraction*100)
		return b.String()
	}
	if c.CUDABuildBytes > c.loadTerm() {
		fmt.Fprintf(&b, "%s needs ~%.1f GB of host memory while building the CUDA expert cache (%.1f GB weights + %.1f GB pinned copy of the experts and packing)",
			c.Name, float64(c.CUDABuildBytes)/gb, float64(c.WeightBytes)/gb, float64(c.CUDABuildBytes-c.WeightBytes)/gb)
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
		fmt.Fprintf(&b, " + %.1f GB reading the checkpoint (the .gguf stays mapped resident for the whole load) + %.1f GB heap beside it",
			float64(c.SourceBytes)/gb, float64(HeapSlackBytes(c.WeightBytes, c.SourceBytes))/gb)
	}
	if c.KVBytes > 0 || c.SourceBytes > 0 {
		fmt.Fprintf(&b, " = %.1f GB", float64(c.Need())/gb)
	}
	fmt.Fprintf(&b, "; this machine currently has %.1f GB of memory available (budget %.1f GB = %.0f%% of that)",
		float64(c.Available)/gb, float64(c.Budget())/gb, c.Policy.WeightsFraction*100)
	return b.String()
}

// stays names what the serving state holds beside the snapshot store.
func (c HostCheck) stays() string {
	switch {
	case c.DeviceKV && c.ExpertBytes > 0:
		return fmt.Sprintf("%.1f GB pinned expert copy", float64(c.ExpertBytes)/gb)
	case c.DeviceKV:
		return "nothing else, its weights and KV are on the device"
	case c.KVBytes > 0:
		return fmt.Sprintf("%.1f GB weights + %.1f GB KV", float64(c.WeightBytes)/gb, float64(c.KVBytes)/gb)
	}
	return fmt.Sprintf("%.1f GB weights", float64(c.WeightBytes)/gb)
}

// Warning is what the banner prints when the load fits but only just.
func (c HostCheck) Warning() string {
	w := fmt.Sprintf("decoder: fit is tight — %s (%.0f%% of budget). %s", c.Arithmetic(), c.Ratio()*100, c.Remedy())
	if c.GGUF && !QuantPages(c.Quant) { // that remedy says only what will not help
		w += " " + c.smallerLoad() + ", or pass a smaller -ctx."
	}
	return w
}

// QuantPages reports whether weight paging can read a bundle's weights at this quant: only int8 and int4 storage has a span
// to page (aikit's WeightMat.MappedSpan), so a retry or a remedy with -stream-weights is sound only there.
func QuantPages(quant string) bool {
	switch quant {
	case "int8", "int8int8", "int4", "int4mix":
		return true
	}
	return false
}

// Remedy names the way out. -stream-weights helps only a .gguf source; a safetensors directory's way out is a one-time .giw
// build with cmd/prequant, which streams one layer at a time.
func (c HostCheck) Remedy() string {
	if c.snapshotBinds() {
		return "Lower -kv-host-bytes (the prefix-snapshot store's budget; 0 turns snapshots off)."
	}
	if c.GGUF && !QuantPages(c.Quant) {
		return fmt.Sprintf("-stream-weights will not help at --quant %s: weight paging reads only int8 and int4 weights.", c.Quant)
	}
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

// smallerLoad names the other ways down. A smaller quant is suggested only where it is smaller: where the loader keeps a
// repacked copy of the nibbles, int4 costs more than int8.
func (c HostCheck) smallerLoad() string {
	if c.Quant != "int4" && c.Quant != "int4mix" && c.Int4Smaller {
		return "Or run a smaller model, or --quant int4, which is smaller on this machine"
	}
	return "Or run a smaller model"
}

// Decline is a refusal: its message, and whether an automatic -stream-weights retry is sound.
type Decline struct {
	Msg             string
	DenseStreamable bool
}

// Decline is the refusal of this load.
func (c HostCheck) Decline() *Decline {
	alt := "  " + c.smallerLoad() + ".\n"
	if c.CUDABuildBytes > c.loadTerm() && c.ExpertBytes > 0 && c.needUnpinned() <= c.Budget() {
		alt = "  Or " + UnpinnedExpertsHint(c.ExpertBytes) + ".\n" + alt
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
	// The snapshot store and the pinned expert copy do not depend on the context: if they alone exceed the budget no smaller context helps.
	if c.DeviceKV && c.servingNeed() > c.Budget() {
		return 0, false
	}
	// The source is a fixed term like the weights: it comes out of the budget too, and so does the snapshot store beside a host-resident
	// load's weights and KV (the two do not coexist, so the larger counts).
	fixed := c.SourceBytes + HeapSlackBytes(c.WeightBytes, c.SourceBytes)
	if !c.DeviceKV {
		fixed = max(fixed, c.SnapshotBytes)
	}
	room := c.Budget() - c.WeightBytes - fixed
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
	lift := ", or -stream-weights to lift"
	if !QuantPages(c.Quant) {
		lift = ""
	}
	return fmt.Sprintf(
		"decoder: context capped at %d (model allows %d) so that KV fits: weights %.1f GB + KV %.1f GB of budget %.1f GB — pass -ctx to choose%s",
		ctx, c.MaxCtx, float64(c.WeightBytes)/gb, float64(kv)/gb, float64(c.Budget())/gb, lift)
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

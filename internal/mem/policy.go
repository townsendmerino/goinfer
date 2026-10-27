package mem

// Tier is where a byte lives.
type Tier uint8

const (
	HostAnon   Tier = iota // host RAM a process commits: what the available reading prices
	HostPinned             // page-locked host RAM
	FileMapped             // host RAM backed by a file the kernel can drop
	VRAM                   // a discrete device's memory
	Unified                // memory a CPU and a GPU share (Apple)
)

// Machine is one box's readings, as data: what internal/mem/probe read, or a recorded profile. A missing or zero figure is
// unknown.
type Machine struct {
	Name string
	Free map[Tier]int64 // bytes available per tier
}

// HostMachine is a Machine with only its host reading, available bytes of host RAM.
func HostMachine(available int64) Machine { return Machine{Free: map[Tier]int64{HostAnon: available}} }

// Available is m's reading for tier t, 0 when unknown.
func (m Machine) Available(t Tier) int64 { return m.Free[t] }

// The guard thresholds. A change to any of them needs Francis's word (CLAUDE.md, "guard threshold").
const (
	// WeightsFraction is the share of available host memory a resident load (weights, KV and the mapped source) may take.
	WeightsFraction = 0.70
	// WarnRatio is how close to its budget a load comes before the banner prints the arithmetic unasked.
	WarnRatio = 0.75
	// CtxFloor is the smallest context the resident and .giw guards auto-pin an unpinned load down to.
	CtxFloor = 2048
	// GIWMargin is the flat margin the .giw guard leaves under available memory.
	GIWMargin = 1 << 30
)

// Policy is the thresholds a decision applies. DefaultPolicy is the committed one.
type Policy struct {
	WeightsFraction float64 // share of available memory for a resident load
	WarnRatio       float64 // need/budget at and above which a fitting load warns
	CtxFloor        int     // the auto-pin floor
	GIWMargin       int64   // the .giw guard's margin under available memory
	RequestFraction float64 // share of available memory one request's KV and scratch may take
}

// DefaultPolicy returns the thresholds in force.
func DefaultPolicy() Policy {
	return Policy{WeightsFraction: WeightsFraction, WarnRatio: WarnRatio, CtxFloor: CtxFloor, GIWMargin: GIWMargin, RequestFraction: WeightsFraction}
}

// gb is the guard's GB: 2^30 bytes.
const gb = 1 << 30

// KVCurve is the KV a load allocates at a context, in bytes. It is non-decreasing in the context; it need not be linear (a
// sliding-window layer's cost flattens past its window).
type KVCurve func(ctx int) int64

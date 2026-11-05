// Package fit holds the types the load-time fit guard and the placement planner exchange with their callers: Placement, PlanRequest,
// Plan, FitAnswer and FitDeclineError. The pricing and the decisions live in internal/runtime/mem (which imports nothing from goinfer)
// and in package decoder, which reads checkpoints and resolves architectures; this package imports only mem, so a caller outside
// decoder can name these types without importing the loader. Package decoder re-exports each one as an alias, so
// decoder.Plan and fit.Plan are the same type.
package fit

import (
	"errors"
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// Each Placement equals internal/mem's of the same name, so Plan converts one to the other; a compile error here means they
// drifted.
var (
	_ [int(PlacementResident) - int(mem.Resident)]struct{}
	_ [int(PlacementExpertCached) - int(mem.ExpertCached)]struct{}
	_ [int(PlacementHostComputedExperts) - int(mem.HostComputedExperts)]struct{}
	_ [int(PlacementWeightPaged) - int(mem.WeightPaged)]struct{}
	_ [int(PlacementDecline) - int(mem.Declined)]struct{}
	_ [int(mem.Resident) - int(PlacementResident)]struct{}
	_ [int(mem.ExpertCached) - int(PlacementExpertCached)]struct{}
	_ [int(mem.HostComputedExperts) - int(PlacementHostComputedExperts)]struct{}
	_ [int(mem.WeightPaged) - int(PlacementWeightPaged)]struct{}
	_ [int(mem.Declined) - int(PlacementDecline)]struct{}
)

// Placement is which of docs/tasks/task-fit-to-hardware.md's five strategies Plan chose for one backend. PlacementHostComputedExperts is
// reserved and never chosen, so adding it later does not rewrite the type.
type Placement int

const (
	PlacementResident Placement = iota
	PlacementExpertCached
	PlacementHostComputedExperts // reserved: misses computed on CPU from a pinned host copy; never chosen
	PlacementWeightPaged
	PlacementDecline
)

func (p Placement) String() string {
	switch p {
	case PlacementResident:
		return "resident"
	case PlacementExpertCached:
		return "expert-cached"
	case PlacementHostComputedExperts:
		return "host-computed-experts"
	case PlacementWeightPaged:
		return "weight-paged"
	case PlacementDecline:
		return "decline"
	default:
		return fmt.Sprintf("Placement(%d)", int(p))
	}
}

// PlanRequest is what the user asked for, with every default already resolved by the CALLER (docs/tasks/task-fit-to-hardware.md §2).
// Plan itself invents no defaults, so a caller (goinfer-chat fit, the startup banner, a table test) can see and vary exactly what it asked
// for rather than a default buried inside the function under test.
type PlanRequest struct {
	Ctx       int  // requested context; Plan may shrink it toward ctxPlanFloor unless CtxPinned
	CtxPinned bool // true for an explicit -ctx: refuse (PlacementDecline) rather than silently shrink
	Slots     int  // an explicit --moe-cache-slots / GOINFER_METAL_MOE_SLOTS; 0 means "let Plan choose"

	// KVF16/KVI8 select a lossy KV precision; both false means f32 (bit-exact, the default). They must come from an explicit flag the caller
	// read, never be set by Plan on its own initiative: the plan never selects a lossy precision to make something fit without saying so
	// (docs/tasks/task-fit-to-hardware.md §0).
	KVF16, KVI8 bool

	// KVSlots is how many KV slots the load holds, and it reaches only Plan.CheckpointCopies: Placement and NeedBytes stay one slot's, because a caller that sizes a context for
	// several prices the further slots through ExtraBytes (a multi-slot fit decides that itself). CheckpointReserve is what a build leaves free beside its checkpoints for what it
	// allocates after them: the expert cache and the companion included, so with a reserve named those two leave the need the copies are counted beside.
	KVSlots           int
	CheckpointReserve int64

	// ExtraBytes prices whatever the model itself does not know about but will share its device: a block drafter's weights and verify/capture
	// buffers, a vision tower (every allocation is a term of the plan, including the ones that attach after load). Without it an expert cache
	// is sized before a later --drafter attach claims the room NewBlockSpec then needs. The backend learns the same value out of band through
	// Options.ExtraResidentBytes / Model.ExtraResidentBytes(), and resolveCtxCapFit passes it here when it asks Plan for an unpinned ctx, so
	// Plan's dry run and the real load price a drafter identically.
	ExtraBytes int64
}

// Plan is one backend's placement decision, with every number that justified it — so a decline
// message, a dry-run printout, and a unit test assertion all read the same fields rather than
// three independent re-derivations of the same arithmetic.
type Plan struct {
	Backend   string
	Placement Placement
	Ctx       int // the context actually used — may be smaller than PlanRequest.Ctx
	Slots     int // expert-cache slot count actually used; 0 when N/A (dense) or "every expert" (unpaged)

	DenseBytes      int64 // decoder.Model.ResidentDenseWeightBytesFor(backend) — the fixed, never-shrinks term
	ExpertBytesFull int64 // every routed expert, unpaged — reported even when Slots caps what's actually resident
	ExpertBytesUsed int64 // what Slots actually keeps resident (equals ExpertBytesFull when Slots==0/unpaged/dense)
	KVBytes         int64 // at Ctx, as this backend allocates it (ResidentKVBytes)
	HostCopyBytes   int64 // Metal only: the host copy of the resident weights unified memory also holds (ResidentHostCopyBytes; 0 when aliased from a .giw)
	StateBytes      int64 // one KV slot's recurrent and conv state as this backend allocates it (BuildDemand.StateBytes); 0 for a model with none
	// CheckpointBytes is one recurrent checkpoint this plan's load keeps (BuildDemand.CheckpointBytes), 0 on a declined plan or where it does
	// not fit beside NeedBytes. It is not in NeedBytes and moves no decision here: a load with no room for it keeps none and still runs.
	CheckpointBytes int64
	// CheckpointCopies is how many checkpoints the load keeps: 1, or on a backend that keeps one per KV slot (BuildDemand.CheckpointPerSlot) one for each of PlanRequest.KVSlots
	// that fits beside every slot's KV and state and PlanRequest.CheckpointReserve; 0 where CheckpointBytes is.
	CheckpointCopies int
	AllocSlackBytes  int64 // the backend's allocator rounding over the weight buffers (PlanSlack): CUDA's 2 MiB granularity
	ExtraBytes       int64 // PlanRequest.ExtraBytes, carried through for NeedBytes()/reporting

	FreeBytes int64  // what Plan was told is available on this backend/device
	Reason    string // human-readable justification (admit) or decline message with the numbers
}

// NeedBytes is the total this Plan actually asks the device to hold — DenseBytes + whatever the
// chosen Slots keeps of the experts + KV at the chosen Ctx + (Metal) the host copy + the caller's own
// extra terms and the recurrent state. Never ExpertBytesFull: that field exists for reporting what was capped, not what is
// resident. Without ExtraBytes it equals Model.ResidentNeedBytes at the same Slots and Ctx — the number Metal's resident guard
// judges — which TestPlan_metalAgreesWithResidentNeedBytes and TestResidentNeedBytes_countsRecurrentState pin.
func (p Plan) NeedBytes() int64 {
	return p.DenseBytes + p.StateBytes + p.AllocSlackBytes + p.ExpertBytesUsed + p.KVBytes + p.HostCopyBytes + p.ExtraBytes
}

// ErrWontFitResident is wrapped into every load-time fit-guard refusal (declineErr), so a caller
// can detect the refusal with errors.Is/errors.As instead of parsing the message text — e.g. to
// decide whether an automatic -stream-weights retry is worth attempting (see FitDeclineError).
var ErrWontFitResident = errors.New("decoder: model will not fit resident RAM")

// FitDeclineError is what the load-time fit guard returns instead of a bare error. Build one with NewDeclineError.
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

// NewDeclineError is the refusal carrying msg, with DenseStreamable as given.
func NewDeclineError(msg string, denseStreamable bool) *FitDeclineError {
	return &FitDeclineError{msg: msg, DenseStreamable: denseStreamable}
}

// FitAnswer is what Load's own memory check says of loading a path with given options, priced against the memory
// available now, without loading anything. FitQuery computes it with the functions Load calls (fitCheckFor and guardFit's
// decision for a .gguf or safetensors directory, decideGIWFit for a .giw), so the two cannot disagree.
type FitAnswer struct {
	// Guard names the check that priced it: "resident" (weights, KV and the mapped source, against 70% of available memory)
	// or "giw" (KV and scratch against available memory less 1 GB: a .giw's weights are file-backed).
	Guard string
	// Fits is whether Load proceeds. Load proceeds when the check cannot price the load; Unpriced then says why.
	Fits     bool
	Unpriced string
	// Ctx is the context priced; when CtxCapped, the smaller one Load would pin an unpinned request to.
	Ctx                                                      int
	CtxPinned                                                bool
	CtxCapped                                                bool
	MaxCtx                                                   int // the model's own window, 0 when it declares none
	NeedBytes, BudgetBytes, WeightBytes, KVBytes, AvailBytes int64
	// PinnedExpertBytes is the page-locked host copy of the routed experts that a CUDA expert cache (--moe-cache-experts) builds from a .giw, priced into
	// NeedBytes; 0 for every other load.
	PinnedExpertBytes int64
	Decline           string // Load's refusal when it does not fit, "" otherwise
	// Streamable is the refusal's DenseStreamable: modelload.Load retries such a .gguf with weight streaming.
	Streamable bool
	// DecodePath is the decode path Load would report: "cpu (<quant>)", as DecodePath begins, when the load runs on the CPU,
	// or, for a .giw on a backend whose pre-build checks can be asked (ResidencyPricer), the resident or the decline they
	// decide. "" when only the load can tell; DecodeUnknown then says why.
	DecodePath    string
	DecodeUnknown string
	// ResidentCtx is the context a resident decode path would hold, when DecodePath names one (0 otherwise).
	ResidentCtx int
}

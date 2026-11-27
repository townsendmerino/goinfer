package residency

import (
	"io"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
	"github.com/townsendmerino/goinfer/internal/runtime/kvsnap"
)

// Host holds a lightweight view of a Model's resident-KV coordination state
// (slots, namespaces, host snapshots, recurrent checkpoint prompt, and drafter
// sync marker) plus the immutable per-model configuration flags the reuse,
// snapshot, and checkpoint paths read.
type Host struct {
	Resident          resident.Forward
	KV                **kvmgr.Slots
	KVNS              *[]string
	KVSnaps           **kvsnap.Store
	CkptPrompt        *[]int
	DrafterSynced     *any
	Arch              *darch.Architecture
	BackendName       string
	KVPrecision       string
	ReuseDisabled     bool
	RecurrentSlotsOK  bool
	ShareOff          bool
	KVDemote          bool
	ScratchSlotPlant  bool
	AllowanceOverride int
	InPlaceCapOff     bool
	KVHostBytesOpt    int64
	ReuseFloor        int
	PrefillTailMin    int
	Stderr            io.Writer
	SnapMinPrompt     int
}

func (h *Host) subsystems() resident.Subsystems { return ResolveSubsystems(h.Resident) }
func (h *Host) resKV() resident.KVSubsystem     { return h.subsystems().KV }
func (h *Host) resCaps() resident.Caps          { return h.subsystems().Runtime.ResidentCaps() }
func (h *Host) hasRecurrentState() bool         { return h.Arch != nil && h.Arch.Recurrent() }

func (h *Host) residentFloors() (fast int, hasFast bool, kernel int, hasKernel bool) {
	c := h.resCaps()
	return c.FastPrefillFloor, c.HasFastPrefill, c.KernelFloor, c.HasKernelFloor
}

func (h *Host) stderr() io.Writer {
	if h.Stderr != nil {
		return h.Stderr
	}
	return io.Discard
}

func (h *Host) snapMinPrompt() int {
	if h.SnapMinPrompt > 0 {
		return h.SnapMinPrompt
	}
	return 2048
}

func (h *Host) stepMarkAlign() (int, bool) {
	pf := h.subsystems().Prefill
	if !pf.PrefillMarkedAvailable() {
		return 0, false
	}
	return pf.PrefillMarkAlign(), true
}

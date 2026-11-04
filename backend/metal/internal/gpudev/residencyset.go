//go:build darwin

package gpudev

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// ResidencySet wraps MTLResidencySet (macOS 15+), which pins a set of allocations resident so the
// driver does NOT re-validate
// their residency on every command-buffer commit — the fix for the per-submit make-resident tax that
// dominates a per-layer-submit decode (measured ~15 ms/boundary of GPU-idle-in-wait, vs ~0.4 ms once
// a set is cached). Add the stable working set (always-on weights + slot buffers) ONCE, commit,
// requestResidency, and attach to the queue; every command buffer on that queue then finds the set
// already resident. NEW entry point — does not alter any existing API. Availability is the caller's
// responsibility: guard with ResidencySetsSupported (nil set / fallback path on older OSes).
type ResidencySet struct{ id objc.ID }

var (
	selNewResidencySet  = objc.RegisterName("newResidencySetWithDescriptor:error:")
	selAddAllocation    = objc.RegisterName("addAllocation:")
	selRequestResidency = objc.RegisterName("requestResidency")
	selEndResidency     = objc.RegisterName("endResidency")
	selAddResidencySet  = objc.RegisterName("addResidencySet:")
	selUseResidencySet  = objc.RegisterName("useResidencySet:")
	// selCommit ("commit") and selAlloc/selInit/selRelease are declared in metal.go — reused here.
)

// ResidencySetsSupported reports whether this OS/driver exposes MTLResidencySet (macOS 15+). It
// probes the descriptor class rather than the OS version so it tracks the actual capability.
func ResidencySetsSupported() bool {
	return objc.GetClass("MTLResidencySetDescriptor") != 0
}

// NewResidencySet creates an empty residency set. Caller adds allocations, Commit()s, then
// RequestResidency(). Returns an error if residency sets are unavailable or creation fails.
func (d *Device) NewResidencySet() (ResidencySet, error) {
	cls := objc.GetClass("MTLResidencySetDescriptor")
	if cls == 0 {
		return ResidencySet{}, fmt.Errorf("gpu: MTLResidencySet unavailable (needs macOS 15+)")
	}
	desc := objc.ID(cls).Send(selAlloc).Send(selInit)
	var nsErr objc.ID
	rs := d.id.Send(selNewResidencySet, desc, unsafe.Pointer(&nsErr))
	desc.Send(selRelease)
	if rs == 0 {
		return ResidencySet{}, fmt.Errorf("gpu: newResidencySetWithDescriptor failed: %s", goString(nsErr.Send(selLocalizedDesc)))
	}
	d.TrackObj(rs) // +1-owned; released at Device close
	return ResidencySet{id: rs}, nil
}

// Add stages a buffer for residency (pending until Commit). A Buffer's MTLBuffer conforms to
// MTLAllocation. Add whole buffers only (offset sub-views share the parent allocation). A zero/nil
// buffer (an unset struct field) is skipped rather than passed to addAllocation:.
func (rs ResidencySet) Add(b Buffer) {
	if b.id == 0 {
		return
	}
	rs.id.Send(selAddAllocation, b.id)
}

// AddAllDeviceBuffers stages EVERY MTLBuffer the device has handed out (the same set it tracks for
// ReleaseAll). DIAGNOSTIC ONLY — do NOT use it to "pin the whole model": a residency set's per-commit
// overhead scales with its SIZE, so pinning everything REGRESSES the heaviest-referenced phase (goinfer
// gemma4-26b paged decode, measured: phase-1 GPU-idle-in-wait 18 → 63 ms/CB, total −22%). Pinning helps
// ONLY buffers whose residency is invalidated between submits (e.g. CPU-written staging slots); stable
// weights are already resident, so pinning them only adds set-size overhead. Retained so the bisect
// that established this stays reproducible. Call once after all buffers exist, then Commit +
// RequestResidency.
func (rs ResidencySet) AddAllDeviceBuffers(d *Device) {
	for _, a := range d.allocs {
		rs.id.Send(selAddAllocation, a.id)
	}
}

// AddAllDeviceBuffersExcept stages every device buffer EXCEPT those in exclude. DIAGNOSTIC ONLY (same
// caveat as AddAllDeviceBuffers): it was the low-enumeration way to test "pin all read-only buffers",
// and the result REFUTED that idea — excluding the GPU-written set still regressed phase-1 (45 ms/CB),
// because the cost is pin-set SIZE, not read/write. Kept so that bisect is reproducible from goinfer.
func (rs ResidencySet) AddAllDeviceBuffersExcept(d *Device, exclude []Buffer) {
	skip := make(map[objc.ID]bool, len(exclude))
	for _, b := range exclude {
		if b.id != 0 {
			skip[b.id] = true
		}
	}
	for _, a := range d.allocs {
		if !skip[a.id] {
			rs.id.Send(selAddAllocation, a.id)
		}
	}
}

// Commit applies the pending Add()s to the set (no residency requested yet).
func (rs ResidencySet) Commit() { rs.id.Send(selCommit) }

// RequestResidency makes the committed allocations resident. Call after Commit; the driver keeps
// them resident (subject to memory pressure) until EndResidency or the set is released.
func (rs ResidencySet) RequestResidency() { rs.id.Send(selRequestResidency) }

// EndResidency releases the residency request (allocations become evictable again).
func (rs ResidencySet) EndResidency() { rs.id.Send(selEndResidency) }

// AddResidencySet attaches the set to this command queue: every command buffer committed on the
// queue references it, so the driver skips per-commit residency validation for its allocations.
//
// This is QUEUE-scoped, not per-command-buffer: EVERY command buffer on the queue carries the
// whole set in its referenced list, even one that never touches any allocation in it. Measured
// cost (goinfer audit-metal-2026-09-12.md M-14): a phase that never touches a ~3 GB paged-MoE slot
// pool still pays ~2.07 ms/CB of GPU-idle-in-wait validating a set it doesn't reference, ~62
// ms/token summed across a decode step's command buffers. UseResidencySet (below) is the
// per-encoder alternative — attach it only to the command buffers that actually touch the set.
func (q Queue) AddResidencySet(rs ResidencySet) { q.id.Send(selAddResidencySet, rs.id) }

// UseResidencySet attaches the set to THIS Encoder's command buffer only (macOS 15+'s
// MTLCommandEncoder.useResidencySet:, a per-encoder alternative to Queue.AddResidencySet's
// queue-wide attach — M-14, audit-metal-2026-09-12.md). Call once per Encoder, before End(); has
// no effect on any other command buffer the queue commits. Prefer this over AddResidencySet
// whenever only SOME of a queue's command buffers actually touch the set's allocations — a phase
// that never references it stops paying to carry an irrelevant set in its own referenced list.
func (e *Encoder) UseResidencySet(rs ResidencySet) { e.enc.Send(selUseResidencySet, rs.id) }

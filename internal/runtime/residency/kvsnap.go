package residency

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
	"github.com/townsendmerino/goinfer/internal/runtime/kvsnap"
)

// KeepsRecurrentSnapshots reports whether this model's prefix snapshots carry its recurrent state.
func (h *Host) KeepsRecurrentSnapshots() bool {
	return h.hasRecurrentState() && h.RecurrentSnapshots()
}

// RecurrentSnapshots reports whether the snapshotter carries recurrent state.
func (h *Host) RecurrentSnapshots() bool {
	sn := h.Snapshotter()
	return sn != nil && sn.KVSnapshotRecurrent()
}

// SnapshotsBlocked reports whether this model's state is outside what a snapshot holds: recurrent state its snapshotter does not carry.
func (h *Host) SnapshotsBlocked() bool { return h.hasRecurrentState() && !h.RecurrentSnapshots() }

// KVSnapshotStats reports the model's host snapshots (false when off).
func (h *Host) KVSnapshotStats() (resident.KVSnapshotStats, bool) {
	if (*h.KVSnaps) == nil {
		return resident.KVSnapshotStats{}, false
	}
	return (*h.KVSnaps).Stats(), true
}

// KVSnapshotsSupported reports whether this model keeps prefix snapshots: a budget, and a resident that can hold them.
func (h *Host) KVSnapshotsSupported() bool { return h.Snapshotter() != nil }

// KVHostBytes returns the host memory budget for prefix snapshots this model was loaded with (Options.KVHostBytes), 0 when off.
func (h *Host) KVHostBytes() int64 { return (*h.KVSnaps).Budget() }

// Snapshotter is the resident's snapshot capability, nil when snapshots are off for this model or the resident has none.
func (h *Host) Snapshotter() resident.KVSubsystem {
	if (*h.KVSnaps) == nil || h.Resident == nil {
		return nil
	}
	kv := h.resKV()
	if !kv.KVSnapshotterAvailable() {
		return nil
	}
	return kv
}

func snapSlackOf(sn resident.KVSubsystem) int {
	if sn != nil {
		if r := sn.KVSnapshotRollback(); r > 0 {
			return r
		}
	}
	return kvmgr.RollbackFloor
}

// RestoreSnapshot restores a held snapshot's rows into the bound slot when that serves more of prompt than the slot's own rows do.
func (h *Host) RestoreSnapshot(prompt []int, adapterID uint64, sc KVScope, acq *KVAcquired) {
	sn := h.Snapshotter()
	recurrent := h.hasRecurrentState()
	if sn == nil || sc.Off || h.ShareOff || adapterID != 0 || h.ReuseDisabled || h.SnapshotsBlocked() || acq.Slot < 0 {
		return
	}
	st := *h.KVSnaps
	root := h.KVRootIn(0, sc.Namespace)
	x, q := st.SelectRestore(root, prompt, acq.Reuse, h.PrefillTailMin, h.KVClassMap(), recurrent)
	if x == nil {
		return
	}
	t0 := time.Now()
	err := sn.RestoreKV(x.Handle, acq.Slot, q)
	d := time.Since(t0)
	if err != nil {
		if dec, ok := errors.AsType[*KVCopyDecline](err); ok {
			st.DeclineDetail("backend", dec.Reason)
			acq.Declined = dec.Reason
			return
		}
		st.DeclineDetail("error", err.Error())
		acq.Declined = err.Error()
		acq.Reuse = 0
		h.ForgetIDs()
		return
	}
	h.KVSlots().Restored(acq.Slot, root, x.IDs[:q], x.Classes)
	acq.Reuse, acq.SnapRows, acq.SnapDur, acq.snapFrom, acq.Shared, acq.Src = q, q, d, x, 0, -1
	st.Use(x, q, d)
	fmt.Fprintf(h.stderr(), "kv snapshot: restored %d rows of %d into slot %d in %.1f ms\n", q, x.Rows(), acq.Slot, float64(d.Microseconds())/1000)
}

// SnapshotPrompt takes an end-of-prompt snapshot of prompt's full blocks from slot before decode begins.
func (h *Host) SnapshotPrompt(slot int, prompt []int, adapterID uint64, sc KVScope, acq KVAcquired) {
	if h.hasRecurrentState() {
		return
	}
	if q, ok := h.KVBoundaryPlan(prompt, sc, adapterID); ok {
		if acq.MarkFired || acq.SnapRows > 0 {
			return
		}
		if st := *h.KVSnaps; st != nil && st.Covering(h.KVRootIn(0, sc.Namespace), prompt, q, snapSlackOf(h.Snapshotter())) != nil {
			st.Decline("held")
			return
		}
	}
	h.SnapshotRows(slot, prompt, len(prompt)/kvmgr.BlockRows*kvmgr.BlockRows, adapterID, sc, acq)
}

// KVBoundaryPlan computes the prefill position at which this request should take its boundary or Keep snapshot instead of at the end of the prompt.
func (h *Host) KVBoundaryPlan(prompt []int, sc KVScope, adapterID uint64) (q int, ok bool) {
	align, ok := h.KVMarkGuard(sc, adapterID)
	if !ok {
		return 0, false
	}
	return kvsnap.BoundaryPlan(kvsnap.BoundaryConfig{
		PromptLen: len(prompt),
		Boundary:  sc.Boundary,
		Keep:      sc.Keep,
		Align:     align,
		MinPrompt: h.snapMinPrompt(),
		Slack:     snapSlackOf(h.Snapshotter()),
		MinTail:   h.PrefillTailMin,
		Recurrent: h.hasRecurrentState(),
	})
}

// KVMarkGuard returns the width of the resident's marked passes and whether a snapshot mark may be planned for this request.
func (h *Host) KVMarkGuard(sc KVScope, adapterID uint64) (align int, ok bool) {
	align, hasMark := h.stepMarkAlign()
	if h.Snapshotter() == nil || !hasMark || adapterID != 0 || sc.Off || h.ShareOff || h.ReuseDisabled || h.SnapshotsBlocked() {
		return 0, false
	}
	return align, true
}

// KVMsgMark returns the mark for the message-boundary snapshot, or nil.
func (h *Host) KVMsgMark(slot int, prompt []int, from int, adapterID uint64, sc KVScope, acq *KVAcquired, keep *PrefillMark) *PrefillMark {
	align, ok := h.KVMarkGuard(sc, adapterID)
	if !ok {
		return nil
	}
	minTail := h.PrefillTailMin
	recurrent := h.hasRecurrentState()
	q, ok := kvsnap.MsgPlan(kvsnap.BoundaryConfig{
		PromptLen:   len(prompt),
		MsgBoundary: sc.MsgBoundary,
		Align:       align,
		MinPrompt:   h.snapMinPrompt(),
		MinTail:     minTail,
		Recurrent:   recurrent,
	})
	if !ok || q <= from || (keep != nil && q >= keep.Q) || !kvsnap.CutSegmentOK(q-from, align, minTail) {
		return nil
	}
	st := *h.KVSnaps
	if c := st.Covering(h.KVRootIn(0, sc.Namespace), prompt, q, 0); c != nil && kvsnap.HoldsTarget(c, q, recurrent) {
		st.Decline("held")
		return nil
	}
	return &PrefillMark{Q: q, Cut: true, At: func(pos int) {
		acq.MarkFired = true
		h.SnapshotRows(slot, prompt, pos, adapterID, sc, *acq)
		acq.MsgRows = pos
		if x := acq.snapFrom; x != nil && !st.Holds(x) {
			acq.snapFrom = nil
		}
	}}
}

// KVPrefillMark returns the mark chain a prefill of prompt from position from carries, or nil when none applies.
func (h *Host) KVPrefillMark(slot int, prompt []int, from int, adapterID uint64, sc KVScope, acq *KVAcquired) *PrefillMark {
	top := h.KVKeepMark(slot, prompt, from, adapterID, sc, acq)
	msg := h.KVMsgMark(slot, prompt, from, adapterID, sc, acq, top)
	lowest := top
	if msg != nil {
		msg.Pre, top, lowest = nil, cmp.Or(top, msg), msg
		if top != msg {
			top.Pre = msg
		}
	}
	if pre := h.KVPreambleMark(slot, prompt, from, adapterID, sc, acq, lowest); pre != nil {
		if lowest == nil {
			return pre
		}
		lowest.Pre = pre
	}
	return top
}

// KVPreambleMark returns the mark for a recurrent model's snapshot at the end of the system-and-tools preamble, or nil.
func (h *Host) KVPreambleMark(slot int, prompt []int, from int, adapterID uint64, sc KVScope, acq *KVAcquired, above *PrefillMark) *PrefillMark {
	align, ok := h.KVMarkGuard(sc, adapterID)
	recurrent := h.hasRecurrentState()
	if !ok || !recurrent || align <= 0 || len(prompt) < h.snapMinPrompt() || sc.Boundary <= 0 || sc.Boundary >= len(prompt) {
		return nil
	}
	minTail := h.PrefillTailMin
	q, ok := kvsnap.RecurrentCut(len(prompt), sc.Boundary, minTail)
	if !ok || q <= from || (above != nil && q >= above.Q) || !kvsnap.CutSegmentOK(q-from, align, minTail) {
		return nil
	}
	st := *h.KVSnaps
	if c := st.Covering(h.KVRootIn(0, sc.Namespace), prompt, q, 0); c != nil && kvsnap.HoldsTarget(c, q, recurrent) {
		st.Decline("held")
		return nil
	}
	return &PrefillMark{Q: q, Cut: true, At: func(pos int) {
		acq.MarkFired = true
		acq.PreambleRows = pos
		h.SnapshotRows(slot, prompt, pos, adapterID, sc, *acq)
		if x := acq.snapFrom; x != nil && !st.Holds(x) {
			acq.snapFrom = nil
		}
	}}
}

// KVKeepMark is the boundary snapshot's mark: Keep's for a recurrent model, the prefix boundary's otherwise.
func (h *Host) KVKeepMark(slot int, prompt []int, from int, adapterID uint64, sc KVScope, acq *KVAcquired) *PrefillMark {
	q, ok := h.KVBoundaryPlan(prompt, sc, adapterID)
	recurrent := h.hasRecurrentState()
	if !ok || (acq.SnapRows > 0 && !recurrent) || q <= from {
		return nil
	}
	align, _ := h.stepMarkAlign()
	if recurrent {
		if !kvsnap.CutSegmentOK(q-from, align, h.PrefillTailMin) {
			return nil
		}
	} else if (q-from)%align != 0 {
		return nil
	}
	if st := *h.KVSnaps; st != nil {
		slack := kvsnap.CoverSlack(snapSlackOf(h.Snapshotter()), recurrent)
		if c := st.Covering(h.KVRootIn(0, sc.Namespace), prompt, q, slack); c != nil && (!recurrent || kvsnap.HoldsTarget(c, q, recurrent)) {
			st.Decline("held")
			return nil
		}
	}
	return &PrefillMark{Q: q, Cut: recurrent, At: func(pos int) {
		acq.MarkFired = true
		h.SnapshotRows(slot, prompt, pos, adapterID, sc, *acq)
	}}
}

// PrefillMark is a position at which a prefill is to call back: between two passes of its batched prefill, or, when cut, where it was split.
type PrefillMark struct {
	Q   int
	At  func(pos int)
	Cut bool
	Pre *PrefillMark
}

// SnapshotRows takes a snapshot of prompt's first b rows from slot.
func (h *Host) SnapshotRows(slot int, prompt []int, b int, adapterID uint64, sc KVScope, acq KVAcquired) {
	sn := h.Snapshotter()
	recurrent := h.hasRecurrentState()
	if sn == nil || sc.Off || h.ShareOff || h.ReuseDisabled || h.SnapshotsBlocked() {
		return
	}
	st := *h.KVSnaps
	n := len(prompt)
	switch {
	case adapterID != 0:
		st.Decline("adapter")
		return
	case n < h.snapMinPrompt():
		st.Decline("short")
		return
	case acq.Reuse > 0 && acq.SnapRows == 0:
		st.Decline("not cold")
		return
	}
	cold := h.KVClassMap().ColdClasses(n)
	if len(cold) == 0 || cold[0].Class == kvmgr.ClassUnknown {
		st.Decline("class")
		return
	}
	root := h.KVRootIn(0, sc.Namespace)
	replaced, ok := st.CheckAdmission(root, prompt, b, acq.snapFrom, acq.MsgRows, snapSlackOf(sn), recurrent)
	if !ok {
		return
	}
	need := sn.KVSnapshotBytes(b)
	if need <= 0 {
		st.Decline("backend")
		return
	}
	if need > st.Budget() {
		st.Decline("budget")
		return
	}
	if replaced != nil {
		st.Drop(replaced)
		replaced = nil
	}
	if !h.KVSnapReserve(st, need) {
		st.Decline("budget")
		return
	}
	t0 := time.Now()
	snap, err := sn.SnapshotKV(slot, b)
	d := time.Since(t0)
	if err != nil {
		st.Release(need)
		if dec, ok := errors.AsType[*KVCopyDecline](err); ok {
			st.DeclineDetail("backend", dec.Reason)
		} else {
			st.DeclineDetail("error", err.Error())
		}
		return
	}
	x := &kvsnap.Entry{
		Root:     root,
		IDs:      slices.Clone(prompt[:b]),
		Classes:  kvsnap.ClipRuns(cold, b),
		Handle:   snap,
		Preamble: recurrent && acq.PreambleRows == b && b > 0,
	}
	st.Add(x, replaced, d, need)
	used := st.Stats().Used
	fmt.Fprintf(h.stderr(), "kv snapshot: took %d rows (%.0f MiB host) from slot %d in %.1f ms; %.0f of %.0f MiB held\n",
		b, float64(snap.Bytes())/(1<<20), slot, float64(d.Microseconds())/1000, float64(used)/(1<<20), float64(st.Budget())/(1<<20))
}

// KVSnapReserve charges need bytes for a snapshot about to be taken.
func (h *Host) KVSnapReserve(st *kvsnap.Store, need int64) bool {
	if kv := h.resKV(); kv.KVDemoterAvailable() && h.KVDemote {
		return kv.ReserveKVHost(need)
	}
	return st.Reserve(need, true)
}

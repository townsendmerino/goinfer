package decoder

import (
	"cmp"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"time"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
	"github.com/townsendmerino/goinfer/internal/runtime/kvsnap"
)

// stderrKV is where a snapshot taken or restored logs its trace; tests may redirect it.
var stderrKV io.Writer = os.Stderr

// kvSnapMinPrompt is the shortest prompt whose prefix is worth a host snapshot; tests may lower it.
var kvSnapMinPrompt = 2048

// KeepsRecurrentSnapshots reports whether this model's prefix snapshots carry its recurrent state.
func (m *Model) KeepsRecurrentSnapshots() bool {
	sn := m.snapshotter()
	return m.hasRecurrentState() && sn != nil && sn.KVSnapshotRecurrent()
}

// KVSnapshotStats reports the model's host snapshots (false when off).
func (m *Model) KVSnapshotStats() (KVSnapshotStats, bool) {
	if m.kvSnaps == nil {
		return KVSnapshotStats{}, false
	}
	return m.kvSnaps.Stats(), true
}

// KVSnapshotsSupported reports whether this model keeps prefix snapshots: a budget, and a resident that can hold them.
func (m *Model) KVSnapshotsSupported() bool { return m.snapshotter() != nil }

// KVHostBytes returns the host memory budget for prefix snapshots this model was loaded with (Options.KVHostBytes), 0 when off.
func (m *Model) KVHostBytes() int64 { return m.kvSnaps.Budget() }

func (m *Model) snapshotter() ResidentKVSubsystem {
	if m.kvSnaps == nil || m.resident == nil {
		return nil
	}
	kv := m.resKV()
	if !kv.KVSnapshotterAvailable() {
		return nil
	}
	return kv
}

func snapSlackOf(sn ResidentKVSubsystem) int {
	if sn != nil {
		if r := sn.KVSnapshotRollback(); r > 0 {
			return r
		}
	}
	return kvmgr.RollbackFloor
}

func snapshotsBlocked(recurrent bool, sn ResidentKVSubsystem) bool {
	return recurrent && (sn == nil || !sn.KVSnapshotRecurrent())
}

// residentRestoreSnapshot restores a held snapshot's rows into the bound slot when that serves more of prompt than the slot's own rows do.
func (m *Model) residentRestoreSnapshot(prompt []int, lora *loraRuntime, sc KVScope, acq *kvAcquired) {
	sn := m.snapshotter()
	recurrent := m.hasRecurrentState()
	if sn == nil || sc.Off || m.opts.KVShareOff || lora != nil || m.residentReuseDisabled() || snapshotsBlocked(recurrent, sn) || acq.slot < 0 {
		return
	}
	st := m.kvSnaps
	root := m.kvRootIn(nil, sc.Namespace)
	x, q := st.SelectRestore(root, prompt, acq.reuse, m.prefillTailMinRows(), m.kvClassMap(), recurrent)
	if x == nil {
		return
	}
	t0 := time.Now()
	err := sn.RestoreKV(x.Handle, acq.slot, q)
	d := time.Since(t0)
	if err != nil {
		if dec, ok := errors.AsType[*KVCopyDecline](err); ok {
			st.DeclineDetail("backend", dec.Reason)
			acq.declined = dec.Reason
			return
		}
		st.DeclineDetail("error", err.Error())
		acq.declined = err.Error()
		acq.reuse = 0
		m.residentForgetIDs()
		return
	}
	m.kvSlots().Restored(acq.slot, root, x.IDs[:q], x.Classes)
	acq.reuse, acq.snapRows, acq.snapDur, acq.snapFrom, acq.shared, acq.src = q, q, d, x, 0, -1
	st.Use(x, q, d)
	fmt.Fprintf(stderrKV, "kv snapshot: restored %d rows of %d into slot %d in %.1f ms\n", q, x.Rows(), acq.slot, float64(d.Microseconds())/1000)
}

// residentSnapshotPrompt takes an end-of-prompt snapshot of prompt's full blocks from slot before decode begins.
func (m *Model) residentSnapshotPrompt(slot int, prompt []int, lora *loraRuntime, sc KVScope, acq kvAcquired) {
	if m.hasRecurrentState() {
		return
	}
	if q, ok := m.kvBoundaryPlan(prompt, sc, lora); ok {
		if acq.markFired || acq.snapRows > 0 {
			return
		}
		if st := m.kvSnaps; st != nil && st.Covering(m.kvRootIn(nil, sc.Namespace), prompt, q, snapSlackOf(m.snapshotter())) != nil {
			st.Decline("held")
			return
		}
	}
	m.residentSnapshotRows(slot, prompt, len(prompt)/kvmgr.BlockRows*kvmgr.BlockRows, lora, sc, acq)
}

// kvBoundaryPlan computes the prefill position at which this request should take its boundary or Keep snapshot instead of at the end of the prompt.
func (m *Model) kvBoundaryPlan(prompt []int, sc KVScope, lora *loraRuntime) (q int, ok bool) {
	align, ok := m.kvMarkGuard(sc, lora)
	if !ok {
		return 0, false
	}
	return kvsnap.BoundaryPlan(kvsnap.BoundaryConfig{
		PromptLen: len(prompt),
		Boundary:  sc.Boundary,
		Keep:      sc.Keep,
		Align:     align,
		MinPrompt: kvSnapMinPrompt,
		Slack:     snapSlackOf(m.snapshotter()),
		MinTail:   m.prefillTailMinRows(),
		Recurrent: m.hasRecurrentState(),
	})
}

func (m *Model) kvMarkGuard(sc KVScope, lora *loraRuntime) (align int, ok bool) {
	align, hasMark := stepMarkAlign(m.resident)
	sn := m.snapshotter()
	if sn == nil || !hasMark || lora != nil || sc.Off || m.opts.KVShareOff || m.residentReuseDisabled() || snapshotsBlocked(m.hasRecurrentState(), sn) {
		return 0, false
	}
	return align, true
}

func (m *Model) kvMsgMark(slot int, prompt []int, from int, lora *loraRuntime, sc KVScope, acq *kvAcquired, keep *prefillMark) *prefillMark {
	align, ok := m.kvMarkGuard(sc, lora)
	if !ok {
		return nil
	}
	minTail := m.prefillTailMinRows()
	recurrent := m.hasRecurrentState()
	q, ok := kvsnap.MsgPlan(kvsnap.BoundaryConfig{
		PromptLen:   len(prompt),
		MsgBoundary: sc.MsgBoundary,
		Align:       align,
		MinPrompt:   kvSnapMinPrompt,
		MinTail:     minTail,
		Recurrent:   recurrent,
	})
	if !ok || q <= from || (keep != nil && q >= keep.q) || !kvsnap.CutSegmentOK(q-from, align, minTail) {
		return nil
	}
	st := m.kvSnaps
	if c := st.Covering(m.kvRootIn(nil, sc.Namespace), prompt, q, 0); c != nil && kvsnap.HoldsTarget(c, q, recurrent) {
		st.Decline("held")
		return nil
	}
	return &prefillMark{q: q, cut: true, at: func(pos int) {
		acq.markFired = true
		m.residentSnapshotRows(slot, prompt, pos, lora, sc, *acq)
		acq.msgRows = pos
		if x := acq.snapFrom; x != nil && !st.Holds(x) {
			acq.snapFrom = nil
		}
	}}
}

// kvPrefillMark returns the mark chain a prefill of prompt from position from carries, or nil when none applies.
func (m *Model) kvPrefillMark(slot int, prompt []int, from int, lora *loraRuntime, sc KVScope, acq *kvAcquired) *prefillMark {
	top := m.kvKeepMark(slot, prompt, from, lora, sc, acq)
	msg := m.kvMsgMark(slot, prompt, from, lora, sc, acq, top)
	lowest := top
	if msg != nil {
		msg.pre, top, lowest = nil, cmp.Or(top, msg), msg
		if top != msg {
			top.pre = msg
		}
	}
	if pre := m.kvPreambleMark(slot, prompt, from, lora, sc, acq, lowest); pre != nil {
		if lowest == nil {
			return pre
		}
		lowest.pre = pre
	}
	return top
}

func (m *Model) kvPreambleMark(slot int, prompt []int, from int, lora *loraRuntime, sc KVScope, acq *kvAcquired, above *prefillMark) *prefillMark {
	align, ok := m.kvMarkGuard(sc, lora)
	recurrent := m.hasRecurrentState()
	if !ok || !recurrent || align <= 0 || len(prompt) < kvSnapMinPrompt || sc.Boundary <= 0 || sc.Boundary >= len(prompt) {
		return nil
	}
	minTail := m.prefillTailMinRows()
	q, ok := kvsnap.RecurrentCut(len(prompt), sc.Boundary, minTail)
	if !ok || q <= from || (above != nil && q >= above.q) || !kvsnap.CutSegmentOK(q-from, align, minTail) {
		return nil
	}
	st := m.kvSnaps
	if c := st.Covering(m.kvRootIn(nil, sc.Namespace), prompt, q, 0); c != nil && kvsnap.HoldsTarget(c, q, recurrent) {
		st.Decline("held")
		return nil
	}
	return &prefillMark{q: q, cut: true, at: func(pos int) {
		acq.markFired = true
		acq.preambleRows = pos
		m.residentSnapshotRows(slot, prompt, pos, lora, sc, *acq)
		if x := acq.snapFrom; x != nil && !st.Holds(x) {
			acq.snapFrom = nil
		}
	}}
}

func (m *Model) kvKeepMark(slot int, prompt []int, from int, lora *loraRuntime, sc KVScope, acq *kvAcquired) *prefillMark {
	q, ok := m.kvBoundaryPlan(prompt, sc, lora)
	recurrent := m.hasRecurrentState()
	if !ok || (acq.snapRows > 0 && !recurrent) || q <= from {
		return nil
	}
	align, _ := stepMarkAlign(m.resident)
	if recurrent {
		if !kvsnap.CutSegmentOK(q-from, align, m.prefillTailMinRows()) {
			return nil
		}
	} else if (q-from)%align != 0 {
		return nil
	}
	if st := m.kvSnaps; st != nil {
		slack := kvsnap.CoverSlack(snapSlackOf(m.snapshotter()), recurrent)
		if c := st.Covering(m.kvRootIn(nil, sc.Namespace), prompt, q, slack); c != nil && (!recurrent || kvsnap.HoldsTarget(c, q, recurrent)) {
			st.Decline("held")
			return nil
		}
	}
	return &prefillMark{q: q, cut: recurrent, at: func(pos int) {
		acq.markFired = true
		m.residentSnapshotRows(slot, prompt, pos, lora, sc, *acq)
	}}
}

type prefillMark struct {
	q   int
	at  func(pos int)
	cut bool
	pre *prefillMark
}

func (m *Model) residentSnapshotRows(slot int, prompt []int, b int, lora *loraRuntime, sc KVScope, acq kvAcquired) {
	sn := m.snapshotter()
	recurrent := m.hasRecurrentState()
	if sn == nil || sc.Off || m.opts.KVShareOff || m.residentReuseDisabled() || snapshotsBlocked(recurrent, sn) {
		return
	}
	st := m.kvSnaps
	n := len(prompt)
	switch {
	case lora != nil:
		st.Decline("adapter")
		return
	case n < kvSnapMinPrompt:
		st.Decline("short")
		return
	case acq.reuse > 0 && acq.snapRows == 0:
		st.Decline("not cold")
		return
	}
	cold := m.kvClassMap().ColdClasses(n)
	if len(cold) == 0 || cold[0].Class == kvmgr.ClassUnknown {
		st.Decline("class")
		return
	}
	root := m.kvRootIn(nil, sc.Namespace)
	replaced, ok := st.CheckAdmission(root, prompt, b, acq.snapFrom, acq.msgRows, snapSlackOf(sn), recurrent)
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
	reserved := false
	if kv := m.resKV(); kv.KVDemoterAvailable() && m.opts.KVDemote {
		reserved = kv.ReserveKVHost(need)
	} else {
		reserved = st.Reserve(need, true)
	}
	if !reserved {
		st.Decline("budget")
		return
	}
	t0 := time.Now()
	h, err := sn.SnapshotKV(slot, b)
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
		Handle:   h,
		Preamble: recurrent && acq.preambleRows == b && b > 0,
	}
	st.Add(x, replaced, d, need)
	used := st.Stats().Used
	fmt.Fprintf(stderrKV, "kv snapshot: took %d rows (%.0f MiB host) from slot %d in %.1f ms; %.0f of %.0f MiB held\n",
		b, float64(h.Bytes())/(1<<20), slot, float64(d.Microseconds())/1000, float64(used)/(1<<20), float64(st.Budget())/(1<<20))
}

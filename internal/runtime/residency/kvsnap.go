package residency

import (
	"cmp"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// Prefix snapshots in host memory. After a cold first prefill of a long prompt the
// resident's KV for the prompt's full blocks is copied to pinned host memory, keyed by the token ids and the KV root, which keys on
// (namespace, model identity, adapter, KV precision). A later request whose prompt starts with those ids restores the rows into its
// slot by device copy and prefills only the rest. It keeps a prefix that a one-slot card would overwrite when the next session
// starts. The rows are prefill-class rows copied before any decode step, so a restored prefix is the prefix computed
// (task-kv-architecture §5.6, property 2).

// kvSnapMinShared is the shortest prefix, in ids, two prompts must share for a held snapshot's failure to serve one to count as
// a first prompt that does not qualify.
const kvSnapMinShared = 512

// KeepsRecurrentSnapshots reports whether this model's prefix snapshots carry its recurrent state: it has recurrent state, and its resident's
// snapshots hold it (KVSnapshotterAvailable.KVSnapshotRecurrent). A route computes KVScope.Keep only for such a model.
func (h *Host) KeepsRecurrentSnapshots() bool {
	return h.hasRecurrentState() && h.RecurrentSnapshots()
}

// recurrentSnapshots reports whether the snapshotter carries recurrent state.
func (h *Host) RecurrentSnapshots() bool {
	sn := h.Snapshotter()
	return sn != nil && sn.KVSnapshotRecurrent()
}

// snapshotsBlocked reports whether this model's state is outside what a snapshot holds: recurrent state its snapshotter does not carry.
func (h *Host) SnapshotsBlocked() bool { return h.hasRecurrentState() && !h.RecurrentSnapshots() }

// KVSnapshotStats is a model's host snapshots, for /admin/status and the request log.
type KVSnapshotStats struct {
	Budget         int64 `json:"budget_bytes"` // host bytes allowed
	Used           int64 `json:"used_bytes"`   // and held
	Held           int   `json:"held"`
	Taken          int   `json:"taken"`    // snapshots taken
	Restored       int   `json:"restored"` // requests that restored one
	RestoredRows   int64 `json:"restored_rows"`
	Demoted        int   `json:"demoted"`        // released slots whose rows were kept in host memory
	DemoteDropped  int   `json:"demote_dropped"` // of those, copies dropped to make room for another copy or a snapshot
	Promoted       int   `json:"promoted"`       // of those, requests that brought them back
	PromotedBlocks int64 `json:"promoted_blocks"`
	Evicted        int   `json:"evicted"`        // snapshots dropped for room or replaced by a longer one
	EvictedUnused  int   `json:"evicted_unused"` // of those, ones never restored
	// Unqualified counts requests that shared at least kvSnapMinShared ids with a held snapshot that its window could not serve:
	// the snapshot's prompt went on more than the rollback past the shared prefix.
	Unqualified  int   `json:"unqualified"`
	TakeNanos    int64 `json:"take_ns"`
	RestoreNanos int64 `json:"restore_ns"`
	// Declined counts, by reason, the snapshots not taken and the restores not made.
	Declined    map[string]int   `json:"declined"`
	LastDecline string           `json:"last_decline,omitempty"` // the backend's own words for the latest refusal
	Snapshots   []KVSnapshotInfo `json:"snapshots,omitempty"`
}

// KVSnapshotInfo is one held snapshot.
type KVSnapshotInfo struct {
	Rows     int   `json:"rows"`
	Bytes    int64 `json:"bytes"`
	Restores int   `json:"restores"`
	AgeSecs  int64 `json:"age_s"`
}

type kvSnap struct {
	root     kvmgr.Root
	ids      []int // the tokens of rows [0, h.Rows())
	classes  []kvmgr.ClassRun
	h        resident.KVSnapshot
	restores int
	last     uint64
	created  time.Time
	// preamble marks a recurrent snapshot taken at the end of the system-and-tools preamble (KVScope.Boundary): other conversations share its
	// prefix, so no deeper snapshot of the same prefix replaces it; only the store's budget drops it.
	preamble bool
}

// kvSnapStore is a model's held snapshots under a host byte budget, most recently used first.
type KVSnapStore struct {
	mu     sync.Mutex
	budget int64
	used   int64
	tick   uint64
	list   []*kvSnap
	st     KVSnapshotStats
}

// newKVSnapStore is the store for a budget in bytes, nil (no snapshots, nothing held) for a budget that is not positive.
func NewKVSnapStore(budget int64) *KVSnapStore {
	if budget <= 0 {
		return nil
	}
	return &KVSnapStore{budget: budget, st: KVSnapshotStats{Declined: map[string]int{}}}
}

func (s *KVSnapStore) decline(reason string) {
	s.mu.Lock()
	s.st.Declined[reason]++
	s.mu.Unlock()
}

// best is the held snapshot under root that shares the most leading ids with prompt, and that count; nil with none shared.
func (s *KVSnapStore) best(root kvmgr.Root, prompt []int) (*kvSnap, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out *kvSnap
	lcp := 0
	for _, x := range s.list {
		if x.root != root {
			continue
		}
		if n := kvmgr.CommonPrefixLen(x.ids, prompt); n > lcp {
			out, lcp = x, n
		}
	}
	return out, lcp
}

// snapMatch is a held snapshot and how many leading ids it shares with a prompt.
type snapMatch struct {
	x   *kvSnap
	lcp int
}

// candidates are the held snapshots under root that share at least one block of leading ids with prompt, the ones that can restore the most
// first (by min(shared ids, rows)): a request restores the first that its window can serve, not merely the longest match, which may be a
// snapshot taken at the end of a long prompt whose ring no longer holds the shared prefix.
func (s *KVSnapStore) candidates(root kvmgr.Root, prompt []int) []snapMatch {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []snapMatch
	for _, x := range s.list {
		if x.root != root {
			continue
		}
		if n := kvmgr.CommonPrefixLen(x.ids, prompt); n >= kvmgr.BlockRows {
			out = append(out, snapMatch{x, n})
		}
	}
	slices.SortStableFunc(out, func(a, b snapMatch) int {
		return cmp.Compare(min(b.lcp, b.x.h.Rows()), min(a.lcp, a.x.h.Rows()))
	})
	return out
}

// covering is the held snapshot under root that already serves a prompt whose full blocks end at b: it shares all but at most
// slack of min(its rows, b) leading ids with it, and its window can serve that share. Taking another from that prompt would hold the same
// prefix twice.
func (s *KVSnapStore) covering(root kvmgr.Root, prompt []int, b, slack int) *kvSnap {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, x := range s.list {
		if x.root != root {
			continue
		}
		lcp := kvmgr.CommonPrefixLen(x.ids, prompt)
		if lcp < min(x.h.Rows(), b)-slack {
			continue
		}
		// It must also be able to serve what it shares: a snapshot much longer than b has a ring that no longer holds the rows behind a shorter
		// prefix, and does not cover it (a boundary snapshot beside an end-of-prompt one of the same family).
		if q := min(lcp, x.h.Rows(), b) / kvmgr.BlockRows * kvmgr.BlockRows; q > 0 && x.h.Serves(q) {
			return x
		}
	}
	return nil
}

// holds reports whether x is still in the store.
func (s *KVSnapStore) holds(x *kvSnap) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Contains(s.list, x)
}

// use marks x as just used and counts a restore of rows rows.
func (s *KVSnapStore) use(x *kvSnap, rows int, d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tick++
	x.last = s.tick
	x.restores++
	s.st.Restored++
	s.st.RestoredRows += int64(rows)
	s.st.RestoreNanos += int64(d)
	s.sortLocked()
}

func (s *KVSnapStore) sortLocked() {
	slices.SortFunc(s.list, func(a, b *kvSnap) int {
		return cmp.Compare(b.last, a.last)
	})
}

// removeLocked drops x, releasing its host memory. No-op if x was already evicted.
func (s *KVSnapStore) removeLocked(x *kvSnap) {
	found := false
	for i, y := range s.list {
		if y == x {
			s.list = append(s.list[:i], s.list[i+1:]...)
			found = true
			break
		}
	}
	if !found {
		return
	}
	s.used -= x.h.Bytes()
	x.h.Release()
	s.st.Evicted++
	if x.restores == 0 {
		s.st.EvictedUnused++
	}
}

// roomLocked makes need bytes free by dropping the least recently used snapshots, and reports whether it could. Bytes reserved outside the
// list cannot be dropped, so when they alone leave no room it drops nothing.
func (s *KVSnapStore) roomLocked(need int64) bool {
	var droppable int64
	for _, x := range s.list {
		droppable += x.h.Bytes()
	}
	if s.used-droppable+need > s.budget {
		return false
	}
	for s.used+need > s.budget {
		s.removeLocked(s.list[len(s.list)-1])
	}
	return true
}

// Fits implements KVHostBudget.
func (s *KVSnapStore) Fits(n, copies int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	left := s.used - copies + n
	for _, x := range s.list {
		left -= x.h.Bytes()
	}
	return left <= s.budget
}

// Reserve implements KVHostBudget.
func (s *KVSnapStore) Reserve(n int64, dropSnapshots bool) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if dropSnapshots && !s.roomLocked(n) {
		return false
	}
	if s.used+n > s.budget {
		return false
	}
	s.used += n
	return true
}

// Release implements KVHostBudget.
func (s *KVSnapStore) Release(n int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.used -= n
}

// Kept implements KVHostBudget.
func (s *KVSnapStore) Kept() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.Demoted++
}

// Dropped implements KVHostBudget.
func (s *KVSnapStore) Dropped() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.DemoteDropped++
}

// Refused implements KVHostBudget.
func (s *KVSnapStore) Refused() { s.decline("demote budget") }

// promoted counts blocks a released slot took back from host memory. Nil-safe: a resident promotes only what a budget let it keep.
func (s *KVSnapStore) Promoted(blocks int) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.Promoted++
	s.st.PromotedBlocks += int64(blocks)
}

// drop removes x and frees its host memory.
func (s *KVSnapStore) drop(x *kvSnap) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removeLocked(x)
}

// add holds x (just taken, its bytes already allocated, reserved of them already charged by Reserve) and drops replaced, if any.
func (s *KVSnapStore) add(x *kvSnap, replaced *kvSnap, d time.Duration, reserved int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tick++
	x.last, x.created = s.tick, time.Now()
	s.list = append([]*kvSnap{x}, s.list...)
	s.used += x.h.Bytes() - reserved
	s.st.Taken++
	s.st.TakeNanos += int64(d)
	if replaced != nil {
		s.removeLocked(replaced)
	}
}

// releaseAll frees every held snapshot; the model is going away or its geometry has changed.
func (s *KVSnapStore) ReleaseAll() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.list) > 0 {
		s.removeLocked(s.list[0])
	}
}

func (s *KVSnapStore) Stats() KVSnapshotStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.st
	out.Budget, out.Used, out.Held = s.budget, s.used, len(s.list)
	out.Declined = make(map[string]int, len(s.st.Declined))
	maps.Copy(out.Declined, s.st.Declined)
	for _, x := range s.list {
		out.Snapshots = append(out.Snapshots, KVSnapshotInfo{Rows: x.h.Rows(), Bytes: x.h.Bytes(), Restores: x.restores, AgeSecs: int64(time.Since(x.created).Seconds())})
	}
	return out
}

// KVSnapshotStats reports the model's host snapshots: the budget, what is held, and what has been taken, restored, evicted and
// refused. ok is false when the model keeps none (a budget of 0, or no resident that can).
func (h *Host) KVSnapshotStats() (KVSnapshotStats, bool) {
	if (*h.KVSnaps) == nil {
		return KVSnapshotStats{}, false
	}
	return (*h.KVSnaps).Stats(), true
}

// KVSnapshotsSupported reports whether this model keeps prefix snapshots: a budget, and a resident that can hold them.
func (h *Host) KVSnapshotsSupported() bool { return h.Snapshotter() != nil }

// KVHostBytes is the host memory budget for prefix snapshots this model was loaded with (Options.KVHostBytes), 0 when off.
func (h *Host) KVHostBytes() int64 {
	if (*h.KVSnaps) == nil {
		return 0
	}
	return (*h.KVSnaps).budget
}

// snapshotter is the resident's snapshot capability, nil when snapshots are off for this model or the resident has none.
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

// residentRestoreSnapshot restores a held snapshot's rows into the bound slot when that serves more of prompt than the slot's own
// rows do (acq.Reuse, after any copy from another slot). Nothing happens under an Off scope or Options.KVShareOff, with an
// adapter, for a recurrent family whose snapshots do not carry its state, or without a snapshotter. A recurrent snapshot restores only
// to its exact end, and declines when the rest of the prompt would be a pass under the resident's minimum. The rows come back as prefill-class rows of the same ids, so the
// requester's cold class must be the snapshot's and a pass of at least the fast minimum must remain (kvmgr.ClassMap.CapHit). A
// refusal before anything is written leaves acq as it was; a failure after leaves the slot unknown.
func (h *Host) RestoreSnapshot(prompt []int, adapterID uint64, sc KVScope, acq *KVAcquired) {
	sn := h.Snapshotter()
	if sn == nil || sc.Off || h.ShareOff || adapterID != 0 || h.ReuseDisabled || h.SnapshotsBlocked() || acq.Slot < 0 {
		return
	}
	st := (*h.KVSnaps)
	root := h.KVRootIn(0, sc.Namespace)
	n, cm := len(prompt), h.KVClassMap()
	recurrent := h.hasRecurrentState()
	// The first candidate whose window can serve its share wins. With one snapshot per family this is the old rule; with a boundary snapshot
	// and an end snapshot of the same family held, a longer match that the window cannot serve does not hide the one that can.
	var x *kvSnap
	var q int
	var sawWindow, sawClass bool
	cands := st.candidates(root, prompt)
	if len(cands) == 0 {
		return
	}
	for _, c := range cands {
		cq := min(c.lcp, c.x.h.Rows()) / kvmgr.BlockRows * kvmgr.BlockRows
		cq = min(cq, (n-1)/kvmgr.BlockRows*kvmgr.BlockRows) // a token is left to compute, for the first logits
		cq = cm.CapHit(n, cq)
		if cq <= acq.Reuse {
			continue // the slot's own rows, or a copy from another slot, serve as much
		}
		if recurrent && cq != c.x.h.Rows() {
			continue // the state is the one after the snapshot's last row: no shorter prefix
		}
		if recurrent && n-cq < h.PrefillTailMin {
			st.decline("short suffix") // a pass this short would run other kernels than the cold prefill's
			continue
		}
		if !c.x.h.Serves(cq) {
			if c.lcp >= kvSnapMinShared {
				sawWindow = true
			}
			continue
		}
		if !classesEqual(clipRuns(c.x.classes, cq), clipRuns(cm.ColdClasses(n), cq)) {
			sawClass = true
			continue
		}
		x, q = c.x, cq
		break
	}
	if x == nil {
		switch {
		case sawWindow:
			st.mu.Lock()
			st.st.Unqualified++
			st.mu.Unlock()
			st.decline("window")
		case sawClass:
			st.decline("class")
		}
		return
	}
	a := h.KVSlots()
	t0 := time.Now()
	err := sn.RestoreKV(x.h, acq.Slot, q)
	d := time.Since(t0)
	if err != nil {
		if dec, ok := errors.AsType[*KVCopyDecline](err); ok {
			st.mu.Lock()
			st.st.Declined["backend"]++
			st.st.LastDecline = dec.Reason
			st.mu.Unlock()
			acq.Declined = dec.Reason
			return
		}
		st.mu.Lock()
		st.st.Declined["error"]++
		st.st.LastDecline = err.Error()
		st.mu.Unlock()
		acq.Declined = err.Error()
		acq.Reuse = 0 // the restore may have written some rows; the slot's own are no longer trusted
		h.ForgetIDs()
		return
	}
	a.Restored(acq.Slot, root, x.ids[:q], x.classes)
	acq.Reuse, acq.SnapRows, acq.SnapDur, acq.snapFrom, acq.Shared, acq.Src = q, q, d, x, 0, -1
	st.use(x, q, d)
	fmt.Fprintf(h.stderr(), "kv snapshot: restored %d rows of %d into slot %d in %.1f ms\n", q, x.h.Rows(), acq.Slot, float64(d.Microseconds())/1000)
}

// residentSnapshotPrompt takes a snapshot of prompt's full blocks from slot, which has just finished prefilling it, before any
// decode step. Admission is first touch: any prompt of at least h.snapMinPrompt() ids whose rows are all prefill-class (the prefill
// was cold, or began from a restore) and that no held snapshot already serves within the window's rollback. A longer prompt
// replaces a shorter snapshot of the same prefix. The budget is made by dropping the least recently used.
func (h *Host) SnapshotPrompt(slot int, prompt []int, adapterID uint64, sc KVScope, acq KVAcquired) {
	// One snapshot per prompt. When the prompt's tail after the stable section is longer than the rollback the boundary snapshot is the one that can
	// serve a later session, and the end-of-prompt snapshot (whose ring no longer holds the shared prefix) would replace or sit beside it for nothing:
	// skip it when the boundary one was taken, declined or restored, and fall back to it only when the mark never fired.
	if h.hasRecurrentState() {
		return // a recurrent model's state at the end of the prompt is the slot's own; its snapshot is the mark's (kvBoundaryPlan)
	}
	if q, ok := h.KVBoundaryPlan(prompt, sc, adapterID); ok {
		if acq.MarkFired || acq.SnapRows > 0 {
			return
		}
		if st := (*h.KVSnaps); st != nil && st.covering(h.KVRootIn(0, sc.Namespace), prompt, q, h.KVSnapSlack()) != nil {
			st.decline("held")
			return
		}
	}
	h.SnapshotRows(slot, prompt, len(prompt)/kvmgr.BlockRows*kvmgr.BlockRows, adapterID, sc, acq)
}

// kvSnapSlack is how many positions behind a snapshot's end a windowed layer still serves a restore.
func (h *Host) KVSnapSlack() int {
	if sn := h.Snapshotter(); sn != nil {
		if r := sn.KVSnapshotRollback(); r > 0 {
			return r
		}
	}
	return kvmgr.RollbackFloor
}

// kvBoundaryPlan is the position at which this request's prefill should take its snapshot instead of at the end of the prompt: the prefix boundary the
// route computed (KVScope.Boundary) rounded down to a pass of the resident's batched prefill, when the part of the prompt after the boundary is longer
// than a windowed layer's rollback, so an end-of-prompt snapshot could not serve a later prompt that diverges there. For a recurrent model it is
// KVScope.Keep rounded down to a pass instead, whatever the rollback (residentSnapshotPrompt takes no end snapshot for one). ok is false when the resident
// cannot mark its passes, there is no boundary, the prompt is short, or the part after the boundary fits the rollback (the end snapshot serves).
func (h *Host) KVBoundaryPlan(prompt []int, sc KVScope, adapterID uint64) (q int, ok bool) {
	var align int
	align, ok = h.KVMarkGuard(sc, adapterID)
	if !ok {
		return 0, false
	}
	n := len(prompt)
	if h.hasRecurrentState() {
		// A recurrent model's snapshot is taken where the next turn's prompt parts from this one (KVScope.Keep): the prefill is cut there
		// (recurrentCut), so the state exists at that row and the next turn re-prefills from it.
		if align <= 0 || n < h.snapMinPrompt() || sc.Keep <= 0 || sc.Keep >= n {
			return 0, false
		}
		return h.RecurrentCut(n, sc.Keep)
	}
	if sc.Boundary <= 0 {
		return 0, false
	}
	if align <= 0 || n < h.snapMinPrompt() || sc.Boundary >= n || n-sc.Boundary <= h.KVSnapSlack() {
		return 0, false
	}
	q = sc.Boundary / align * align
	if q < kvSnapMinShared || q%kvmgr.BlockRows != 0 {
		return 0, false
	}
	return q, true
}

// recurrentCut is the row a recurrent model's snapshot for point is cut at in a prompt of n ids: point rounded down to a snapshot block, and
// lowered a block at a time while fewer rows than the resident's final-pass minimum would follow it (a shorter last pass runs other kernels
// than a cold prefill's). ok is false below a snapshot's minimum.
func (h *Host) RecurrentCut(n, point int) (q int, ok bool) {
	q = point / kvmgr.BlockRows * kvmgr.BlockRows
	for q > 0 && n-q < h.PrefillTailMin {
		q -= kvmgr.BlockRows
	}
	return q, q >= kvSnapMinShared
}

// cutSegmentOK reports whether a prefill of rows rows from a restore point or a cut keeps every pass at or above the resident's final-pass
// minimum: passes are align rows wide, so the last is rows mod align, or align itself when that divides.
func (h *Host) CutSegmentOK(rows, align int) bool {
	last := rows % align
	return rows >= h.PrefillTailMin && (last == 0 || last >= h.PrefillTailMin)
}

// kvMarkGuard is the width of the resident's marked passes, and ok when a snapshot mark may be planned for this request at all.
func (h *Host) KVMarkGuard(sc KVScope, adapterID uint64) (align int, ok bool) {
	align, hasMark := h.stepMarkAlign()
	if h.Snapshotter() == nil || !hasMark || adapterID != 0 || sc.Off || h.ShareOff || h.ReuseDisabled || h.SnapshotsBlocked() {
		return 0, false
	}
	return align, true
}

// kvMsgPlan is the position of a recurrent model's message-boundary snapshot (KVScope.MsgBoundary) rounded down to a pass, on the grid kvBoundaryPlan
// keeps Keep's on. ok is false for a model without recurrent state, or when the resident cannot mark, there is no boundary, or the prompt is short.
func (h *Host) KVMsgPlan(prompt []int, sc KVScope, adapterID uint64) (q int, ok bool) {
	align, ok := h.KVMarkGuard(sc, adapterID)
	if !ok || !h.hasRecurrentState() || align <= 0 || len(prompt) < h.snapMinPrompt() || sc.MsgBoundary <= 0 || sc.MsgBoundary >= len(prompt) {
		return 0, false
	}
	return h.RecurrentCut(len(prompt), sc.MsgBoundary)
}

// kvMsgMark is the mark for the message-boundary snapshot, or nil. It is taken beside Keep's, which a prompt that rewrites its last message from
// near the start does not match through: the snapshot Keep's mark takes later does not replace it (acq.MsgRows), though a recurrent snapshot of
// the same prefix that ends earlier is otherwise replaced (kvSnapHolds). keep is the request's Keep mark, nil when there is none; a point at or
// past it is Keep's to serve.
func (h *Host) KVMsgMark(slot int, prompt []int, from int, adapterID uint64, sc KVScope, acq *KVAcquired, keep *PrefillMark) *PrefillMark {
	q, ok := h.KVMsgPlan(prompt, sc, adapterID)
	if !ok || q <= from || (keep != nil && q >= keep.Q) {
		return nil
	}
	if align, _ := h.stepMarkAlign(); !h.CutSegmentOK(q-from, align) {
		return nil
	}
	st := (*h.KVSnaps)
	if c := st.covering(h.KVRootIn(0, sc.Namespace), prompt, q, 0); c != nil && h.KVSnapHolds(c, q) {
		st.decline("held")
		return nil
	}
	return &PrefillMark{Q: q, Cut: true, At: func(pos int) {
		acq.MarkFired = true
		h.SnapshotRows(slot, prompt, pos, adapterID, sc, *acq)
		acq.MsgRows = pos
		if x := acq.snapFrom; x != nil && !st.holds(x) {
			acq.snapFrom = nil // the snapshot this request restored from went to this one's admission; the next mark must not drop it again
		}
	}}
}

// kvPrefillMark returns the mark a cold prefill of prompt from position from should carry: the boundary position and the function that takes the
// snapshot there. nil when no boundary snapshot applies, when from does not start on a pass boundary the mark could meet, or when a held snapshot
// already serves the prefix (this request restored it, or another one covers it). A recurrent model's restored request still marks its own
// Keep past the restored point, so the kept state moves to each new question instead of staying at the first one
// (docs/measurements/b35-keep-advance-2026-10/README.md); admission then replaces the snapshot it restored from (residentSnapshotRows).
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

// kvPreambleMark is the mark for a recurrent model's snapshot at the end of the system-and-tools preamble (KVScope.Boundary), or nil. above is the
// request's lowest other mark: the preamble's point must lie below it. It is taken when no snapshot is held at that row, which is the first cold
// request to reach a preamble; the snapshot is kept apart from the replacement rule (kvSnap.preamble).
func (h *Host) KVPreambleMark(slot int, prompt []int, from int, adapterID uint64, sc KVScope, acq *KVAcquired, above *PrefillMark) *PrefillMark {
	align, ok := h.KVMarkGuard(sc, adapterID)
	if !ok || !h.hasRecurrentState() || align <= 0 || len(prompt) < h.snapMinPrompt() || sc.Boundary <= 0 || sc.Boundary >= len(prompt) {
		return nil
	}
	q, ok := h.RecurrentCut(len(prompt), sc.Boundary)
	if !ok || q <= from || (above != nil && q >= above.Q) || !h.CutSegmentOK(q-from, align) {
		return nil
	}
	st := (*h.KVSnaps)
	if c := st.covering(h.KVRootIn(0, sc.Namespace), prompt, q, 0); c != nil && h.KVSnapHolds(c, q) {
		st.decline("held")
		return nil
	}
	return &PrefillMark{Q: q, Cut: true, At: func(pos int) {
		acq.MarkFired = true
		acq.PreambleRows = pos
		h.SnapshotRows(slot, prompt, pos, adapterID, sc, *acq)
		if x := acq.snapFrom; x != nil && !st.holds(x) {
			acq.snapFrom = nil
		}
	}}
}

// kvKeepMark is the boundary snapshot's mark: Keep's for a recurrent model, the prefix boundary's otherwise.
func (h *Host) KVKeepMark(slot int, prompt []int, from int, adapterID uint64, sc KVScope, acq *KVAcquired) *PrefillMark {
	q, ok := h.KVBoundaryPlan(prompt, sc, adapterID)
	if !ok || (acq.SnapRows > 0 && !h.hasRecurrentState()) || q <= from {
		return nil
	}
	align, _ := h.stepMarkAlign()
	if h.hasRecurrentState() {
		if !h.CutSegmentOK(q-from, align) {
			return nil
		}
	} else if (q-from)%align != 0 {
		return nil // the passes of this prefill do not end at q
	}
	if st := (*h.KVSnaps); st != nil {
		if c := st.covering(h.KVRootIn(0, sc.Namespace), prompt, q, h.KVSnapCoverSlack(h.KVSnapSlack())); c != nil && (!h.hasRecurrentState() || h.KVSnapHolds(c, q)) {
			st.decline("held")
			return nil
		}
	}
	return &PrefillMark{Q: q, Cut: h.hasRecurrentState(), At: func(pos int) {
		acq.MarkFired = true
		h.SnapshotRows(slot, prompt, pos, adapterID, sc, *acq)
	}}
}

// prefillMark is a position at which a prefill is to call back: between two passes of its batched prefill, or, when cut, where it was split.
type PrefillMark struct {
	Q   int
	At  func(pos int)
	Cut bool
	Pre *PrefillMark
}

// kvSnapHolds reports whether held snapshot c already serves a snapshot of the first b rows, the rule kvPrefillMark and
// residentSnapshotRows share. Attention KV: c reaches b. A recurrent snapshot serves only its own end, so it holds b only when it ends
// exactly there; a shorter one is replaced and a longer one is a different point, kept beside the new one.
func (h *Host) KVSnapHolds(c *kvSnap, b int) bool {
	if h.hasRecurrentState() {
		return c.h.Rows() == b
	}
	return c.h.Rows() >= b
}

// kvSnapCoverSlack is the rollback a covering search allows: slack for attention KV, 0 for a recurrent model, whose snapshot shares
// nothing with a prompt that leaves its prefix before its end.
func (h *Host) KVSnapCoverSlack(slack int) int {
	if h.hasRecurrentState() {
		return 0
	}
	return slack
}

// residentSnapshotRows takes a snapshot of prompt's first b rows from slot, which holds them: at the end of the prefill (b the full blocks of the
// prompt, residentSnapshotPrompt) or between two passes of it (b the boundary, kvPrefillMark). It is the same admission either way.
func (h *Host) SnapshotRows(slot int, prompt []int, b int, adapterID uint64, sc KVScope, acq KVAcquired) {
	sn := h.Snapshotter()
	if sn == nil || sc.Off || h.ShareOff || h.ReuseDisabled || h.SnapshotsBlocked() {
		return
	}
	st := (*h.KVSnaps)
	n := len(prompt)
	switch {
	case adapterID != 0:
		st.decline("adapter")
		return
	case n < h.snapMinPrompt():
		st.decline("short")
		return
	case acq.Reuse > 0 && acq.SnapRows == 0:
		st.decline("not cold") // rows from a continuation or a copy: not rows this prefill wrote
		return
	}
	cm := h.KVClassMap()
	cold := cm.ColdClasses(n)
	if len(cold) == 0 || cold[0].Class == kvmgr.ClassUnknown {
		st.decline("class")
		return
	}
	slack := sn.KVSnapshotRollback()
	if slack == 0 {
		slack = kvmgr.RollbackFloor
	}
	root := h.KVRootIn(0, sc.Namespace)
	var replaced *kvSnap
	recurrent := h.hasRecurrentState()
	switch x := acq.snapFrom; {
	case x != nil && !recurrent && b <= x.h.Rows()+kvmgr.RollbackFloor:
		// This request restored its prefix from x, so the family is held: a snapshot of this prompt would hold the same prefix again,
		// for the rows only this prompt has. Only a prompt that runs on past x's rows by more than a rollback replaces it.
		st.decline("held")
		return
	case x != nil && recurrent && h.KVSnapHolds(x, b):
		st.decline("held")
		return
	case x != nil && x.h.Rows() < b && !x.preamble:
		replaced = x
	case x != nil:
		// A recurrent snapshot longer than b: a different point, kept beside the new one.
	default:
		if c := st.covering(root, prompt, b, h.KVSnapCoverSlack(slack)); c != nil {
			if h.KVSnapHolds(c, b) {
				st.decline("held")
				return
			}
			if c.h.Rows() < b && !c.preamble && !(recurrent && acq.MsgRows > 0 && c.h.Rows() == acq.MsgRows) {
				replaced = c // a longer prompt of the same prefix: take it, then drop the shorter
			}
		}
	}
	need := sn.KVSnapshotBytes(b)
	if need <= 0 {
		st.decline("backend")
		return
	}
	if need > st.budget {
		st.decline("budget")
		return
	}
	if replaced != nil {
		// The shorter snapshot of the same prefix goes first, so the budget is never exceeded, not even while the longer one is taken.
		st.drop(replaced)
		replaced = nil
	}
	if !h.KVSnapReserve(st, need) {
		st.decline("budget")
		return
	}
	t0 := time.Now()
	snap, err := sn.SnapshotKV(slot, b)
	d := time.Since(t0)
	if err != nil {
		st.Release(need)
		st.mu.Lock()
		if dec, ok := errors.AsType[*KVCopyDecline](err); ok {
			st.st.Declined["backend"]++
			st.st.LastDecline = dec.Reason
		} else {
			st.st.Declined["error"]++
			st.st.LastDecline = err.Error()
		}
		st.mu.Unlock()
		return
	}
	x := &kvSnap{root: root, ids: slices.Clone(prompt[:b]), classes: clipRuns(cold, b), h: snap, preamble: recurrent && acq.PreambleRows == b && b > 0}
	st.add(x, replaced, d, need)
	used := st.Stats().Used
	fmt.Fprintf(h.stderr(), "kv snapshot: took %d rows (%.0f MiB host) from slot %d in %.1f ms; %.0f of %.0f MiB held\n",
		b, float64(snap.Bytes())/(1<<20), slot, float64(d.Microseconds())/1000, float64(used)/(1<<20), float64(st.budget)/(1<<20))
}

// kvSnapReserve charges need bytes for a snapshot about to be taken. A demoted copy goes before a held snapshot: it serves one conversation
// that may never return, a snapshot any prompt with its prefix.
func (h *Host) KVSnapReserve(st *KVSnapStore, need int64) bool {
	if kv := h.resKV(); kv.KVDemoterAvailable() && h.KVDemote {
		return kv.ReserveKVHost(need)
	}
	return st.Reserve(need, true)
}

func clipRuns(rs []kvmgr.ClassRun, rows int) []kvmgr.ClassRun {
	var out []kvmgr.ClassRun
	for _, r := range rs {
		if r.Start >= rows {
			break
		}
		r.Len = min(r.Len, rows-r.Start)
		out = append(out, r)
	}
	return out
}

func classesEqual(a, b []kvmgr.ClassRun) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestAddAndDoubleDrop inserts a synthetic snapshot and calls drop twice to verify idempotent eviction (used by concurrency_bounds_test).
func (s *KVSnapStore) TestAddAndDoubleDrop(h resident.KVSnapshot) int64 {
	x := &kvSnap{h: h}
	s.list = append(s.list, x)
	s.used += h.Bytes()
	s.drop(x)
	s.drop(x)
	return s.used
}

// BestMatchLen returns the longest matching snapshot prefix length for (root, prompt), or 0 if none.
func (s *KVSnapStore) BestMatchLen(root kvmgr.Root, prompt []int) int {
	x, lcp := s.best(root, prompt)
	if x == nil {
		return 0
	}
	return lcp
}

// OverBudget reports whether s's used bytes currently exceed its budget (thread-safe).
func (s *KVSnapStore) OverBudget() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.used > s.budget
}

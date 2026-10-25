package decoder

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

// Prefix snapshots in host memory (docs/tasks/task-kv7-prefix-snapshot-2026-10.md). After a cold first prefill of a long prompt the
// resident's KV for the prompt's full blocks is copied to pinned host memory, keyed by the token ids and the root KV2 keys on
// (namespace, model identity, adapter, KV precision). A later request whose prompt starts with those ids restores the rows into its
// slot by device copy and prefills only the rest. It keeps a prefix that a one-slot card would overwrite when the next session
// starts. The rows are prefill-class rows copied before any decode step, so a restored prefix is the prefix computed
// (task-kv-architecture §5.6, property 2).

// stderrKV is where a snapshot taken or restored says so; a test points it elsewhere.
var stderrKV io.Writer = os.Stderr

// kvSnapMinPrompt is the shortest prompt whose prefix is worth a snapshot: under it the prefill is cheap and the pinned copy,
// and the host memory it holds, are not. A var so an in-package test can lower it.
var kvSnapMinPrompt = 2048

// kvSnapMinShared is the shortest prefix, in ids, two prompts must share for a held snapshot's failure to serve one to count as
// a first prompt that does not qualify.
const kvSnapMinShared = 512

// ResidentKVSnapshot is one prefix's KV held in host memory by a resident. Release frees it; it is idempotent.
type ResidentKVSnapshot interface {
	// Rows is how many leading positions it holds, a whole number of blocks.
	Rows() int
	// Bytes is the host memory it occupies.
	Bytes() int64
	// Serves reports whether RestoreKV can restore its first end rows into a slot: false when a windowed layer's rows behind the
	// snapshot's end, which the window of position end needs, are not among those it holds, or when the resident's geometry has
	// changed since it was taken.
	Serves(end int) bool
	Release()
}

// ResidentKVSnapshotter is the optional capability of a resident to hold a slot's KV rows in host memory and put them back. Like
// ResidentKVCopier it works on today's slot storage: a windowed layer's rows are those its window and the rollback need, and the
// restore sets the slot's ring base as a copy does.
type ResidentKVSnapshotter interface {
	// SnapshotKV copies rows [0, end) of slot into pinned host memory, end a whole number of blocks; for a windowed layer, the rows
	// [end-(window-1)-rollback, end) its slot holds. A *KVCopyDecline, before anything is allocated, refuses what it cannot hold.
	SnapshotKV(slot, end int) (ResidentKVSnapshot, error)
	// RestoreKV writes the first end rows of snap into slot, so that the slot continues at position end. A *KVCopyDecline, before
	// anything is written, refuses what it cannot serve.
	RestoreKV(snap ResidentKVSnapshot, slot, end int) error
	// KVSnapshotBytes is the host memory SnapshotKV(_, end) would hold, 0 when it would decline for a reason that needs no slot.
	KVSnapshotBytes(end int) int64
	// KVSnapshotRollback is how many positions before a snapshot's end a windowed layer still serves a restore; 0 with no windowed layer.
	KVSnapshotRollback() int
}

// ResidentRecurrentSnapshots is the optional capability of a ResidentKVSnapshotter to carry a recurrent model's state with its rows: a snapshot
// to end holds the state as it stood after position end-1, so it serves a restore to exactly end and to no shorter prefix.
type ResidentRecurrentSnapshots interface {
	KVSnapshotRecurrent() bool
}

// KeepsRecurrentSnapshots reports whether this model's prefix snapshots carry its recurrent state: it has recurrent state, and its resident's
// snapshots hold it (ResidentRecurrentSnapshots). A route computes KVScope.Keep only for such a model.
func (m *Model) KeepsRecurrentSnapshots() bool {
	return m.hasRecurrentState() && m.recurrentSnapshots()
}

// recurrentSnapshots reports whether the snapshotter carries recurrent state.
func (m *Model) recurrentSnapshots() bool {
	rs, ok := m.snapshotter().(ResidentRecurrentSnapshots)
	return ok && rs.KVSnapshotRecurrent()
}

// snapshotsBlocked reports whether this model's state is outside what a snapshot holds: recurrent state its snapshotter does not carry.
func (m *Model) snapshotsBlocked() bool { return m.hasRecurrentState() && !m.recurrentSnapshots() }

// ResidentPrefillMarked is the optional capability of a resident to tell its caller when a batched prefill has just written a given position,
// between two of its passes: the one moment a windowed layer's ring holds exactly what a shorter prompt would leave there, so a prefix snapshot
// can be taken in the middle of a cold prefill (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md).
type ResidentPrefillMarked interface {
	Prefiller
	// PrefillMarkAlign is the width of the passes of a batched prefill, in positions: a mark can fire only at startPos plus a multiple of it.
	PrefillMarkAlign() int
	// PrefillLastMarked is PrefillLast that calls at(mark) on the caller's goroutine after the pass that ends exactly at position mark, with no
	// pass running. A mark that is not at a pass boundary of this prefill (a single pass, a start that is not aligned to it, the passes cut
	// differently) calls nothing. at must not fail the prefill; a snapshot it cannot take is the caller's to count.
	PrefillLastMarked(ctx context.Context, embeddings [][]float32, startPos, mark int, at func(pos int)) ([]float32, error)
}

// KVSnapshotStats is a model's host snapshots, for /admin/status and the request log.
type KVSnapshotStats struct {
	Budget        int64 `json:"budget_bytes"` // host bytes allowed
	Used          int64 `json:"used_bytes"`   // and held
	Held          int   `json:"held"`
	Taken         int   `json:"taken"`    // snapshots taken
	Restored      int   `json:"restored"` // requests that restored one
	RestoredRows  int64 `json:"restored_rows"`
	Evicted       int   `json:"evicted"`        // snapshots dropped for room or replaced by a longer one
	EvictedUnused int   `json:"evicted_unused"` // of those, ones never restored
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
	h        ResidentKVSnapshot
	restores int
	last     uint64
	created  time.Time
}

// kvSnapStore is a model's held snapshots under a host byte budget, most recently used first.
type kvSnapStore struct {
	mu     sync.Mutex
	budget int64
	used   int64
	tick   uint64
	list   []*kvSnap
	st     KVSnapshotStats
}

// newKVSnapStore is the store for a budget in bytes, nil (no snapshots, nothing held) for a budget that is not positive.
func newKVSnapStore(budget int64) *kvSnapStore {
	if budget <= 0 {
		return nil
	}
	return &kvSnapStore{budget: budget, st: KVSnapshotStats{Declined: map[string]int{}}}
}

func (s *kvSnapStore) decline(reason string) {
	s.mu.Lock()
	s.st.Declined[reason]++
	s.mu.Unlock()
}

// best is the held snapshot under root that shares the most leading ids with prompt, and that count; nil with none shared.
func (s *kvSnapStore) best(root kvmgr.Root, prompt []int) (*kvSnap, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out *kvSnap
	lcp := 0
	for _, x := range s.list {
		if x.root != root {
			continue
		}
		if n := commonPrefixLen(x.ids, prompt); n > lcp {
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
func (s *kvSnapStore) candidates(root kvmgr.Root, prompt []int) []snapMatch {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []snapMatch
	for _, x := range s.list {
		if x.root != root {
			continue
		}
		if n := commonPrefixLen(x.ids, prompt); n >= kvmgr.BlockRows {
			out = append(out, snapMatch{x, n})
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		return min(out[i].lcp, out[i].x.h.Rows()) > min(out[j].lcp, out[j].x.h.Rows())
	})
	return out
}

// covering is the held snapshot under root that already serves a prompt whose full blocks end at b: it shares all but at most
// slack of min(its rows, b) leading ids with it, and its window can serve that share. Taking another from that prompt would hold the same
// prefix twice.
func (s *kvSnapStore) covering(root kvmgr.Root, prompt []int, b, slack int) *kvSnap {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, x := range s.list {
		if x.root != root {
			continue
		}
		lcp := commonPrefixLen(x.ids, prompt)
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

// use marks x as just used and counts a restore of rows rows.
func (s *kvSnapStore) use(x *kvSnap, rows int, d time.Duration) {
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

func (s *kvSnapStore) sortLocked() {
	for i := 1; i < len(s.list); i++ {
		for j := i; j > 0 && s.list[j].last > s.list[j-1].last; j-- {
			s.list[j], s.list[j-1] = s.list[j-1], s.list[j]
		}
	}
}

// removeLocked drops x, releasing its host memory.
func (s *kvSnapStore) removeLocked(x *kvSnap) {
	for i, y := range s.list {
		if y == x {
			s.list = append(s.list[:i], s.list[i+1:]...)
			break
		}
	}
	s.used -= x.h.Bytes()
	x.h.Release()
	s.st.Evicted++
	if x.restores == 0 {
		s.st.EvictedUnused++
	}
}

// room makes need bytes free by dropping the least recently used snapshots other than keep, and reports whether it could.
func (s *kvSnapStore) room(need int64, keep *kvSnap) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if need > s.budget {
		return false
	}
	for s.used+need > s.budget {
		var victim *kvSnap
		for i := len(s.list) - 1; i >= 0; i-- {
			if s.list[i] != keep {
				victim = s.list[i]
				break
			}
		}
		if victim == nil {
			return s.used+need <= s.budget
		}
		s.removeLocked(victim)
	}
	return true
}

// drop removes x and frees its host memory.
func (s *kvSnapStore) drop(x *kvSnap) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removeLocked(x)
}

// add holds x (just taken, its bytes already allocated) and drops replaced, if any.
func (s *kvSnapStore) add(x *kvSnap, replaced *kvSnap, d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tick++
	x.last, x.created = s.tick, time.Now()
	s.list = append([]*kvSnap{x}, s.list...)
	s.used += x.h.Bytes()
	s.st.Taken++
	s.st.TakeNanos += int64(d)
	if replaced != nil {
		s.removeLocked(replaced)
	}
}

// releaseAll frees every held snapshot; the model is going away or its geometry has changed.
func (s *kvSnapStore) releaseAll() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.list) > 0 {
		s.removeLocked(s.list[0])
	}
}

func (s *kvSnapStore) stats() KVSnapshotStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.st
	out.Budget, out.Used, out.Held = s.budget, s.used, len(s.list)
	out.Declined = make(map[string]int, len(s.st.Declined))
	for k, v := range s.st.Declined {
		out.Declined[k] = v
	}
	for _, x := range s.list {
		out.Snapshots = append(out.Snapshots, KVSnapshotInfo{Rows: x.h.Rows(), Bytes: x.h.Bytes(), Restores: x.restores, AgeSecs: int64(time.Since(x.created).Seconds())})
	}
	return out
}

// KVSnapshotStats reports the model's host snapshots: the budget, what is held, and what has been taken, restored, evicted and
// refused. ok is false when the model keeps none (a budget of 0, or no resident that can).
func (m *Model) KVSnapshotStats() (KVSnapshotStats, bool) {
	if m.kvSnaps == nil {
		return KVSnapshotStats{}, false
	}
	return m.kvSnaps.stats(), true
}

// KVSnapshotsSupported reports whether this model keeps prefix snapshots: a budget, and a resident that can hold them.
func (m *Model) KVSnapshotsSupported() bool { return m.snapshotter() != nil }

// KVHostBytes is the host memory budget for prefix snapshots this model was loaded with (Options.KVHostBytes), 0 when off.
func (m *Model) KVHostBytes() int64 {
	if m.kvSnaps == nil {
		return 0
	}
	return m.kvSnaps.budget
}

// snapshotter is the resident's snapshot capability, nil when snapshots are off for this model or the resident has none.
func (m *Model) snapshotter() ResidentKVSnapshotter {
	if m.kvSnaps == nil || m.resident == nil {
		return nil
	}
	sn, _ := m.resident.(ResidentKVSnapshotter)
	return sn
}

// residentRestoreSnapshot restores a held snapshot's rows into the bound slot when that serves more of prompt than the slot's own
// rows do (acq.reuse, after any copy from another slot). Nothing happens under an Off scope or Options.KVShareOff, with an
// adapter, for a recurrent family whose snapshots do not carry its state, or without a snapshotter. A recurrent snapshot restores only
// to its exact end, and declines when the rest of the prompt would be a pass under the resident's minimum. The rows come back as prefill-class rows of the same ids, so the
// requester's cold class must be the snapshot's and a pass of at least the fast minimum must remain (kvmgr.ClassMap.CapHit). A
// refusal before anything is written leaves acq as it was; a failure after leaves the slot unknown.
func (m *Model) residentRestoreSnapshot(prompt []int, lora *loraRuntime, sc KVScope, acq *kvAcquired) {
	sn := m.snapshotter()
	if sn == nil || sc.Off || m.kvShareOff || lora != nil || m.residentReuseDisabled() || m.snapshotsBlocked() || acq.slot < 0 {
		return
	}
	st := m.kvSnaps
	root := m.kvRootIn(nil, sc.Namespace)
	n, cm := len(prompt), m.kvClassMap()
	recurrent := m.hasRecurrentState()
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
		if cq <= acq.reuse {
			continue // the slot's own rows, or a copy from another slot, serve as much
		}
		if recurrent && cq != c.x.h.Rows() {
			continue // the state is the one after the snapshot's last row: no shorter prefix
		}
		if recurrent && n-cq < m.prefillTailMinRows() {
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
	a := m.kvSlots()
	t0 := time.Now()
	err := sn.RestoreKV(x.h, acq.slot, q)
	d := time.Since(t0)
	if err != nil {
		var dec *KVCopyDecline
		if errors.As(err, &dec) {
			st.mu.Lock()
			st.st.Declined["backend"]++
			st.st.LastDecline = dec.Reason
			st.mu.Unlock()
			acq.declined = dec.Reason
			return
		}
		st.mu.Lock()
		st.st.Declined["error"]++
		st.st.LastDecline = err.Error()
		st.mu.Unlock()
		acq.declined = err.Error()
		acq.reuse = 0 // the restore may have written some rows; the slot's own are no longer trusted
		m.residentForgetIDs()
		return
	}
	a.Restored(acq.slot, root, x.ids[:q], x.classes)
	s := a.Seq(acq.slot)
	m.resIDs, m.resIDsLora, m.resImgBlocks = s.Tokens(), nil, nil
	m.kvSetLora(acq.slot, nil)
	acq.reuse, acq.snapRows, acq.snapDur, acq.snapFrom, acq.shared, acq.src = q, q, d, x, 0, -1
	st.use(x, q, d)
	fmt.Fprintf(stderrKV, "kv snapshot: restored %d rows of %d into slot %d in %.1f ms\n", q, x.h.Rows(), acq.slot, float64(d.Microseconds())/1000)
}

// residentSnapshotPrompt takes a snapshot of prompt's full blocks from slot, which has just finished prefilling it, before any
// decode step. Admission is first touch: any prompt of at least kvSnapMinPrompt ids whose rows are all prefill-class (the prefill
// was cold, or began from a restore) and that no held snapshot already serves within the window's rollback. A longer prompt
// replaces a shorter snapshot of the same prefix. The budget is made by dropping the least recently used.
func (m *Model) residentSnapshotPrompt(slot int, prompt []int, lora *loraRuntime, sc KVScope, acq kvAcquired) {
	// One snapshot per prompt. When the prompt's tail after the stable section is longer than the rollback the boundary snapshot is the one that can
	// serve a later session, and the end-of-prompt snapshot (whose ring no longer holds the shared prefix) would replace or sit beside it for nothing:
	// skip it when the boundary one was taken, declined or restored, and fall back to it only when the mark never fired.
	if m.hasRecurrentState() {
		return // a recurrent model's state at the end of the prompt is the slot's own; its snapshot is the mark's (kvBoundaryPlan)
	}
	if q, ok := m.kvBoundaryPlan(prompt, sc, lora); ok {
		if acq.markFired || acq.snapRows > 0 {
			return
		}
		if st := m.kvSnaps; st != nil && st.covering(m.kvRootIn(nil, sc.Namespace), prompt, q, m.kvSnapSlack()) != nil {
			st.decline("held")
			return
		}
	}
	m.residentSnapshotRows(slot, prompt, len(prompt)/kvmgr.BlockRows*kvmgr.BlockRows, lora, sc, acq)
}

// kvSnapSlack is how many positions behind a snapshot's end a windowed layer still serves a restore.
func (m *Model) kvSnapSlack() int {
	if sn := m.snapshotter(); sn != nil {
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
func (m *Model) kvBoundaryPlan(prompt []int, sc KVScope, lora *loraRuntime) (q int, ok bool) {
	sn := m.snapshotter()
	mp, hasMark := m.resident.(ResidentPrefillMarked)
	if sn == nil || !hasMark || lora != nil || sc.Off || m.kvShareOff || m.residentReuseDisabled() || m.snapshotsBlocked() {
		return 0, false
	}
	n, align := len(prompt), mp.PrefillMarkAlign()
	if m.hasRecurrentState() {
		// A recurrent model's snapshot is taken where the next turn's prompt parts from this one (KVScope.Keep), on a pass boundary: the
		// state exists only between passes, and the next turn re-prefills from there on the same pass grid as a cold prefill.
		if align <= 0 || n < kvSnapMinPrompt || sc.Keep <= 0 || sc.Keep >= n {
			return 0, false
		}
		q = sc.Keep / align * align
		if q < kvSnapMinShared || q%kvmgr.BlockRows != 0 {
			return 0, false
		}
		return q, true
	}
	if sc.Boundary <= 0 {
		return 0, false
	}
	if align <= 0 || n < kvSnapMinPrompt || sc.Boundary >= n || n-sc.Boundary <= m.kvSnapSlack() {
		return 0, false
	}
	q = sc.Boundary / align * align
	if q < kvSnapMinShared || q%kvmgr.BlockRows != 0 {
		return 0, false
	}
	return q, true
}

// kvPrefillMark returns the mark a cold prefill of prompt from position from should carry: the boundary position and the function that takes the
// snapshot there. nil when no boundary snapshot applies, when from does not start on a pass boundary the mark could meet, or when a held snapshot
// already serves the prefix (this request restored it, or another one covers it). A recurrent model's restored request still marks its own
// Keep past the restored point, so the kept state moves to each new question instead of staying at the first one
// (docs/measurements/b35-keep-advance-2026-10/README.md); admission then replaces the snapshot it restored from (residentSnapshotRows).
func (m *Model) kvPrefillMark(slot int, prompt []int, from int, lora *loraRuntime, sc KVScope, acq *kvAcquired) *prefillMark {
	q, ok := m.kvBoundaryPlan(prompt, sc, lora)
	if !ok || (acq.snapRows > 0 && !m.hasRecurrentState()) || q <= from {
		return nil
	}
	mp := m.resident.(ResidentPrefillMarked)
	if (q-from)%mp.PrefillMarkAlign() != 0 {
		return nil // the passes of this prefill do not end at q
	}
	if st := m.kvSnaps; st != nil {
		if c := st.covering(m.kvRootIn(nil, sc.Namespace), prompt, q, m.kvSnapCoverSlack(m.kvSnapSlack())); c != nil && (!m.hasRecurrentState() || m.kvSnapHolds(c, q)) {
			st.decline("held")
			return nil
		}
	}
	return &prefillMark{q: q, at: func(pos int) {
		acq.markFired = true
		m.residentSnapshotRows(slot, prompt, pos, lora, sc, *acq)
	}}
}

// prefillMark is a position at which a batched prefill is to call back between two passes.
type prefillMark struct {
	q  int
	at func(pos int)
}

// kvSnapHolds reports whether held snapshot c already serves a snapshot of the first b rows, the rule kvPrefillMark and
// residentSnapshotRows share. Attention KV: c reaches b. A recurrent snapshot serves only its own end, so it holds b only when it ends
// exactly there; a shorter one is replaced and a longer one is a different point, kept beside the new one.
func (m *Model) kvSnapHolds(c *kvSnap, b int) bool {
	if m.hasRecurrentState() {
		return c.h.Rows() == b
	}
	return c.h.Rows() >= b
}

// kvSnapCoverSlack is the rollback a covering search allows: slack for attention KV, 0 for a recurrent model, whose snapshot shares
// nothing with a prompt that leaves its prefix before its end.
func (m *Model) kvSnapCoverSlack(slack int) int {
	if m.hasRecurrentState() {
		return 0
	}
	return slack
}

// residentSnapshotRows takes a snapshot of prompt's first b rows from slot, which holds them: at the end of the prefill (b the full blocks of the
// prompt, residentSnapshotPrompt) or between two passes of it (b the boundary, kvPrefillMark). It is the same admission either way.
func (m *Model) residentSnapshotRows(slot int, prompt []int, b int, lora *loraRuntime, sc KVScope, acq kvAcquired) {
	sn := m.snapshotter()
	if sn == nil || sc.Off || m.kvShareOff || m.residentReuseDisabled() || m.snapshotsBlocked() {
		return
	}
	st := m.kvSnaps
	n := len(prompt)
	switch {
	case lora != nil:
		st.decline("adapter")
		return
	case n < kvSnapMinPrompt:
		st.decline("short")
		return
	case acq.reuse > 0 && acq.snapRows == 0:
		st.decline("not cold") // rows from a continuation or a copy: not rows this prefill wrote
		return
	}
	cm := m.kvClassMap()
	cold := cm.ColdClasses(n)
	if len(cold) == 0 || cold[0].Class == kvmgr.ClassUnknown {
		st.decline("class")
		return
	}
	slack := sn.KVSnapshotRollback()
	if slack == 0 {
		slack = kvmgr.RollbackFloor
	}
	root := m.kvRootIn(nil, sc.Namespace)
	var replaced *kvSnap
	recurrent := m.hasRecurrentState()
	switch x := acq.snapFrom; {
	case x != nil && !recurrent && b <= x.h.Rows()+kvmgr.RollbackFloor:
		// This request restored its prefix from x, so the family is held: a snapshot of this prompt would hold the same prefix again,
		// for the rows only this prompt has. Only a prompt that runs on past x's rows by more than a rollback replaces it.
		st.decline("held")
		return
	case x != nil && recurrent && m.kvSnapHolds(x, b):
		st.decline("held")
		return
	case x != nil && x.h.Rows() < b:
		replaced = x
	case x != nil:
		// A recurrent snapshot longer than b: a different point, kept beside the new one.
	default:
		if c := st.covering(root, prompt, b, m.kvSnapCoverSlack(slack)); c != nil {
			if m.kvSnapHolds(c, b) {
				st.decline("held")
				return
			}
			if c.h.Rows() < b {
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
	if !st.room(need, nil) {
		st.decline("budget")
		return
	}
	t0 := time.Now()
	h, err := sn.SnapshotKV(slot, b)
	d := time.Since(t0)
	if err != nil {
		var dec *KVCopyDecline
		st.mu.Lock()
		if errors.As(err, &dec) {
			st.st.Declined["backend"]++
			st.st.LastDecline = dec.Reason
		} else {
			st.st.Declined["error"]++
			st.st.LastDecline = err.Error()
		}
		st.mu.Unlock()
		return
	}
	x := &kvSnap{root: root, ids: append([]int(nil), prompt[:b]...), classes: clipRuns(cold, b), h: h}
	st.add(x, replaced, d)
	fmt.Fprintf(stderrKV, "kv snapshot: took %d rows (%.0f MiB host) from slot %d in %.1f ms; %.0f of %.0f MiB held\n",
		b, float64(h.Bytes())/(1<<20), slot, float64(d.Microseconds())/1000, float64(st.used)/(1<<20), float64(st.budget)/(1<<20))
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

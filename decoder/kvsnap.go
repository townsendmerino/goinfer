package decoder

import (
	"errors"
	"fmt"
	"io"
	"os"
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

// covering is the held snapshot under root that already serves a prompt whose full blocks end at b: it shares all but at most
// slack of min(its rows, b) leading ids with it. Taking another from that prompt would hold the same prefix twice.
func (s *kvSnapStore) covering(root kvmgr.Root, prompt []int, b, slack int) *kvSnap {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, x := range s.list {
		if x.root != root {
			continue
		}
		if commonPrefixLen(x.ids, prompt) >= min(x.h.Rows(), b)-slack {
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
// adapter, for a recurrent family, or without a snapshotter. The rows come back as prefill-class rows of the same ids, so the
// requester's cold class must be the snapshot's and a pass of at least the fast minimum must remain (kvmgr.ClassMap.CapHit). A
// refusal before anything is written leaves acq as it was; a failure after leaves the slot unknown.
func (m *Model) residentRestoreSnapshot(prompt []int, lora *loraRuntime, sc KVScope, acq *kvAcquired) {
	sn := m.snapshotter()
	if sn == nil || sc.Off || m.kvShareOff || lora != nil || m.residentReuseDisabled() || m.hasRecurrentState() || acq.slot < 0 {
		return
	}
	st := m.kvSnaps
	root := m.kvRootIn(nil, sc.Namespace)
	x, lcp := st.best(root, prompt)
	if x == nil || lcp < kvmgr.BlockRows {
		return
	}
	n, cm := len(prompt), m.kvClassMap()
	q := min(lcp, x.h.Rows()) / kvmgr.BlockRows * kvmgr.BlockRows
	q = min(q, (n-1)/kvmgr.BlockRows*kvmgr.BlockRows) // a token is left to compute, for the first logits
	q = cm.CapHit(n, q)
	if q <= acq.reuse {
		return // the slot's own rows, or a copy from another slot, serve as much
	}
	if !x.h.Serves(q) {
		if lcp >= kvSnapMinShared {
			st.mu.Lock()
			st.st.Unqualified++
			st.mu.Unlock()
		}
		st.decline("window")
		return
	}
	if !classesEqual(clipRuns(x.classes, q), clipRuns(cm.ColdClasses(n), q)) {
		st.decline("class")
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
	sn := m.snapshotter()
	if sn == nil || sc.Off || m.kvShareOff || m.residentReuseDisabled() || m.hasRecurrentState() {
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
	b := n / kvmgr.BlockRows * kvmgr.BlockRows
	slack := sn.KVSnapshotRollback()
	if slack == 0 {
		slack = kvmgr.RollbackFloor
	}
	root := m.kvRootIn(nil, sc.Namespace)
	var replaced *kvSnap
	switch x := acq.snapFrom; {
	case x != nil && b <= x.h.Rows()+kvmgr.RollbackFloor:
		// This request restored its prefix from x, so the family is held: a snapshot of this prompt would hold the same prefix again,
		// for the rows only this prompt has. Only a prompt that runs on past x's rows by more than a rollback replaces it.
		st.decline("held")
		return
	case x != nil:
		replaced = x
	default:
		if c := st.covering(root, prompt, b, slack); c != nil {
			if c.h.Rows() >= b {
				st.decline("held")
				return
			}
			replaced = c // a longer prompt of the same prefix: take it, then drop the shorter
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

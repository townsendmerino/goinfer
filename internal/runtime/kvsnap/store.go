package kvsnap

import (
	"cmp"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// MinShared is the shortest prefix, in token ids, two prompts must share for a held snapshot's window refusal to count as unqualified.
const MinShared = 512

// Entry is one held prefix snapshot in host memory.
type Entry struct {
	Root     kvmgr.Root
	IDs      []int
	Classes  []kvmgr.ClassRun
	Handle   resident.KVSnapshot
	Preamble bool

	restores int
	last     uint64
	created  time.Time
}

// Rows returns how many leading positions the snapshot holds.
func (e *Entry) Rows() int { return e.Handle.Rows() }

// Bytes returns the host bytes the snapshot occupies.
func (e *Entry) Bytes() int64 { return e.Handle.Bytes() }

// Serves reports whether the snapshot can restore end leading positions.
func (e *Entry) Serves(end int) bool { return e.Handle.Serves(end) }

// Match pairs a held snapshot with the number of leading token ids it shares with a prompt.
type Match struct {
	Entry *Entry
	LCP   int
}

// Store manages held prefix snapshots under a host byte budget, ordered most recently used first.
// It implements resident.KVHostBudget so demoted slot copies share the same byte ceiling.
type Store struct {
	mu     sync.Mutex
	budget int64
	used   int64
	tick   uint64
	list   []*Entry
	st     resident.KVSnapshotStats
}

var _ resident.KVHostBudget = (*Store)(nil)

// NewStore returns a Store for budget bytes, or nil when budget <= 0.
func NewStore(budget int64) *Store {
	if budget <= 0 {
		return nil
	}
	return &Store{budget: budget, st: resident.KVSnapshotStats{Declined: map[string]int{}}}
}

// Budget returns the configured host byte ceiling, or 0 when s is nil.
func (s *Store) Budget() int64 {
	if s == nil {
		return 0
	}
	return s.budget
}

// Decline increments the count for reason.
func (s *Store) Decline(reason string) {
	s.mu.Lock()
	s.st.Declined[reason]++
	s.mu.Unlock()
}

// DeclineDetail increments kind and records detail as the latest refusal message.
func (s *Store) DeclineDetail(kind, detail string) {
	s.mu.Lock()
	s.st.Declined[kind]++
	s.st.LastDecline = detail
	s.mu.Unlock()
}

// Best returns the held snapshot under root that shares the most leading ids with prompt, and that count (nil, 0 when none).
func (s *Store) Best(root kvmgr.Root, prompt []int) (*Entry, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out *Entry
	lcp := 0
	for _, x := range s.list {
		if x.Root != root {
			continue
		}
		if n := kvmgr.CommonPrefixLen(x.IDs, prompt); n > lcp {
			out, lcp = x, n
		}
	}
	return out, lcp
}

// Candidates returns held snapshots under root that share at least one block of leading ids with prompt, ordered by restorable rows descending.
func (s *Store) Candidates(root kvmgr.Root, prompt []int) []Match {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Match
	for _, x := range s.list {
		if x.Root != root {
			continue
		}
		if n := kvmgr.CommonPrefixLen(x.IDs, prompt); n >= kvmgr.BlockRows {
			out = append(out, Match{Entry: x, LCP: n})
		}
	}
	slices.SortStableFunc(out, func(a, b Match) int {
		return cmp.Compare(min(b.LCP, b.Entry.Rows()), min(a.LCP, a.Entry.Rows()))
	})
	return out
}

// Covering returns a held snapshot under root that already serves a prompt up to b within slack positions, or nil.
func (s *Store) Covering(root kvmgr.Root, prompt []int, b, slack int) *Entry {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, x := range s.list {
		if x.Root != root {
			continue
		}
		lcp := kvmgr.CommonPrefixLen(x.IDs, prompt)
		if lcp < min(x.Rows(), b)-slack {
			continue
		}
		if q := min(lcp, x.Rows(), b) / kvmgr.BlockRows * kvmgr.BlockRows; q > 0 && x.Serves(q) {
			return x
		}
	}
	return nil
}

// Holds reports whether x is still present in s.
func (s *Store) Holds(x *Entry) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Contains(s.list, x)
}

// Use marks x as just restored for rows positions in duration d.
func (s *Store) Use(x *Entry, rows int, d time.Duration) {
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

func (s *Store) sortLocked() {
	slices.SortFunc(s.list, func(a, b *Entry) int {
		return cmp.Compare(b.last, a.last)
	})
}

func (s *Store) removeLocked(x *Entry) {
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
	s.used -= x.Bytes()
	x.Handle.Release()
	s.st.Evicted++
	if x.restores == 0 {
		s.st.EvictedUnused++
	}
}

func (s *Store) roomLocked(need int64) bool {
	var droppable int64
	for _, x := range s.list {
		droppable += x.Bytes()
	}
	if s.used-droppable+need > s.budget {
		return false
	}
	for s.used+need > s.budget {
		s.removeLocked(s.list[len(s.list)-1])
	}
	return true
}

// Fits implements resident.KVHostBudget.
func (s *Store) Fits(n, copies int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	left := s.used - copies + n
	for _, x := range s.list {
		left -= x.Bytes()
	}
	return left <= s.budget
}

// Reserve implements resident.KVHostBudget.
func (s *Store) Reserve(n int64, dropSnapshots bool) bool {
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

// Release implements resident.KVHostBudget.
func (s *Store) Release(n int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.used -= n
}

// Kept implements resident.KVHostBudget.
func (s *Store) Kept() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.Demoted++
}

// Dropped implements resident.KVHostBudget.
func (s *Store) Dropped() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.DemoteDropped++
}

// Refused implements resident.KVHostBudget.
func (s *Store) Refused() { s.Decline("demote budget") }

// Promoted records blocks promoted back from host memory. Nil-safe.
func (s *Store) Promoted(blocks int) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.st.Promoted++
	s.st.PromotedBlocks += int64(blocks)
}

// Drop removes x from s and releases its host memory.
func (s *Store) Drop(x *Entry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removeLocked(x)
}

// Add inserts x at the front of s (accounting for reserved bytes already charged via Reserve) and drops replaced when non-nil.
func (s *Store) Add(x *Entry, replaced *Entry, d time.Duration, reserved int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tick++
	x.last, x.created = s.tick, time.Now()
	s.list = append([]*Entry{x}, s.list...)
	s.used += x.Bytes() - reserved
	s.st.Taken++
	s.st.TakeNanos += int64(d)
	if replaced != nil {
		s.removeLocked(replaced)
	}
}

// ReleaseAll frees every held snapshot in s. Nil-safe.
func (s *Store) ReleaseAll() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.list) > 0 {
		s.removeLocked(s.list[0])
	}
}

// Stats returns a copy of s's current snapshot statistics.
func (s *Store) Stats() resident.KVSnapshotStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.st
	out.Budget, out.Used, out.Held = s.budget, s.used, len(s.list)
	out.Declined = make(map[string]int, len(s.st.Declined))
	maps.Copy(out.Declined, s.st.Declined)
	for _, x := range s.list {
		out.Snapshots = append(out.Snapshots, resident.KVSnapshotInfo{
			Rows:     x.Rows(),
			Bytes:    x.Bytes(),
			Restores: x.restores,
			AgeSecs:  int64(time.Since(x.created).Seconds()),
		})
	}
	return out
}

// SelectRestore picks the best servable snapshot in s for prompt that beats currentReuse, recording decline counters when candidates fail window, class, or tail rules.
func (s *Store) SelectRestore(root kvmgr.Root, prompt []int, currentReuse, minTail int, cm kvmgr.ClassMap, recurrent bool) (*Entry, int) {
	cands := s.Candidates(root, prompt)
	if len(cands) == 0 {
		return nil, 0
	}
	n := len(prompt)
	var sawWindow, sawClass bool
	for _, c := range cands {
		cq := min(c.LCP, c.Entry.Rows()) / kvmgr.BlockRows * kvmgr.BlockRows
		cq = min(cq, (n-1)/kvmgr.BlockRows*kvmgr.BlockRows)
		cq = cm.CapHit(n, cq)
		if cq <= currentReuse {
			continue
		}
		if recurrent && cq != c.Entry.Rows() {
			continue
		}
		if recurrent && n-cq < minTail {
			s.Decline("short suffix")
			continue
		}
		if !c.Entry.Serves(cq) {
			if c.LCP >= MinShared {
				sawWindow = true
			}
			continue
		}
		if !ClassesEqual(ClipRuns(c.Entry.Classes, cq), ClipRuns(cm.ColdClasses(n), cq)) {
			sawClass = true
			continue
		}
		return c.Entry, cq
	}
	switch {
	case sawWindow:
		s.mu.Lock()
		s.st.Unqualified++
		s.mu.Unlock()
		s.Decline("window")
	case sawClass:
		s.Decline("class")
	}
	return nil, 0
}

// CheckAdmission decides whether taking a snapshot of prompt[:b] under root is admitted, returning any shorter snapshot to replace and false when already covered ("held").
func (s *Store) CheckAdmission(root kvmgr.Root, prompt []int, b int, from *Entry, msgRows, slack int, recurrent bool) (replaced *Entry, ok bool) {
	switch {
	case from != nil && !recurrent && b <= from.Rows()+kvmgr.RollbackFloor:
		s.Decline("held")
		return nil, false
	case from != nil && recurrent && HoldsTarget(from, b, recurrent):
		s.Decline("held")
		return nil, false
	case from != nil && from.Rows() < b && !from.Preamble:
		return from, true
	case from != nil:
		return nil, true
	default:
		if c := s.Covering(root, prompt, b, CoverSlack(slack, recurrent)); c != nil {
			if HoldsTarget(c, b, recurrent) {
				s.Decline("held")
				return nil, false
			}
			if c.Rows() < b && !c.Preamble && !(recurrent && msgRows > 0 && c.Rows() == msgRows) {
				return c, true
			}
		}
		return nil, true
	}
}

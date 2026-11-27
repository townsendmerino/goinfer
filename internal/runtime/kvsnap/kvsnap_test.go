package kvsnap

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

type fixedSizeSnap struct {
	rows     int
	bytes    int64
	released int
}

func (f *fixedSizeSnap) Rows() int {
	if f.rows > 0 {
		return f.rows
	}
	return 64
}
func (f *fixedSizeSnap) Bytes() int64    { return f.bytes }
func (f *fixedSizeSnap) Serves(int) bool { return true }
func (f *fixedSizeSnap) Release()        { f.released++ }

func TestKVSnapStore_reserveHoldsTheBudget(t *testing.T) {
	const n = 1 << 20
	st := NewStore(n)
	var over atomic.Int64
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 200000 {
				if st.Reserve(n, true) {
					st.mu.Lock()
					if st.used > st.budget {
						over.Add(1)
					}
					st.mu.Unlock()
					st.Release(n)
				}
			}
		})
	}
	wg.Wait()
	if c := over.Load(); c != 0 {
		t.Errorf("%d reservations charged past the budget", c)
	}
	if st.used != 0 {
		t.Errorf("%d bytes left charged", st.used)
	}
}

func TestKVSnapStore_noEvictionForWhatCannotFit(t *testing.T) {
	st := NewStore(100)
	st.Add(&Entry{Handle: &fixedSizeSnap{bytes: 40}}, nil, 0, 0)
	if !st.Reserve(40, false) {
		t.Fatal("the first reservation")
	}
	if st.Reserve(70, true) {
		t.Fatal("70 bytes fit beside 40 reserved under a budget of 100")
	}
	if len(st.list) != 1 || st.used != 80 {
		t.Errorf("after the refusal: %d snapshots held, %d bytes used; want the one snapshot, 80", len(st.list), st.used)
	}
	if !st.Reserve(60, true) {
		t.Fatal("60 bytes fit beside 40 reserved once the snapshot goes")
	}
	if len(st.list) != 0 || st.used != 100 {
		t.Errorf("after making room: %d snapshots held, %d bytes used", len(st.list), st.used)
	}
}

func TestKVSnapStore_removeLockedIdempotent(t *testing.T) {
	st := NewStore(1024)
	h := &fixedSizeSnap{rows: 4, bytes: 128}
	x := &Entry{Handle: h, IDs: []int{1, 2, 3, 4}}
	st.Add(x, nil, 0, 0)
	st.Drop(x)
	st.Drop(x)
	if st.used != 0 {
		t.Fatalf("used = %d, want 0", st.used)
	}
	if h.released != 1 {
		t.Fatalf("released = %d, want 1", h.released)
	}
}

func TestKVSnap_planAndRuns(t *testing.T) {
	q, ok := RecurrentCut(1200, 1196, 16)
	if !ok || q != 1152 {
		t.Fatalf("RecurrentCut(1200, 1196, 16) = (%d, %v), want (1152, true)", q, ok)
	}
	if !CutSegmentOK(256, 128, 16) || CutSegmentOK(130, 128, 16) {
		t.Fatal("CutSegmentOK segment tail bounds broken")
	}
	runs := []kvmgr.ClassRun{{Start: 0, Len: 128, Class: kvmgr.ClassFast}, {Start: 128, Len: 64, Class: kvmgr.ClassExact}}
	clipped := ClipRuns(runs, 96)
	if !ClassesEqual(clipped, []kvmgr.ClassRun{{Start: 0, Len: 96, Class: kvmgr.ClassFast}}) {
		t.Fatalf("ClipRuns = %+v", clipped)
	}
}

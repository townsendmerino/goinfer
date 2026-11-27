package kvmgr

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

func TestBlocksForAndWithinShare(t *testing.T) {
	for _, tc := range []struct{ pos, want int }{
		{-5, 0}, {0, 0}, {1, 1}, {63, 1}, {64, 1}, {65, 2}, {128, 2}, {129, 3},
	} {
		if got := BlocksFor(tc.pos); got != tc.want {
			t.Errorf("BlocksFor(%d) = %d, want %d", tc.pos, got, tc.want)
		}
	}
	if !WithinShare(3, 4, false) || WithinShare(4, 4, false) {
		t.Error("WithinShare non-CoW boundary wrong")
	}
	if !WithinShare(4, 4, true) || WithinShare(5, 4, true) {
		t.Error("WithinShare CoW boundary wrong")
	}
}

func TestBlockScanOrder(t *testing.T) {
	if got := BlockScanOrder(4, false, false); !slices.Equal(got, []uint32{1, 2, 3, 4}) {
		t.Errorf("ascending = %v", got)
	}
	if got := BlockScanOrder(5, true, false); !slices.Equal(got, []uint32{5, 1, 4, 2, 3}) {
		t.Errorf("scramble primary = %v, want [5 1 4 2 3]", got)
	}
	if got := BlockScanOrder(5, true, true); !slices.Equal(got, []uint32{1, 5, 2, 4, 3}) {
		t.Errorf("scramble reverse = %v, want [1 5 2 4 3]", got)
	}
}

func TestBlockPool_TakeRefUnrefAndCustomOrder(t *testing.T) {
	p := NewBlockPool(4, nil) // blocks 0 (null), 1, 2, 3
	if p.Blocks() != 4 || p.Free() != 3 || p.InUse() != 0 {
		t.Fatalf("initial state: blocks=%d free=%d inUse=%d", p.Blocks(), p.Free(), p.InUse())
	}
	b1, ok1 := p.Take()
	b2, ok2 := p.Take()
	if !ok1 || !ok2 || b1 != 1 || b2 != 2 || p.InUse() != 2 || p.Free() != 1 {
		t.Fatalf("takes = (%d,%v), (%d,%v), inUse=%d free=%d", b1, ok1, b2, ok2, p.InUse(), p.Free())
	}
	p.Ref(b1)
	if p.Refs(b1) != 2 {
		t.Fatalf("Refs(b1) = %d, want 2", p.Refs(b1))
	}
	p.Unref(0) // null block ignored
	p.Unref(b1)
	if p.Refs(b1) != 1 || p.InUse() != 2 {
		t.Fatalf("after first unref: refs=%d inUse=%d", p.Refs(b1), p.InUse())
	}
	p.Unref(b1)
	if p.Refs(b1) != 0 || p.InUse() != 1 || p.Free() != 2 {
		t.Fatalf("after final unref: refs=%d inUse=%d free=%d", p.Refs(b1), p.InUse(), p.Free())
	}
	// Custom order
	custom := NewBlockPool(4, []int32{3, 1, 2})
	var got []int32
	for {
		b, ok := custom.Take()
		if !ok {
			break
		}
		got = append(got, b)
	}
	if !slices.Equal(got, []int32{3, 1, 2}) {
		t.Errorf("custom order = %v, want [3 1 2]", got)
	}
}

func TestPickIdleAndBorrowerVictims(t *testing.T) {
	uses := []uint64{40, 10, 25, 10}
	eligible := []bool{true, true, false, true}
	if got := PickIdleVictim(4, 1, func(s int) bool { return eligible[s] }, func(s int) uint64 { return uses[s] }, false); got != 3 {
		t.Errorf("PickIdleVictim = %d, want 3 (slot 1 skipped, slot 3 oldest)", got)
	}
	if got := PickIdleVictim(4, 1, func(s int) bool { return eligible[s] }, func(s int) uint64 { return uses[s] }, true); got != 0 {
		t.Errorf("PickIdleVictim reverse = %d, want 0", got)
	}
	excess := []int{0, 2, 5, 3}
	v, most := PickBorrowerVictim(4, 2, func(int) bool { return true }, func(s int) int { return excess[s] })
	if v != 3 || most != 3 {
		t.Errorf("PickBorrowerVictim = (%d, %d), want (3, 3)", v, most)
	}
}

func TestRowRuns(t *testing.T) {
	ids := []int32{2, 3, 7, 0}
	type run struct{ p, n, row int }
	var runs []run
	unmapped, err := RowRuns(ids, 10, 150, func(p, n, row int) error {
		runs = append(runs, run{p, n, row})
		return nil
	})
	if err != nil || unmapped != -1 {
		t.Fatalf("RowRuns = (%d, %v)", unmapped, err)
	}
	// [10, 128) spans blocks 0->2 and 1->3 (consecutive!), starting at 2*64+10 = 138, length 118;
	// [128, 150) is in block 2->7, starting at 7*64 = 448, length 22.
	want := []run{{10, 118, 138}, {128, 22, 448}}
	if !slices.Equal(runs, want) {
		t.Errorf("runs = %v, want %v", runs, want)
	}
	unmapped, err = RowRuns(ids, 128, 200, func(int, int, int) error { return nil })
	if err != nil || unmapped != 192 {
		t.Errorf("unmapped = (%d, %v), want (192, nil)", unmapped, err)
	}
	sentinel := errors.New("boom")
	_, err = RowRuns(ids, 0, 64, func(int, int, int) error { return sentinel })
	if !errors.Is(err, sentinel) {
		t.Errorf("err = %v, want sentinel", err)
	}
}

func TestUploadPass_ValidationAndFinish(t *testing.T) {
	p := NewUploadPass(1)
	if err := p.CheckLayer("cuda", 0, 0, 0, 128, 0, true); err == nil || !strings.Contains(err.Error(), "slot 0 is bound") {
		t.Errorf("wrong slot err = %v", err)
	}
	if err := p.CheckLayer("cuda", 1, 0, 64, 128, 0, true); err == nil || !strings.Contains(err.Error(), "not position 0") {
		t.Errorf("nonzero full-attn base err = %v", err)
	}
	if err := p.CheckLayer("metal", 1, 1, 100, 128, 64, true); err == nil || !strings.Contains(err.Error(), "do not cover its 64-row window") {
		t.Errorf("insufficient window err = %v", err)
	}
	if err := p.CheckLayer("cuda", 1, 0, 0, 128, 0, true); err != nil {
		t.Fatalf("layer 0 check: %v", err)
	}
	p.MarkWrote(128)
	p.RecordLayer(0, 0, 128, 0)
	if err := p.CheckLayer("cuda", 1, 0, 0, 128, 0, true); err == nil || !strings.Contains(err.Error(), "uploaded twice") {
		t.Errorf("duplicate layer err = %v", err)
	}
	if err := p.CheckLayer("metal", 1, 1, 65, 128, 64, true); err != nil {
		t.Fatalf("wrapped window layer check: %v", err)
	}
	if _, _, err := p.Finish("cuda", false, 2, func(int) bool { return true }); err == nil || !strings.Contains(err.Error(), "layer 1 reached 0") {
		t.Errorf("incomplete finish err = %v", err)
	}
	p.RecordLayer(1, 65, 128, 64)
	end, floor, err := p.Finish("cuda", false, 2, func(int) bool { return true })
	if err != nil || end != 128 || floor != 128 {
		t.Errorf("Finish = (%d, %d, %v), want (128, 128, nil)", end, floor, err)
	}
}

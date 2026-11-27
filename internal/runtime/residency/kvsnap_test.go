package residency

import "testing"

// fixedSizeSnap is a held snapshot of a size, with no device behind it.
type fixedSizeSnap struct{ bytes int64 }

func (f fixedSizeSnap) Rows() int       { return 64 }
func (f fixedSizeSnap) Bytes() int64    { return f.bytes }
func (f fixedSizeSnap) Serves(int) bool { return true }
func (f fixedSizeSnap) Release()        {}

// TestKVSnapStore_noEvictionForWhatCannotFit: bytes reserved outside the list cannot be dropped, so a reservation that would not fit even with
// every snapshot gone drops none of them.
func TestKVSnapStore_noEvictionForWhatCannotFit(t *testing.T) {
	st := NewKVSnapStore(100)
	st.add(&kvSnap{h: fixedSizeSnap{40}}, nil, 0, 0)
	if !st.Reserve(40, false) { // a copy another caller holds
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

package decoder

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/kvmgr"
)

type fakeReusable map[int]int // slot -> the prefix its window still holds

func (f fakeReusable) ReusableKVSlot(slot, prefix int) int { return min(prefix, f[slot]) }

// The G5-C diagnostic counts only slots other than the donor and the destination whose own hit beats reuse and whose
// window still holds it.
func TestKVOtherDonors_countsOnlyServableOthers(t *testing.T) {
	hits := []int{900, 0, 700, 600, 300}
	rk := fakeReusable{0: 100, 2: 700, 3: 200, 4: 300}
	// src 0 (declined), dst 1; slot 2 holds its 700, slot 3 does not hold its 600, slot 4's 300 does not beat reuse 400.
	if got, want := kvOtherDonors(rk, hits, kvmgr.ClassMap{}, 1000, 0, 1, 400), "; other slots that could serve: 1"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if got, want := kvOtherDonors(rk, hits, kvmgr.ClassMap{}, 1000, 0, 2, 400), "; other slots that could serve: 0"; got != want {
		t.Fatalf("dst excluded: got %q, want %q", got, want)
	}
}

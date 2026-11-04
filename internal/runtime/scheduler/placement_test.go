package scheduler

import "testing"

// TestPickDonor_shortBusy: a busy slice shorter than hits, or nil, marks the sessions past its end idle instead of
// panicking; a busy session is never the donor; among equal hits the least recent wins.
func TestPickDonor_shortBusy(t *testing.T) {
	for _, c := range []struct {
		name       string
		hits       []int
		busy       []bool
		donor, hit int
	}{
		{"nil busy", []int{64, 128, 128}, nil, 2, 128},
		{"short busy", []int{64, 128, 128}, []bool{false, true}, 2, 128},
		{"busy best", []int{64, 128}, []bool{false, true}, 0, 64},
		{"no hit", []int{0, 0}, nil, -1, 0},
	} {
		if d, h := PickDonor(c.hits, c.busy); d != c.donor || h != c.hit {
			t.Errorf("%s: PickDonor = (%d, %d), want (%d, %d)", c.name, d, h, c.donor, c.hit)
		}
	}
}

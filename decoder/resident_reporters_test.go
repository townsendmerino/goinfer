package decoder

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// noCapResident has ResidentForward's methods and nothing else, so it reports no ResidentCaps.
type noCapResident struct{ ResidentForward }

type fastFloorResident struct {
	*fakeResident
	floor int
}

func (r *fastFloorResident) ResidentCaps() ResidentCaps {
	c := r.fakeResident.ResidentCaps()
	c.HasFastPrefill, c.FastPrefillFloor = true, r.floor
	return c
}

type kernelFloorResident struct {
	*fakeResident
	floor int
}

func (r *kernelFloorResident) ResidentCaps() ResidentCaps {
	c := r.fakeResident.ResidentCaps()
	c.HasKernelFloor, c.KernelFloor = true, r.floor
	return c
}

// Each fake that reports a value overrides the ResidentCaps it embeds; a fake that fell out of the interface would report nothing.
var (
	_ ResidentCapsReporter = (*fakeResident)(nil)
	_ ResidentCapsReporter = (*residentForwardOnly)(nil)
	_ ResidentCapsReporter = quantReportingResident{}
	_ ResidentCapsReporter = quantReporting{}
	_ ResidentCapsReporter = (*fastPrefillResident)(nil)
	_ ResidentCapsReporter = (*equivResident)(nil)
	_ ResidentCapsReporter = (*chunkFake)(nil)
	_ ResidentCapsReporter = (*blockFake)(nil)
	_ ResidentCapsReporter = (*twoClassFake)(nil)
	_ ResidentCapsReporter = (*fastFloorResident)(nil)
	_ ResidentCapsReporter = (*kernelFloorResident)(nil)
)

// TestClampToResidentCap holds the clamp every generate path takes: no resident, no cap and a cap of 0 cut nothing; a turn that
// would run past the cap ends at it, down to 0 when the prompt fills it; a turn that ends exactly at the cap is not cut.
func TestClampToResidentCap(t *testing.T) {
	cases := []struct {
		name              string
		rf                ResidentForward
		gpuPos, maxTokens int
		want              int
		clamped           bool
	}{
		{"no resident", nil, 90, 20, 20, false},
		{"no cap reported", noCapResident{&fakeResident{}}, 90, 20, 20, false},
		{"cap 0", &fakeResident{capPos: 0}, 90, 20, 20, false},
		{"past the cap", &fakeResident{capPos: 100}, 90, 20, 10, true},
		{"prompt fills the cap", &fakeResident{capPos: 100}, 100, 20, 0, true},
		{"ends at the cap", &fakeResident{capPos: 100}, 80, 20, 20, false},
	}
	for _, c := range cases {
		m := &Model{resident: c.rf}
		if got, clamped := m.clampToResidentCap(c.gpuPos, c.maxTokens); got != c.want || clamped != c.clamped {
			t.Errorf("%s: clampToResidentCap(%d, %d) = %d, %v; want %d, %v", c.name, c.gpuPos, c.maxTokens, got, clamped, c.want, c.clamped)
		}
	}
}

// TestResidentFloors_presenceNotValue: the class rule keys on whether a resident reports a floor, so a FastPrefill resident whose
// floor is 0 (fast kernels off) still takes the fast class's row minimum, and a kernel-floor resident takes MinRows 1 at any floor, 0 included.
func TestResidentFloors_presenceNotValue(t *testing.T) {
	cases := []struct {
		name      string
		rf        ResidentForward
		wantMap   kvmgr.ClassMap
		wantFloor int
	}{
		{"neither", noCapResident{&fakeResident{}}, kvmgr.ClassMap{}, 0},
		{"fast floor 0", &fastFloorResident{&fakeResident{}, 0}, kvmgr.ClassMap{Floor: 0, MinRows: fastPrefillMinRows}, 0},
		{"fast floor 512", &fastFloorResident{&fakeResident{}, 512}, kvmgr.ClassMap{Floor: 512, MinRows: fastPrefillMinRows}, 512},
		{"kernel floor 0", &kernelFloorResident{&fakeResident{}, 0}, kvmgr.ClassMap{Floor: 0, MinRows: 1}, 0},
		{"kernel floor 7", &kernelFloorResident{&fakeResident{}, 7}, kvmgr.ClassMap{Floor: 7, MinRows: 1}, 7},
	}
	for _, c := range cases {
		m := &Model{resident: c.rf}
		if got := m.kvClassMap(); got != c.wantMap {
			t.Errorf("%s: kvClassMap = %+v, want %+v", c.name, got, c.wantMap)
		}
		if got := m.prefillKernelFloor(); got != c.wantFloor {
			t.Errorf("%s: prefillKernelFloor = %d, want %d", c.name, got, c.wantFloor)
		}
	}
}

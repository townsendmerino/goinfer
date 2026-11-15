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

func (f *fastFloorResident) ResidentSubsystems() ResidentSubsystems { return selfSubsystems(f) }

func (r *fastFloorResident) ResidentCaps() ResidentCaps {
	c := r.fakeResident.ResidentCaps()
	c.HasFastPrefill, c.FastPrefillFloor = true, r.floor
	return c
}

type contextNoteResident struct {
	*fakeResident
	note string
}

func (f *contextNoteResident) ResidentSubsystems() ResidentSubsystems { return selfSubsystems(f) }

func (r *contextNoteResident) ResidentCaps() ResidentCaps {
	c := r.fakeResident.ResidentCaps()
	c.ContextNote = r.note
	return c
}

type kernelFloorResident struct {
	*fakeResident
	floor int
}

func (f *kernelFloorResident) ResidentSubsystems() ResidentSubsystems { return selfSubsystems(f) }

func (r *kernelFloorResident) ResidentCaps() ResidentCaps {
	c := r.fakeResident.ResidentCaps()
	c.HasKernelFloor, c.KernelFloor = true, r.floor
	return c
}

// Each fake that reports a value overrides the ResidentCaps it embeds; a fake that fell out of the interface would report nothing.
var (
	_ legacyCapsReporter = (*fakeResident)(nil)
	_ legacyCapsReporter = (*residentForwardOnly)(nil)
	_ legacyCapsReporter = quantReportingResident{}
	_ legacyCapsReporter = quantReporting{}
	_ legacyCapsReporter = (*fastPrefillResident)(nil)
	_ legacyCapsReporter = (*equivResident)(nil)
	_ legacyCapsReporter = (*chunkFake)(nil)
	_ legacyCapsReporter = (*blockFake)(nil)
	_ legacyCapsReporter = (*twoClassFake)(nil)
	_ legacyCapsReporter = (*fastFloorResident)(nil)
	_ legacyCapsReporter = (*kernelFloorResident)(nil)
	_ legacyCapsReporter = (*contextNoteResident)(nil)
	_ legacyCapsReporter = (*mc3Fake)(nil)
)

// TestResidentContextNote_readsTheCaps: Model.ResidentContextNote is the resident's ResidentCaps.ContextNote, and "" with no
// resident or one that reports no caps.
func TestResidentContextNote_readsTheCaps(t *testing.T) {
	for _, c := range []struct {
		name string
		rf   ResidentForward
		want string
	}{
		{"no resident", nil, ""},
		{"no caps", noCapResident{&fakeResident{}}, ""},
		{"a note", &contextNoteResident{&fakeResident{}, "fit: 2 slots, 1.9 GB KV"}, "fit: 2 slots, 1.9 GB KV"},
	} {
		if got := (&Model{resident: c.rf}).ResidentContextNote(); got != c.want {
			t.Errorf("%s: ResidentContextNote = %q, want %q", c.name, got, c.want)
		}
	}
}

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

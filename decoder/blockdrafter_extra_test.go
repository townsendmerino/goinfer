package decoder

import "testing"

// TestExtraResidentBytes_reachesTheModel closes the loop the estimate feeds: internal/serveapp's
// loadDecoder sets Options.ExtraResidentBytes before calling Load, and a backend
// (cuda/resident.go's checkWeightsFit/checkKVFits/allocSlots) reads it back off *Model, not off
// Options — so a struct-literal typo at either of Load's two &Model{...} sites would silently
// zero it and no backend-level test could ever see why a --drafter attach still ran out of room.
// Load()-based (a real tracked fixture) rather than constructing *Model directly, so this proves
// the actual production path, matching TestFitDisabled/TestResolveCtxCapFit_shortcuts' own
// "load a real model, don't pass nil" precedent (cuda/resident_cap_test.go).
func TestExtraResidentBytes_reachesTheModel(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32", ExtraResidentBytes: 123456789})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if got := m.ExtraResidentBytes(); got != 123456789 {
		t.Errorf("ExtraResidentBytes() = %d, want 123456789", got)
	}
}

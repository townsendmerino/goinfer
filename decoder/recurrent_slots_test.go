package decoder

import "testing"

// TestResidentKVSlotsRequestFor_recurrentSlotsByTable is OP4m-d4a's D4a-3 and D4a-4, with no GPU
// (docs/tasks/task-forward-architecture-2026-10.md §4.9).
//   - A Gated-DeltaNet model that asks for three slots gets them on Metal, whose table keeps that state per slot.
//   - It gets one on every other backend, on the no-backend request, and at the caller's default request.
//   - A Mamba-2 hybrid, whose mixer Metal's table does not list, gets one on Metal.
//   - A dense model gets its request everywhere.
func TestResidentKVSlotsRequestFor_recurrentSlotsByTable(t *testing.T) {
	load := func(dir string, o Options) *Model {
		t.Helper()
		m, err := Load(dir, o)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	dn := load("../testdata/qwen35-tiny", Options{ResidentKVSlots: 3})
	for be, want := range map[string]int{"metal": 3, "cuda": 1, "webgpu": 1, "": 1} {
		if got := dn.ResidentKVSlotsRequestFor(be); got != want {
			t.Errorf("Gated-DeltaNet on %q: %d slots, want %d", be, got, want)
		}
	}
	if got := dn.ResidentKVSlotsRequest(); got != 1 {
		t.Errorf("Gated-DeltaNet, ResidentKVSlotsRequest: %d slots, want 1", got)
	}
	if got := load("../testdata/qwen35-tiny", Options{ResidentKVSlots: 3, ResidentKVSlotsDefault: true}).ResidentKVSlotsRequestFor("metal"); got != 1 {
		t.Errorf("Gated-DeltaNet at the caller's default request on metal: %d slots, want 1", got)
	}
	if got := load("../testdata/nemotron-tiny", Options{ResidentKVSlots: 3}).ResidentKVSlotsRequestFor("metal"); got != 1 {
		t.Errorf("the Mamba-2 hybrid on metal: %d slots, want 1", got)
	}
	dense := load("../testdata/mixtral-tiny", Options{ResidentKVSlots: 3})
	for _, be := range []string{"metal", "cuda", ""} {
		if got := dense.ResidentKVSlotsRequestFor(be); got != 3 {
			t.Errorf("a dense model on %q: %d slots, want 3", be, got)
		}
	}
}

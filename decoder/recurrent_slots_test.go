package decoder

import "testing"

// TestResidentKVSlotsRequestFor_recurrentSlotsByTable is OP4m-d4a's D4a-3 and D4a-4, and OP4m-d6a's D6a-3 and D6a-4, with no
// GPU (docs/tasks/task-forward-architecture-2026-10.md §4.9).
//   - A Gated-DeltaNet model or a Mamba-2 hybrid that asks for three slots gets them on Metal, whose table keeps that state
//     per slot.
//   - Each gets one on every other backend, on the no-backend request, and at the caller's default request.
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
	for _, c := range []struct{ name, dir string }{{"Gated-DeltaNet", "../testdata/qwen35-tiny"}, {"the Mamba-2 hybrid", "../testdata/nemotron-tiny"}} {
		rec := load(c.dir, Options{ResidentKVSlots: 3})
		for be, want := range map[string]int{"metal": 3, "cuda": 1, "webgpu": 1, "": 1} {
			if got := rec.ResidentKVSlotsRequestFor(be); got != want {
				t.Errorf("%s on %q: %d slots, want %d", c.name, be, got, want)
			}
		}
		if got := rec.ResidentKVSlotsRequest(); got != 1 {
			t.Errorf("%s, ResidentKVSlotsRequest: %d slots, want 1", c.name, got)
		}
		if got := load(c.dir, Options{ResidentKVSlots: 3, ResidentKVSlotsDefault: true}).ResidentKVSlotsRequestFor("metal"); got != 1 {
			t.Errorf("%s at the caller's default request on metal: %d slots, want 1", c.name, got)
		}
	}
	dense := load("../testdata/mixtral-tiny", Options{ResidentKVSlots: 3})
	for _, be := range []string{"metal", "cuda", ""} {
		if got := dense.ResidentKVSlotsRequestFor(be); got != 3 {
			t.Errorf("a dense model on %q: %d slots, want 3", be, got)
		}
	}
}

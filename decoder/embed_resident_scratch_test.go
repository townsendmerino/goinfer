package decoder

import (
	"testing"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
)

// TestEmbedResidentInto_reusesBuffer is P-08 (audit-2026-09-10): embedResident allocated a fresh
// [hidden]float32 on every call, including from the resident decode loop's steady state (one
// call per generated token). embedResidentInto lets that loop reuse one buffer instead. This
// asserts BOTH halves directly — correctness (matches the always-fresh embedResident exactly, not
// just "some values") and the reuse itself (the returned slice's backing array is the one passed
// in, not a fresh allocation) — since a fix that reused the wrong bytes or silently stopped
// reusing would both pass a test that checked only one side.
func TestEmbedResidentInto_reusesBuffer(t *testing.T) {
	const vocab, hidden = 8, 4
	vals := make([]float32, vocab*hidden)
	for i := range vals {
		vals[i] = float32(i)
	}
	m := &Model{w: &Weights{
		arch:  &Architecture{VocabSize: vocab, HiddenDim: hidden},
		Embed: linalg.WrapF32(vals, vocab, hidden),
	}}

	want := m.embedResident(3)
	got := m.embedResidentInto(3, nil)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("embedResidentInto(3, nil)[%d] = %v, want %v (embedResident's own answer)", i, got[i], want[i])
		}
	}

	// Reuse: a second call passed the first's own return value must write into that SAME backing
	// array, not allocate a new one.
	scratch := got
	base := unsafe.Pointer(&scratch[0])
	got2 := m.embedResidentInto(5, scratch)
	if unsafe.Pointer(&got2[0]) != base {
		t.Error("embedResidentInto did not reuse the destination buffer's backing array — the allocation this fix removes is still happening")
	}
	want2 := m.embedResident(5)
	for i := range want2 {
		if got2[i] != want2[i] {
			t.Fatalf("embedResidentInto(5, scratch)[%d] = %v, want %v (embedResident's own answer) — reused buffer holds stale/wrong data", i, got2[i], want2[i])
		}
	}
}

// TestEmbedResidentInto_growsWhenTooSmall proves the destination is grown (not corrupted or
// left short) when it doesn't already fit — the escape hatch every grow-on-demand buffer needs,
// exercised directly rather than assumed from the happy path above.
func TestEmbedResidentInto_growsWhenTooSmall(t *testing.T) {
	const vocab, hidden = 4, 6
	vals := make([]float32, vocab*hidden)
	for i := range vals {
		vals[i] = float32(i)
	}
	m := &Model{w: &Weights{
		arch:  &Architecture{VocabSize: vocab, HiddenDim: hidden},
		Embed: linalg.WrapF32(vals, vocab, hidden),
	}}
	tooSmall := make([]float32, 2) // shorter than hidden
	got := m.embedResidentInto(1, tooSmall)
	if len(got) != hidden {
		t.Fatalf("len = %d, want %d — did not grow a too-small destination", len(got), hidden)
	}
	want := m.embedResident(1)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

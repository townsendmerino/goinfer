package decoder

import (
	"strings"
	"testing"
)

// C-07: HiddenLast HAD NO LENGTH BOUND, only a vocab one.
//
// It preallocates KV for len(ids) positions and then runs one sequential forward per token with no
// context to cancel it. So an over-long input is not slow, it is a ~114 GB allocation (28 layers,
// kvDim 1024, 500k positions) plus attention over up to len(ids) keys per token, holding the
// caller's mutex until the process is OOM-killed. The serving embedder's only bound was C-21's
// 1 MiB of BYTES — about 500k tokens of short words.
//
// The bound is asserted HERE, where the cost is incurred, so a caller other than the serving
// embedder cannot reintroduce it.
func TestHiddenLast_refusesMoreTokensThanTheContextWindow(t *testing.T) {
	m := &Model{w: &Weights{
		Cfg:  Config{MaxPositions: 8},
		arch: &Architecture{Name: "test", NumLayers: 1, HiddenDim: 4, VocabSize: 16},
	}}
	// THE FIELD MATTERS. The Architecture's own MaxPositions is the GPT-2 learned-position table
	// size and is 0 for every RoPE family; keying the guard on it would make this test pass while
	// the guard never fired in production. Pinned so a "simplification" back to a.MaxPositions
	// fails here.
	if m.w.arch.MaxPositions != 0 {
		t.Fatal("premise broke: this fixture's Architecture.MaxPositions must be 0, which is what " +
			"every RoPE family has — that is the field the guard must NOT use")
	}

	ids := make([]int, 9) // one past the window
	_, err := m.HiddenLast(ids)
	if err == nil {
		t.Fatal("HiddenLast accepted more tokens than the context window: it would preallocate KV " +
			"for all of them and pool from out-of-range RoPE positions")
	}
	if !strings.Contains(err.Error(), "context") {
		t.Errorf("refused for the wrong reason: %v", err)
	}

	// A within-window input must still be refused only for reasons of its own — here, the seam is
	// not wired for a model with no weights, which is a different error. What must NOT happen is
	// the length guard firing on a legal length.
	if _, err := m.HiddenLast(make([]int, 8)); err != nil && strings.Contains(err.Error(), "context_length_exceeded") {
		t.Errorf("a full-window input was rejected as too long: %v", err)
	}
}

// A model that declares no window (0) must not be bounded to zero — that would refuse everything.
func TestHiddenLast_unknownWindowDoesNotRefuseEverything(t *testing.T) {
	m := &Model{w: &Weights{
		Cfg:  Config{MaxPositions: 0},
		arch: &Architecture{Name: "test", NumLayers: 1, HiddenDim: 4, VocabSize: 16},
	}}
	if _, err := m.HiddenLast(make([]int, 32)); err != nil && strings.Contains(err.Error(), "context_length_exceeded") {
		t.Errorf("an unknown context window rejected a 32-token input: %v", err)
	}
}

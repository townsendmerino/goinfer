package constrain

import (
	"math"
	"testing"
)

// A toy vocabulary: prose words, a whole-trigger token, a trigger split across two tokens, the
// pieces of a call, and EOS (id eosID).
var lazyVocab = [][]byte{
	[]byte("Hello"), []byte(" world"), []byte("<tool_call>"), []byte("<tool"), []byte("_call>"),
	[]byte("\n"), []byte(`{"name": "`), []byte("read"), []byte("git_commit"), []byte(`", "arguments": {"x": "1"}}`),
	[]byte("\n</tool_call>"), []byte(" tag"), []byte(" "), nil,
}

const (
	tHello, tWorld, tTrig, tTrigA, tTrigB, tNL, tOpen, tRead, tGitCommit, tArgs, tClose, tTag, tSpace, eosID = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
)

func newLazy(t *testing.T) *LazyMasker {
	t.Helper()
	g, err := ToolCallsGrammar("<tool_call>", "\n</tool_call>", "arguments", false, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	return NewLazyMasker(NewMasker(g, lazyVocab, []int{eosID}), "<tool_call>")
}

// step runs Process over the generated ids and returns which ids stayed legal.
func step(l *LazyMasker, gen []int) map[int]bool {
	logits := make([]float32, len(lazyVocab))
	l.Process(gen, logits)
	legal := map[int]bool{}
	for id, v := range logits {
		if !math.IsInf(float64(v), -1) {
			legal[id] = true
		}
	}
	return legal
}

// The non-forcing gate: an output that never contains the trigger is never masked, at any step.
func TestLazyMasker_neverMasksWithoutTrigger(t *testing.T) {
	l := newLazy(t)
	gen := []int{}
	for _, id := range []int{tHello, tWorld, tSpace, tTag, tHello} {
		if got := step(l, gen); len(got) != len(lazyVocab) {
			t.Fatalf("after %v: %d of %d ids legal — masked without a trigger", gen, len(got), len(lazyVocab))
		}
		gen = append(gen, id)
	}
	if l.Arms() != 0 {
		t.Errorf("armed %d times on prose", l.Arms())
	}
}

func TestLazyMasker_armsOnTrigger_andConstrains(t *testing.T) {
	for name, gen := range map[string][]int{"whole token": {tHello, tTrig}, "split across tokens": {tHello, tTrigA, tTrigB}} {
		t.Run(name, func(t *testing.T) {
			l := newLazy(t)
			legal := step(l, gen)
			if !l.Armed() {
				t.Fatal("did not arm on the trigger")
			}
			if legal[tHello] || legal[eosID] || legal[tRead] {
				t.Errorf("after the trigger: prose=%v eos=%v name-without-opener=%v legal; want all false", legal[tHello], legal[eosID], legal[tRead])
			}
			if !legal[tNL] || !legal[tOpen] {
				t.Errorf("after the trigger: newline=%v opener=%v legal; want both", legal[tNL], legal[tOpen])
			}
			legal = step(l, append(gen, tNL, tOpen))
			if !legal[tRead] || legal[tGitCommit] {
				t.Errorf("at the name: read=%v git_commit=%v; want true, false (unknown tool)", legal[tRead], legal[tGitCommit])
			}
		})
	}
}

// A space instead of a newline after the opener is still a call the grammar accepts (the lazy
// grammar's prefix is the trigger alone; JSON admits leading whitespace).
func TestLazyMasker_toleratesSpaceAfterOpener(t *testing.T) {
	l := newLazy(t)
	if legal := step(l, []int{tTrig, tSpace}); !l.Armed() || !legal[tOpen] {
		t.Errorf("armed=%v opener-legal=%v after \"<tool_call> \"", l.Armed(), legal[tOpen])
	}
}

// The trigger inside prose, continued in a shape no call can take, fails OPEN: no masking.
func TestLazyMasker_failsOpenOnProseMention(t *testing.T) {
	l := newLazy(t)
	gen := []int{tHello, tTrig, tTag} // "Hello<tool_call> tag" — the second token arrives with the trigger
	// fold the trigger token alone first: it arms, and " tag" is then masked, as intended
	step(l, gen[:2])
	if !l.Armed() {
		t.Fatal("a bare trigger token must arm (the model committed to a call)")
	}
	// but a token that carries the trigger AND an impossible continuation cannot arm
	l2 := NewLazyMasker(NewMasker(mustUnion(t), [][]byte{[]byte("use <tool_call> tags"), []byte("x"), nil}, []int{2}), "<tool_call>")
	logits := make([]float32, 3)
	l2.Process([]int{0}, logits)
	if l2.Armed() || math.IsInf(float64(logits[1]), -1) {
		t.Errorf("armed=%v on \"use <tool_call> tags\"; the continuation is impossible, so it must fail open", l2.Armed())
	}
}

// Completing a call disarms (the model is free again), and a second call arms again (T5).
func TestLazyMasker_disarmsAfterCall_andRearms(t *testing.T) {
	l := newLazy(t)
	call := []int{tTrig, tNL, tOpen, tRead, tArgs}
	legal := step(l, call)
	if legal[eosID] {
		t.Error("EOS legal before the call's suffix — the call would be cut off")
	}
	gen := append(append([]int{}, call...), tClose)
	legal = step(l, gen)
	if l.Armed() || len(legal) != len(lazyVocab) {
		t.Fatalf("after a complete call: armed=%v, %d/%d legal; want disarmed and unmasked", l.Armed(), len(legal), len(lazyVocab))
	}
	gen = append(gen, tNL, tTrig)
	step(l, gen)
	if !l.Armed() || l.Arms() != 2 {
		t.Errorf("second call: armed=%v arms=%d; want true, 2", l.Armed(), l.Arms())
	}
}

func mustUnion(t *testing.T) Grammar {
	t.Helper()
	g, err := ToolCallsGrammar("<tool_call>", "\n</tool_call>", "arguments", false, unionTools)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

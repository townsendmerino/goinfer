package kvmgr

import (
	"slices"
	"testing"
)

type fakeSeqState struct {
	tokens    []int
	root      Root
	images    []ImageSpan
	classes   []ClassRun
	recurrent bool
	ckpt      int
	floor     int
	winBase   int
	window    int
	truncates []int
	truncOK   func(int) bool
}

func (f *fakeSeqState) Len() int            { return len(f.tokens) }
func (f *fakeSeqState) Tokens() []int       { return f.tokens }
func (f *fakeSeqState) Root() Root          { return f.root }
func (f *fakeSeqState) Images() []ImageSpan { return f.images }
func (f *fakeSeqState) Classes() []ClassRun { return f.classes }
func (f *fakeSeqState) Recurrent() bool     { return f.recurrent }
func (f *fakeSeqState) Checkpoint() int     { return f.ckpt }
func (f *fakeSeqState) Floor() int          { return f.floor }

func (f *fakeSeqState) ReusablePrefix(prefix int) int {
	if prefix <= 0 || prefix > len(f.tokens) {
		return 0
	}
	if f.window > 0 && !WindowReusable(f.winBase, prefix, f.window) {
		return 0
	}
	return prefix
}

func (f *fakeSeqState) Truncate(target int) bool {
	f.truncates = append(f.truncates, target)
	if f.truncOK != nil && !f.truncOK(target) {
		return false
	}
	if target == 0 {
		f.tokens = nil
		f.images = nil
		f.classes = nil
		f.ckpt = 0
		f.winBase = 0
		return true
	}
	if target > len(f.tokens) {
		return false
	}
	if f.recurrent && target < len(f.tokens) {
		return false
	}
	f.tokens = f.tokens[:target]
	f.classes = ClipClassRuns(f.classes, target)
	return true
}

func TestWindowHorizonAndRingTruncate(t *testing.T) {
	if got := WindowNeedStart(10, 0); got != 0 {
		t.Fatalf("WindowNeedStart global = %d, want 0", got)
	}
	if got := WindowNeedStart(10, 4); got != 7 {
		t.Fatalf("WindowNeedStart(10, 4) = %d, want 7", got)
	}
	if got := WindowHorizon(10, 4, 2); got != 4 {
		t.Fatalf("WindowHorizon(10, 4, 2) = %d, want 4", got)
	}
	if got := WindowHorizon(6, 4, 2); got != 0 {
		t.Fatalf("WindowHorizon(6, 4, 2) = %d, want 0", got)
	}

	// RingTruncateExact agrees with a ring of capacity W: unwrapped cuts are always exact; wrapped cuts are exact only
	// at written-1 or above.
	for written := 0; written <= 12; written++ {
		for target := 0; target <= written+2; target++ {
			want := target >= written || written <= 4 || target >= written-1
			if got := RingTruncateExact(written, 4, target); got != want {
				t.Fatalf("RingTruncateExact(written=%d, w=4, target=%d) = %v, want %v", written, target, got, want)
			}
		}
	}

	// WindowNextBase advances only when capN is exceeded, preserves rollback, and rejects history below base.
	from, to, _, st := WindowNextBase(0, 0, 8, 4, 10, 2)
	if st != WindowStepOK || from != 0 || to != 0 {
		t.Fatalf("WindowNextBase within cap = (%d, %d, %v), want (0, 0, ok)", from, to, st)
	}
	from, to, _, st = WindowNextBase(0, 8, 4, 4, 10, 2)
	if st != WindowStepOK || from != 0 || to != 5 {
		t.Fatalf("WindowNextBase compact = (%d, %d, %v), want (0, 5, ok)", from, to, st)
	}
	if !WindowReusable(to, 8, 4) || WindowReusable(to, 7, 4) {
		t.Fatalf("WindowReusable around base=%d window=4 disagrees with horizon", to)
	}
	if _, _, _, st = WindowNextBase(5, 7, 1, 4, 10, 2); st != WindowStepBelowBase {
		t.Fatalf("WindowNextBase below base = %v, want WindowStepBelowBase", st)
	}
	if _, _, _, st = WindowNextBase(0, 0, 12, 4, 10, 2); st != WindowStepOverCap {
		t.Fatalf("WindowNextBase over cap = %v, want WindowStepOverCap", st)
	}
}

func TestSequenceState_ScoreRewindReconcileAndShare(t *testing.T) {
	root := Root{Model: "m", Precision: "f16"}
	otherRoot := Root{Model: "m", Adapter: 7, Precision: "f16"}

	slots := NewSlots(2)
	prompt := make([]int, 160)
	for i := range prompt {
		prompt[i] = i + 1
	}
	cm := ClassMap{Floor: 64, MinRows: 32}
	slots.Acquired(0, 0, prompt[:128])
	slots.Commit(0, root, slices.Clone(prompt[:140]), []ImageSpan{{Start: 64, Len: 16, Hash: 99}}, 128, cm)
	slots.SetFloor(0, 32)

	st := slots.State(0)
	// Matching image claim reuses across the span; missing claim stops at span start; floor blocks shorter matches.
	reqOK := Request{Root: root, Tokens: prompt, Images: []ImageSpan{{Start: 64, Len: 16, Hash: 99}}}
	if got, want := ScoreSequence(st, reqOK), slots.Score(0, reqOK, false); got != 140 || got != want {
		t.Fatalf("ScoreSequence with matching image = %d (Slots.Score=%d), want 140", got, want)
	}
	reqNoImg := Request{Root: root, Tokens: prompt}
	if got := ScoreSequence(st, reqNoImg); got != 64 {
		t.Fatalf("ScoreSequence without image claim = %d, want 64 (span start)", got)
	}
	reqShort := Request{Root: root, Tokens: append(slices.Clone(prompt[:20]), 999, 1000)}
	if got := ScoreSequence(st, reqShort); got != 0 {
		t.Fatalf("ScoreSequence below floor = %d, want 0", got)
	}
	if got := SharedPrefixHit(st, prompt, cm, true); got != 128 {
		t.Fatalf("SharedPrefixHit = %d, want 128", got)
	}

	// Windowed SequenceState drops to 0 when prefix falls behind winBase+window-1.
	fs := &fakeSeqState{tokens: slices.Clone(prompt[:100]), root: root, winBase: 80, window: 16}
	if got := ScoreSequence(fs, Request{Root: root, Tokens: append(slices.Clone(prompt[:90]), 500)}); got != 0 {
		t.Fatalf("ScoreSequence behind window base = %d, want 0", got)
	}
	if got := ScoreSequence(fs, Request{Root: root, Tokens: append(slices.Clone(prompt[:96]), 500)}); got != 96 {
		t.Fatalf("ScoreSequence inside window base = %d, want 96", got)
	}

	// Clean rewind never touches Truncate; adapter/root mismatch resets to 0; inexact cut resets to 0.
	cleanSt := &fakeSeqState{root: root}
	if got := RewindSequence(cleanSt, Request{Root: root, Tokens: []int{1, 2}}, true); got != 0 || len(cleanSt.truncates) != 0 {
		t.Fatalf("clean RewindSequence = %d, truncates=%v, want 0 and none", got, cleanSt.truncates)
	}
	mismatchSt := &fakeSeqState{tokens: []int{1, 2, 3}, root: root}
	if got := RewindSequence(mismatchSt, Request{Root: otherRoot, Tokens: []int{1, 2, 3, 4}}, false); got != 0 || !slices.Equal(mismatchSt.truncates, []int{0}) {
		t.Fatalf("root-mismatch RewindSequence = %d, truncates=%v, want 0 and [0]", got, mismatchSt.truncates)
	}

	// Recurrent ReconcileSequence resets to 0 on mid-write rollback; non-recurrent clamps to pos.
	var recTrunc []int
	toks, keep := ReconcileSequence(3, []int{1, 2, 3, 4, 5}, func() bool { return true }, func(p int) bool {
		recTrunc = append(recTrunc, p)
		return true
	})
	if keep || toks != nil || !slices.Equal(recTrunc, []int{0}) {
		t.Fatalf("recurrent ReconcileSequence = (%v, %v, %v), want (nil, false, [0])", toks, keep, recTrunc)
	}
	toks, keep = ReconcileSequence(3, []int{1, 2, 3, 4, 5}, func() bool { return false }, func(int) bool { return true })
	if !keep || !slices.Equal(toks, []int{1, 2, 3}) {
		t.Fatalf("positional ReconcileSequence = (%v, %v), want ([1 2 3], true)", toks, keep)
	}

	// DecideRecurrentCheckpoint enforces checkpoint availability and final-pass tail minimum.
	if next, restore, why := DecideRecurrentCheckpoint(64, 100, 80, 16, true, true); next != 64 || !restore || why != "" {
		t.Fatalf("DecideRecurrentCheckpoint valid = (%d, %v, %q), want (64, true, \"\")", next, restore, why)
	}
	if next, restore, why := DecideRecurrentCheckpoint(64, 100, 72, 16, true, true); next != 0 || restore || why != "suffix under the final-pass minimum" {
		t.Fatalf("DecideRecurrentCheckpoint short tail = (%d, %v, %q)", next, restore, why)
	}
	if next, restore, why := DecideRecurrentCheckpoint(64, 100, 80, 16, true, false); next != 0 || restore || why != "no checkpoint kept" {
		t.Fatalf("DecideRecurrentCheckpoint missing = (%d, %v, %q)", next, restore, why)
	}
}

package decoder

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

func TestSequenceState_CPUAndResidentParity(t *testing.T) {
	m := plainModel(t)

	// 1. CPU Session SequenceState: prefix score, wrapped-ring horizon, and len(tokens)>pos guard.
	s := m.NewSession(16)
	for range 6 {
		for l := 0; l < s.cache.numLayers; l++ {
			s.cache.Append(l, make([]float32, s.cache.kvDim), make([]float32, s.cache.kvDim))
		}
	}
	s.reconcile([]int{10, 20, 30, 40, 50, 60})

	cpuSt := s.SequenceState()
	if cpuSt.Len() != 6 || !slices.Equal(cpuSt.Tokens(), []int{10, 20, 30, 40, 50, 60}) {
		t.Fatalf("CPU SequenceState tokens = %v (len %d), want 6 tokens", cpuSt.Tokens(), cpuSt.Len())
	}
	score := kvmgr.ScoreSequence(cpuSt, kvmgr.Request{Root: cpuSt.Root(), Tokens: []int{10, 20, 30, 40, 99}})
	if score != 4 {
		t.Fatalf("CPU ScoreSequence = %d, want 4", score)
	}

	// If tokens ever exceed cache.Pos(), ReusablePrefix refuses the out-of-bounds prefix.
	s.tokens = append(s.tokens, 70, 80)
	if got := cpuSt.ReusablePrefix(7); got != 0 {
		t.Fatalf("CPU ReusablePrefix past cache.Pos() = %d, want 0", got)
	}
	s.tokens = s.tokens[:6]

	// Wrapped ring on CPU SequenceState: rewinding by 1 is servable; rewinding by 2 drops below the horizon.
	ringCache := NewKVCache(1, 1, 4, 4, 8, nil)
	ringCache.enableRings(4, func(int) bool { return false })
	for range 6 {
		ringCache.Append(0, make([]float32, 4), make([]float32, 4))
	}
	ringSess := &Session{m: m, cache: ringCache, tokens: []int{1, 2, 3, 4, 5, 6}}
	ringSt := ringSess.SequenceState()
	if got := kvmgr.ScoreSequence(ringSt, kvmgr.Request{Root: ringSt.Root(), Tokens: []int{1, 2, 3, 4, 5, 99}}); got != 5 {
		t.Fatalf("wrapped ring ScoreSequence(len-1) = %d, want 5", got)
	}
	if got := kvmgr.ScoreSequence(ringSt, kvmgr.Request{Root: ringSt.Root(), Tokens: []int{1, 2, 3, 4, 99}}); got != 0 {
		t.Fatalf("wrapped ring ScoreSequence(len-2) = %d, want 0", got)
	}

	// 2. Resident slot SequenceState: shares the identical ScoreSequence and Truncate(0) contract.
	slots := m.kvSlots()
	slots.Resize(2)
	prompt := []int{10, 20, 30, 40, 50}
	slots.Acquired(0, 0, prompt)
	slots.Commit(0, m.kvRoot(nil), []int{10, 20, 30, 40, 50, 60}, nil, len(prompt), m.kvClassMap())

	resSt := m.ResidentSequenceState(0)
	if resSt.Len() != 6 || !slices.Equal(resSt.Tokens(), []int{10, 20, 30, 40, 50, 60}) {
		t.Fatalf("Resident SequenceState tokens = %v (len %d), want 6 tokens", resSt.Tokens(), resSt.Len())
	}
	if got := kvmgr.ScoreSequence(resSt, kvmgr.Request{Root: m.kvRoot(nil), Tokens: []int{10, 20, 30, 40, 99}}); got != 4 {
		t.Fatalf("Resident ScoreSequence = %d, want 4", got)
	}
	slots.SetFloor(0, 5)
	if got := kvmgr.ScoreSequence(resSt, kvmgr.Request{Root: m.kvRoot(nil), Tokens: []int{10, 20, 30, 40, 99}}); got != 0 {
		t.Fatalf("Resident ScoreSequence below floor = %d, want 0", got)
	}
	if !resSt.Truncate(0) || resSt.Len() != 0 {
		t.Fatalf("Resident SequenceState Truncate(0) failed to clear slot: len=%d", resSt.Len())
	}
}

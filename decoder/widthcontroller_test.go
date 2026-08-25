// Unit gate for the adaptive verify-width controller. No checkpoints: the controller is pure
// arithmetic over committed-tokens-per-round, so its SHAPE can and should be pinned in plain
// CI rather than behind `realckpt` where only a release sweep would ever run it.
//
// What this cannot show is whether adapting is FASTER — that needs real pairings, interleaved,
// and it is the ship-gate in docs/prompts/adaptive-speculation.md. This file pins the four
// behaviours the design argues for, so a later tuning change has to break a stated claim
// rather than quietly drift.
package decoder

import "testing"

// TestWidthController_settlesAtMeasuredOptima pins the target function against the two suites
// whose tok/round AND optimum width were both measured (11aeed4): chat 1.96 -> 4, math 5.88 -> 8.
// If this fails after a tuning change, widthHeadroom is the thing that moved.
func TestWidthController_settlesAtMeasuredOptima(t *testing.T) {
	for _, tc := range []struct {
		name      string
		commit    int
		start     int
		wantWidth int
	}{
		{"chat-like avg 2/round", 2, 8, 4},
		{"math-like avg 6/round", 6, 4, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newWidthController(tc.start, 2, 16, nil)
			for i := 0; i < 40; i++ {
				if !c.observe(tc.commit) {
					t.Fatalf("stopped at round %d: a marginal workload must NARROW, not disable", i)
				}
			}
			if c.cur != tc.wantWidth {
				t.Errorf("settled at width %d, want %d (the measured optimum for this acceptance)", c.cur, tc.wantWidth)
			}
		})
	}
}

// TestWidthController_deadDrafterStops is the regression the narrowing behaviour could
// otherwise introduce. The binary guard disabled anything below break-even; the controller
// narrows instead, so a drafter accepting NOTHING must still be switched off rather than left
// idling at some narrow width paying draft+verify to produce one token a round.
func TestWidthController_deadDrafterStops(t *testing.T) {
	c := newWidthController(8, 2, 16, nil)
	for i := 0; i < 40; i++ {
		if !c.observe(1) { // 1 committed = the target's own token, nothing accepted
			return
		}
	}
	t.Fatalf("never stopped at 1.0 tok/round (width %d) — that is a pure loss every round, "+
		"and the binary guard it replaces would have caught it", c.cur)
}

// TestWidthController_shrinksFastGrowsSlow pins the asymmetry. A too-wide draft wastes
// draft+verify on tail positions that rarely land; a too-narrow one only leaves throughput on
// the table. So width rounds toward the cheap error.
func TestWidthController_shrinksFastGrowsSlow(t *testing.T) {
	// Shrink: one qualifying round takes it all the way down.
	c := newWidthController(16, 2, 16, nil)
	for i := 0; i < guardWindow; i++ {
		c.observe(2)
	}
	if c.cur > 4 {
		t.Errorf("shrink took more than the first qualifying round: width %d, want <= 4", c.cur)
	}
	// Grow: sustained evidence, one position at a time.
	g := newWidthController(2, 2, 16, nil)
	for i := 0; i < guardWindow; i++ {
		g.observe(6)
	}
	first := g.cur
	if first > 3 {
		t.Errorf("grew to %d in one step; growth must be +1 on sustained evidence", first)
	}
	for i := 0; i < growPatience*3; i++ {
		g.observe(6)
	}
	if g.cur <= first {
		t.Errorf("width never grew under sustained high acceptance (stuck at %d)", g.cur)
	}
}

// TestWidthController_defaultOffIsTheOldGuard proves the shipped default did not move. With
// Adaptive false, generate() pins min==max==width, and the controller must then reproduce
// acceptanceGuard exactly: cumulative average, judged no earlier than guardWindow, stopping
// below breakEvenTokensPerRound.
func TestWidthController_defaultOffIsTheOldGuard(t *testing.T) {
	for _, commit := range []int{1, 2, 3, 5} {
		pinned := newWidthController(8, 8, 8, nil)
		var old acceptanceGuard
		for round := 1; round <= 25; round++ {
			gotNew, gotOld := pinned.observe(commit), old.observe(commit)
			if gotNew != gotOld {
				t.Fatalf("commit=%d round=%d: controller says continue=%v, guard says %v — "+
					"the default-off path must be behaviourally identical to the guard it replaces",
					commit, round, gotNew, gotOld)
			}
		}
		if pinned.cur != 8 {
			t.Errorf("commit=%d: pinned width moved to %d — Adaptive=false must never change width", commit, pinned.cur)
		}
	}
}

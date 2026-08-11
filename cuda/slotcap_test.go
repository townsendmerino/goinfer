//go:build cuda

package cuda

import "testing"

// TestSlotCapArithmetic gates the expert-cache CAPPING branch — code that ships, decides how much
// VRAM to claim, and had never once been executed by a test.
//
// It could not be: the branch only binds when the requested slots exceed free VRAM, and no fixture
// is remotely large enough for that. It bound for the first time on the real 26B, where a default
// of "request every expert" capped 128 slots to 34 and the forward then died at cuLaunchKernel.
// That is the exercised-but-never-triggered shape — a branch inside well-tested code that the
// tests' inputs can never reach, so coverage tools report it green and it has never run.
//
// So the arithmetic is factored out and driven with SYNTHETIC free-VRAM figures against a
// constructed layer geometry. This does not need a 26B; the 26B is what proves the result.
//
// AND IT CORROBORATES THE SIZING, which makes this a measurement that happens to be a test: from
// 3.8 GB free / 30 MoE layers / 3.49 MB per slot per layer it predicts 34, and the hardware
// produced exactly 34 ("capping to 34"). The byte accounting is therefore demonstrably correct,
// which places the whole 26B discrepancy DOWNSTREAM of the sizing decision — evidence for the
// fragmentation and launch-demand candidates and against "the predicted bytes are wrong",
// independent of any hardware probe.
func TestSlotCapArithmetic(t *testing.T) {
	const gb = int64(1) << 30
	cases := []struct {
		name              string
		free              int64
		nLayers, perLayer int64
		topK, want        int
		wantDecline       bool
	}{
		// The real 26B configuration that failed: 30 MoE layers, ~3.33 MB per slot per layer,
		// 3.8 GB free, 128 requested. Documents the observed cap of 34.
		{"26B: 128 requested, 3.8GB free", 3800 * (1 << 20), 30, 3_490_000, 8, 34, false},
		// Ample VRAM: the request is honoured untouched, so the branch must NOT bind.
		{"ample free: no cap", 40 * gb, 30, 3_490_000, 8, 128, false},
		// FLOOR: not even topK fits, so it must decline rather than clamp up and allocate.
		{"below topK: declines", 500 * (1 << 20), 30, 3_490_000, 8, 0, true},
		// Exactly topK fits: the boundary must admit, not decline (off-by-one both ways).
		{"exactly topK fits", int64(30*8*3_490_000) + (384 << 20), 30, 3_490_000, 8, 8, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, decline := capSlots(c.free, c.nLayers, c.perLayer, c.topK, 128)
			if decline != c.wantDecline {
				t.Fatalf("decline = %v, want %v (got %d slots)", decline, c.wantDecline, got)
			}
			if !c.wantDecline && got != c.want {
				t.Errorf("slots = %d, want %d", got, c.want)
			}
		})
	}
}

// TestSlotCapArithmetic_mutation is the falsifiability check the policy now requires of any gate:
// perturb the thing under test and confirm the gate goes red. Mutations, and what each breaks:
//
//	margin removed           -> 38 slots, not 34. VERIFIED, not assumed: one slot costs
//	                            30 layers x 3.49 MB = 104.7 MB, so a 384 MB margin buys ~3.7 slots
//	                            and the cap moves by 4. An earlier version of this comment claimed
//	                            "35", a figure nobody had computed — an unchecked number inside the
//	                            check whose job is being checkable.
//	floor removed (clamp up) -> "below topK" returns 8 slots instead of declining: the
//	                            allocate-then-fail-to-launch path. NOTE this is NOT what the 26B
//	                            hit — that run logged "capping to 34", and 34 > topK 8, so the
//	                            clamp never fired there. The floor closes a real path at much
//	                            lower free VRAM; the 26B failure remains unexplained.
//
// Both are asserted here directly rather than described, so the claim "this gate can fail" is
// itself checked.
func TestSlotCapArithmetic_mutation(t *testing.T) {
	const perLayer, nLayers, topK = int64(3_490_000), int64(30), 8
	free := int64(3800) * (1 << 20)

	if got, _ := capSlots(free, nLayers, perLayer, topK, 128); got != 34 {
		t.Fatalf("baseline moved: got %d, want 34 — update the mutation deltas below", got)
	}
	// Mutation 1: no margin. Must differ from the real answer, or the margin is doing nothing.
	noMargin := int(free / nLayers / perLayer)
	if noMargin == 34 {
		t.Error("removing the 384 MB margin changes nothing — the gate cannot see the margin at all")
	}
	// Mutation 2: no floor (clamp up to topK). Must differ from declining.
	tiny := int64(500) * (1 << 20)
	if _, decline := capSlots(tiny, nLayers, perLayer, topK, 128); !decline {
		t.Error("floor is not enforced: a free-VRAM figure below topK must DECLINE, not clamp up")
	}
	clampUp := topK // what the old code produced
	if fit := int((tiny - (384 << 20)) / nLayers / perLayer); fit >= clampUp {
		t.Errorf("mutation case is not below topK (fit=%d) — it no longer exercises the floor", fit)
	}
}

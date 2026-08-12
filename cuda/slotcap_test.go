//go:build cuda

package cuda

import "testing"

// The real 26B's per-slot buffer strides, in the order allocSlots allocates them:
// expGU.W, expGU.scales, expDown.W, expDown.scales — 123,904 x {16, 2, 8, 1} bytes. The 8:1 ratio
// between a packed-int4 W and its scale buffer is the layout (8 weights per uint32 against one f16
// scale per 32 weights); the 2:1 between expGU and expDown is expGU carrying twice the rows.
var strides26B = []int64{1982464, 247808, 991232, 123904} // sums to 3,345,408 per slot per layer

const (
	free26B    = 3847880704 // measured free VRAM at allocSlots on this card, real 26B
	nLayers26B = 30
	topK26B    = 8
)

// TestSlotCapArithmetic gates the expert-cache CAPPING branch — code that ships, decides how much
// VRAM to claim, and had never once been executed by a test.
//
// It could not be: the branch only binds when the requested slots exceed free VRAM, and no fixture
// is remotely large enough for that. It bound for the first time on the real 26B. That is the
// exercised-but-never-triggered shape — a branch inside well-tested code that the tests' inputs can
// never reach, so coverage tools report it green and it has never run.
//
// TWO CLAIMS THIS FILE USED TO MAKE, BOTH WITHDRAWN.
//
// It said it "corroborates the sizing", on the grounds that it predicted 34 and the hardware
// produced 34. It corroborated a PARALLEL COPY: allocSlots had its own inline arithmetic and this
// gate drove capSlots, so agreement between them showed only that two transcriptions of the same
// formula agreed. allocSlots now calls capSlots, so the gate points at the shipping path.
//
// And it said the agreement placed the 26B discrepancy "downstream of the sizing decision". It did
// not. The formula both copies implemented was WRONG: it summed requested bytes, while the driver
// charges each of the four buffers per layer its own whole 2 MiB quanta. 34 was the answer to the
// wrong question, and the forward at 34 slots generated zero tokens. The two copies agreeing was
// never evidence about the answer — only about the copying.
func TestSlotCapArithmetic(t *testing.T) {
	const gb = int64(1) << 30
	cases := []struct {
		name        string
		free        int64
		nLayers     int64
		strides     []int64
		topK, want  int
		wantDecline bool
	}{
		// The real 26B configuration. 33 is the largest count whose ROUNDED requirement plus the
		// margin fits: at 34 the ratio n*123904/2MiB crosses 2 and all four buffers tip a quantum at
		// once, a 4-quanta step, putting the requirement 203,816,960 B over free.
		{"26B: 128 requested, 3.8GB free", free26B, nLayers26B, strides26B, topK26B, 33, false},
		// Ample VRAM: the request is honoured untouched, so the branch must NOT bind.
		{"ample free: no cap", 40 * gb, nLayers26B, strides26B, topK26B, 128, false},
		// FLOOR: not even topK fits, so it must decline rather than clamp up and allocate.
		{"below topK: declines", 500 * (1 << 20), nLayers26B, strides26B, topK26B, 0, true},
		// Exactly topK fits: the boundary must admit, not decline (off-by-one both ways). Derived
		// from the same rounding form rather than from a raw sum, or the boundary is off by the
		// rounding this gate exists to enforce.
		{"exactly topK fits", slotRequirement(topK26B, nLayers26B, strides26B) + slotMarginBytes,
			nLayers26B, strides26B, topK26B, topK26B, false},
		// One byte short of the topK boundary must decline. This is the assertion that catches a
		// `<=` / `<` slip, which no other case here can distinguish.
		{"one byte below topK: declines", slotRequirement(topK26B, nLayers26B, strides26B) + slotMarginBytes - 1,
			nLayers26B, strides26B, topK26B, 0, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, decline := capSlots(c.free, c.nLayers, c.strides, c.topK, 128)
			if decline != c.wantDecline {
				t.Fatalf("decline = %v, want %v (got %d slots)", decline, c.wantDecline, got)
			}
			if !c.wantDecline && got != c.want {
				t.Errorf("slots = %d, want %d", got, c.want)
			}
		})
	}
}

// TestSlotCapArithmetic_search asserts the property that makes this a search rather than a division,
// and that the search disagrees with the division exactly where it mattered.
func TestSlotCapArithmetic_search(t *testing.T) {
	// 1. The fix changes the answer at the configuration that failed. A division over the raw sum
	//    returns 34; the search returns 33; the hardware generates zero tokens at 34 and four at 33.
	//    Without this assertion the gate would pass just as happily against the old arithmetic.
	perLayerRaw := int64(0)
	for _, p := range strides26B {
		perLayerRaw += p
	}
	division := int((free26B - slotMarginBytes) / nLayers26B / perLayerRaw)
	search, _ := capSlots(free26B, nLayers26B, strides26B, topK26B, 128)
	if division != 34 {
		t.Errorf("the raw-sum division no longer returns 34 (got %d) — the historical figures in "+
			"docs/QUEUE.md A1 were derived from it and need re-checking", division)
	}
	if search != 33 {
		t.Fatalf("the search returns %d, want 33", search)
	}
	if search == division {
		t.Error("search and division agree at the 26B configuration, so this gate cannot tell the " +
			"fix from the bug it replaced")
	}

	// 2. The requirement must be MONOTONE non-decreasing in n. The binary search is only valid if it
	//    is, and monotonicity is a property of the rounding form rather than an obvious fact — a
	//    per-buffer floor, or a stride that shrank with n, would break it silently.
	prev := int64(-1)
	for n := 0; n <= 200; n++ {
		got := slotRequirement(n, nLayers26B, strides26B)
		if got < prev {
			t.Fatalf("requirement is not monotone: n=%d needs %d, less than n=%d's %d — the binary "+
				"search in capSlots is invalid", n, got, n-1, prev)
		}
		prev = got
	}

	// 3. The step at 34 is real and is four quanta, which is why a division plus a correction term
	//    cannot work: the error is not a constant offset, it is concentrated at boundaries.
	step := slotRequirement(34, nLayers26B, strides26B) - slotRequirement(33, nLayers26B, strides26B)
	if want := int64(4 * nLayers26B * allocQuantumBytes); step != want {
		t.Errorf("the 33->34 step is %d B, expected %d (4 quanta x %d layers) — the boundary this "+
			"whole item turns on has moved", step, want, nLayers26B)
	}

	// 4. capSlots must never return a count whose requirement does not actually fit. This is the
	//    property, stated independently of any particular number, so it survives the constants above
	//    changing.
	for _, free := range []int64{free26B, 1 << 30, 2 << 30, 6 << 30, 40 << 30} {
		n, decline := capSlots(free, nLayers26B, strides26B, topK26B, 128)
		if decline {
			continue
		}
		if req := slotRequirement(n, nLayers26B, strides26B); req+slotMarginBytes > free {
			t.Errorf("free=%d: capSlots returned %d slots needing %d B + %d margin > %d free",
				free, n, req, int64(slotMarginBytes), free)
		}
		if n < 128 {
			if req := slotRequirement(n+1, nLayers26B, strides26B); req+slotMarginBytes <= free {
				t.Errorf("free=%d: capSlots returned %d but %d also fits — the search is not "+
					"returning the LARGEST admissible count", free, n, n+1)
			}
		}
	}
}

// TestSlotCapArithmetic_mutation is the falsifiability check the policy requires of any gate:
// perturb the thing under test and confirm the gate goes red. Mutations, and what each breaks:
//
//	rounding removed (raw sum) -> 34 slots, not 33. That is the ORIGINAL DEFECT, and the forward at
//	                              34 generates zero tokens on the real 26B.
//	margin removed             -> 37 slots. DERIVED under the ROUNDING form, and worth stating how
//	                              it was got wrong first: 38 was carried over from the old raw-sum
//	                              derivation without re-deriving, and this gate caught it. With
//	                              x = n*123904/2MiB, n=37 gives quanta 3+5+18+35 = 61, so
//	                              30 x 61 x 2 MiB = 3,837,788,160 <= 3,847,880,704 free; n=38 gives
//	                              3+5+18+36 = 62, so 3,900,702,720 > free. The margin costs 4 slots
//	                              (33 -> 37), not the 4 the old sum happened to give (34 -> 38) —
//	                              same delta, different endpoints, and only one of them is real.
//	floor removed (clamp up)   -> "below topK" returns 8 slots instead of declining: the
//	                              allocate-then-fail-to-launch path.
//
// Each is asserted directly rather than described, so the claim "this gate can fail" is itself
// checked.
func TestSlotCapArithmetic_mutation(t *testing.T) {
	base, _ := capSlots(free26B, nLayers26B, strides26B, topK26B, 128)
	if base != 33 {
		t.Fatalf("baseline moved: got %d, want 33 — update the mutation deltas below", base)
	}

	// Mutation 1: sum the requested bytes instead of rounding each buffer. This is the defect.
	perLayerRaw := int64(0)
	for _, p := range strides26B {
		perLayerRaw += p
	}
	if noRound := int((free26B - slotMarginBytes) / nLayers26B / perLayerRaw); noRound == base {
		t.Error("dropping per-buffer quantum rounding changes nothing — the gate cannot see the " +
			"rounding, which is the entire content of the fix")
	}

	// Mutation 2: no margin. Must differ, or the margin is doing nothing.
	noMargin := 0
	for n := 128; n >= 0; n-- {
		if slotRequirement(n, nLayers26B, strides26B) <= free26B {
			noMargin = n
			break
		}
	}
	if noMargin == base {
		t.Error("removing the margin changes nothing — the gate cannot see the margin at all")
	}
	if noMargin != 37 {
		t.Errorf("removing the margin gives %d slots, expected 37 — the documented mutation delta "+
			"is a number that must be derived, not asserted", noMargin)
	}

	// Mutation 3: no floor (clamp up to topK). Must differ from declining.
	tiny := int64(500) * (1 << 20)
	if _, decline := capSlots(tiny, nLayers26B, strides26B, topK26B, 128); !decline {
		t.Error("floor is not enforced: a free-VRAM figure below topK must DECLINE, not clamp up")
	}
	if fit, _ := capSlots(tiny, nLayers26B, strides26B, topK26B, 128); fit >= topK26B {
		t.Errorf("mutation case is not below topK (fit=%d) — it no longer exercises the floor", fit)
	}
}

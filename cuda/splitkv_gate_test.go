//go:build cuda

package cuda

import "testing"

// TestSplitKVGate_measuredGeometries pins the split-KV selection decision for the four geometries
// characterized in P6a (docs/benchmarks.md §B6). This is a UNIT test of the gate arithmetic — no GPU,
// no model — so it runs everywhere and guards the one thing that is expensive to rediscover: which
// geometry/depth combinations may take the split path.
//
// This test exists because the gate was previously a single constant (splitkvMinKeys = 256)
// characterized on ONE geometry (qwen2.5-1.5b) by a tight in-process loop, then applied to all
// models. e2e measurement showed that constant regressed three of four geometries by up to 18–25%,
// and was wrong on its own geometry too. A future simplification back to one constant MUST fail here.
func TestSplitKVGate_measuredGeometries(t *testing.T) {
	// nWin is the EFFECTIVE attended span (window-clamped), not the raw position.
	cases := []struct {
		name    string
		nH, hd  int
		nWin    int
		wantSK  bool
		measure string
	}{
		// qwen2.5-0.5b (nH=14, hd=64): OFF wins 256–2048 (worst 0.819 at 512), ON only at 3900.
		{"qwen0.5b@256", 14, 64, 256, false, "0.839 — the 18% regression band the old gate fired in"},
		{"qwen0.5b@512", 14, 64, 512, false, "0.819 — worst cell on this geometry (−18%)"},
		{"qwen0.5b@2048", 14, 64, 2048, false, "0.955"},
		{"qwen0.5b@3900", 14, 64, 3900, true, "1.197"},

		// qwen2.5-1.5b (nH=12, hd=128): the geometry the OLD constant was characterized on. It loses
		// at 256 and 512 — the old "break-even at 256, clear win from 384+" is refuted here.
		{"qwen1.5b@256", 12, 128, 256, false, "0.941 — old gate fired; refutes 'break-even at 256'"},
		{"qwen1.5b@512", 12, 128, 512, false, "0.939 — refutes 'clear win from 384+'"},
		{"qwen1.5b@1024", 12, 128, 1024, true, "1.078"},
		{"qwen1.5b@2048", 12, 128, 2048, true, "1.191"},

		// phi3-mini (nH=32, MHA): NEVER crosses over — the ratio declines monotonically with depth
		// (0.993 → 0.969 → 0.919 → 0.815 → 0.754). No threshold in nWin can express this: a formula
		// gate has the form "ON iff nWin ≥ f(geometry)", which ALWAYS predicts ON wins at sufficient
		// depth. phi3 falsifies that FORM, not just its constants — hence a lookup with a "never" class.
		{"phi3@256", 32, 96, 256, false, "0.993"},
		{"phi3@2048", 32, 96, 2048, false, "0.815 — −19%"},
		{"phi3@3900", 32, 96, 3900, false, "0.754 — worst cell measured (−25%); still declining"},

		// gemma3-1b (nH=4, hd=256, window=512): its windowed layers cap at nWin=512 forever. Gating on
		// nWin (not position) is what keeps them on the single-block path at every depth.
		{"gemma3-windowed-layer@512", 4, 256, 512, false, "windowed layer never exceeds its 512 span"},
	}
	for _, c := range cases {
		got := c.nWin >= splitkvThreshold(c.nH, c.hd)
		if got != c.wantSK {
			t.Errorf("%s (nH=%d hd=%d nWin=%d): split-KV selected=%v, want %v (measured ON/OFF %s)",
				c.name, c.nH, c.hd, c.nWin, got, c.wantSK, c.measure)
		}
	}
}

// TestSplitKVGate_neverClassIsUnreachable guards the "never" encoding: splitkvNever must exceed any
// nWin the resident can produce, so the plain >= comparison at the gate can never accidentally admit
// a high-head geometry at extreme depth.
func TestSplitKVGate_neverClassIsUnreachable(t *testing.T) {
	if got := splitkvThreshold(splitkvMaxHeads, 128); got != splitkvNever {
		t.Fatalf("nH=%d should be the never class, got threshold %d", splitkvMaxHeads, got)
	}
	if splitkvNever <= cudaCtxCap {
		t.Fatalf("splitkvNever (%d) must exceed cudaCtxCap (%d) or the never class is reachable",
			splitkvNever, cudaCtxCap)
	}
}

// TestSplitKVGate_conservativeDefault pins the asymmetric-loss policy for UNMEASURED geometries:
// firing early costs up to 18–25%, firing late costs a few percent, so an unknown geometry must not
// take the split path in the shallow band where every measured geometry regressed.
func TestSplitKVGate_conservativeDefault(t *testing.T) {
	// An unmeasured GQA-ish geometry.
	const nH, hd = 16, 128
	for _, nWin := range []int{256, 512, 1024, 2048} {
		if nWin >= splitkvThreshold(nH, hd) {
			t.Errorf("unmeasured geometry (nH=%d hd=%d) selected split-KV at nWin=%d; the default must "+
				"stay conservative through the band where all four measured geometries lost", nH, hd, nWin)
		}
	}
}

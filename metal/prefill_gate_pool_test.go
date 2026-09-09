//go:build darwin && goinfer_testhooks

package metal

import "testing"

// TestPoolCells pins §3.2's pooled arithmetic (prefill_gate_ref_test.go's poolCells) against
// hand-computed numbers on synthetic cells — no device, no checkpoint, pure logic — before
// trusting it with real compute (docs/task-prefill-gap.md §4 L1, 2026-09-09). The bug class this
// guards: a per-cell veto silently reappearing (the exact defect §3.2 found and removed from the
// 2026-09-05 form) would still compile and would still produce plausible-looking numbers.
func TestPoolCells(t *testing.T) {
	// Two cells, hand-chosen so every criterion's arithmetic is checkable by hand.
	cells := []*cellSummary{
		{
			K: 256, n: 1, contN: 10,
			exactHF: 2, fastHF: 2,
			exactMatch: 8, fastMatch: 8,
			d:          0,
			exactKLsum: 1.0, fastKLsum: 0.9,
			exactMeanKL: 0.10, fastMeanKL: 0.09,
			promptFastLowerKL: 1, promptsCounted: 1,
		},
		{
			K: 1024, n: 1, contN: 10,
			exactHF: 1, fastHF: 1,
			exactMatch: 9, fastMatch: 9,
			d:          0,
			exactKLsum: 0.5, fastKLsum: 0.4,
			exactMeanKL: 0.05, fastMeanKL: 0.04,
			promptFastLowerKL: 1, promptsCounted: 1,
		},
	}
	p := poolCells(cells)

	if p.n != 20 {
		t.Fatalf("pooled n = %d, want 20", p.n)
	}
	if p.exactHF != 3 || p.fastHF != 3 {
		t.Fatalf("pooled HF = exact %d fast %d, want 3/3", p.exactHF, p.fastHF)
	}
	// critA: fast(3) <= exact(3) + 2*sqrt(3) = 3 + 3.464 = 6.464 -> true
	if !p.critA {
		t.Error("critA = false, want true (3 <= 3 + 2*sqrt(3))")
	}
	// exactAgree = fastAgree = 17/20 = 0.85, d=0 -> critB trivially true (0.85 >= 0.85 - 0)
	if got, want := p.exactAgreeRate, 0.85; abs(got-want) > 1e-9 {
		t.Errorf("exactAgreeRate = %v, want %v", got, want)
	}
	if got, want := p.fastAgreeRate, 0.85; abs(got-want) > 1e-9 {
		t.Errorf("fastAgreeRate = %v, want %v", got, want)
	}
	if !p.critB {
		t.Error("critB = false, want true (agreement tied, d=0)")
	}
	// pooled KL: exact (1.0+0.5)/20 = 0.075, fast (0.9+0.4)/20 = 0.065 -> fast lower
	if got, want := p.exactMeanKL, 0.075; abs(got-want) > 1e-9 {
		t.Errorf("exactMeanKL = %v, want %v", got, want)
	}
	if got, want := p.fastMeanKL, 0.065; abs(got-want) > 1e-9 {
		t.Errorf("fastMeanKL = %v, want %v", got, want)
	}
	// both cells' fastMeanKL is under 1.1x their own exactMeanKL: 0.09<=0.11, 0.04<=0.055
	if !p.klCeilingOK {
		t.Error("klCeilingOK = false, want true (both cells within 1.1x)")
	}
	// fast lower on 2/2 prompts >= half
	if !p.critC {
		t.Error("critC = false, want true (pooled KL lower, all prompts, ceiling OK)")
	}

	// TestPoolCells_ceilingVetoesEvenWhenPooledMeanIsFine below covers the ceiling-alone failure
	// mode; this test's job is the equal/ship case above.
}

// TestPoolCells_ceilingVetoesEvenWhenPooledMeanIsFine proves the 1.1x ceiling is a REAL per-cell
// check, not folded into the pooled mean where one very-good cell could hide one bad one — exactly
// the shape §3's own "hard ceiling... in any single cell" wording exists to prevent.
func TestPoolCells_ceilingVetoesEvenWhenPooledMeanIsFine(t *testing.T) {
	cells := []*cellSummary{
		{ // one cell way over its own 1.1x ceiling
			K: 256, n: 1, contN: 10,
			exactHF: 0, fastHF: 0, exactMatch: 10, fastMatch: 10, d: 0,
			exactKLsum: 1.0, fastKLsum: 2.0, exactMeanKL: 0.10, fastMeanKL: 0.20, // fast is 2x exact here
			promptFastLowerKL: 0, promptsCounted: 1,
		},
		{ // a much bigger cell where fast is far better, dragging the POOLED mean below exact
			K: 1024, n: 1, contN: 1000,
			exactHF: 0, fastHF: 0, exactMatch: 1000, fastMatch: 1000, d: 0,
			exactKLsum: 100.0, fastKLsum: 1.0, exactMeanKL: 0.10, fastMeanKL: 0.001,
			promptFastLowerKL: 1, promptsCounted: 1,
		},
	}
	p := poolCells(cells)
	// pooled: exact (1+100)/1010 ≈ 0.1, fast (2+1)/1010 ≈ 0.003 -- pooled mean strongly favors fast
	if p.fastMeanKL >= p.exactMeanKL {
		t.Fatalf("test setup: expected pooled fast mean KL below exact (fast=%v exact=%v)", p.fastMeanKL, p.exactMeanKL)
	}
	if p.klCeilingOK {
		t.Fatal("klCeilingOK = true, want false — cell 1's fast (0.20) exceeds 1.1x its own exact (0.11), and the pooled mean must not hide that")
	}
	if p.critC {
		t.Error("critC = true, want false — the ceiling veto must still fail criterion C even though the pooled mean and prompt-sign parts both pass")
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

package decoder

import (
	"context"
	"sync/atomic"
	"testing"
)

// TestGenerate_residentPathAllocatesNoHostKV is P-01(a) (audit-2026-09-10): Model.Generate used
// to allocate the full host KV cache (m.NewCache) BEFORE the resBusy CAS even ran, so a
// resident-and-won call paid for capacity it never touched. NewCache is the sole place
// prefillEnters advances (R13's own discipline: "a test can OBSERVE that a check placed one line
// too late produces the identical error text" — same idea, applied to allocation instead of a
// refusal), so a zero delta here is a direct, non-inferred proof no host KV was allocated, not
// just that generation still produced the right tokens.
func TestGenerate_residentPathAllocatesNoHostKV(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}

	before := prefillEnters.Load()
	ch, g := m.Generate(context.Background(), []int{1, 2, 3}, 4, SamplingParams{Temperature: 0})
	for range ch {
	}
	if g.Err() != nil {
		t.Fatalf("Generate: %v", g.Err())
	}
	if delta := prefillEnters.Load() - before; delta != 0 {
		t.Errorf("prefillEnters advanced by %d on a resident-and-won Generate — the host KV cache "+
			"was still allocated even though this call never touches it", delta)
	}
}

// TestGenerate_casLoserStillCompletesOnCPU is P-01(a)'s other half: a call that LOSES the resBusy
// race must still lazily allocate exactly one CPU cache (not zero — it genuinely needs one now)
// and complete generation correctly on the CPU fallback. Pre-claims resBusy directly (the same
// primitive tryClaimResident itself uses) to force the loss deterministically, rather than trying
// to race two real goroutines against timing.
func TestGenerate_casLoserStillCompletesOnCPU(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}

	if !atomic.CompareAndSwapInt32(&m.resBusy, 0, 1) {
		t.Fatal("test setup: resBusy was already claimed before this test could claim it")
	}
	defer atomic.StoreInt32(&m.resBusy, 0)

	before := prefillEnters.Load()
	ch, g := m.Generate(context.Background(), []int{1, 2, 3}, 4, SamplingParams{Temperature: 0})
	var got []int
	for tok := range ch {
		got = append(got, tok)
	}
	if g.Err() != nil {
		t.Fatalf("Generate (CAS loser): %v", g.Err())
	}
	if len(got) == 0 {
		t.Fatal("CAS loser produced no tokens — the CPU fallback did not run to completion")
	}
	if delta := prefillEnters.Load() - before; delta != 1 {
		t.Errorf("prefillEnters advanced by %d, want exactly 1 — the CAS-loser branch must lazily "+
			"allocate its CPU cache exactly once, not zero (it genuinely needs one) and not more "+
			"than once (a repeated allocation mid-decode would be its own bug)", delta)
	}
}

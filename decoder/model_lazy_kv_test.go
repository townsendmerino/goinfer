package decoder

import (
	"context"
	"sync/atomic"
	"testing"
)

// TestGenerate_residentPathAllocatesNoHostKV is P-01(a) (audit-2026-09-10): a resident-and-won Model.Generate must not
// allocate the host KV cache (m.NewCache) at all; it once did so BEFORE the resBusy CAS ran. NewCache is the sole place
// prefillEnters advances, so a zero delta here is a direct proof that no host KV was allocated, not just that generation
// still produced the right tokens.
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

// TestGenerate_residentContextCapDeclinesToCPUInsteadOfErroring is M-01's decoder-seam gate (docs/audit-2026-09-10.md):
// a prompt in (ResidentContextCap, MaxPositions) must decline the resident path up front and fall through to the CPU
// path, not commit to a resident prefill that dies mid-write with no fallback (residentPrefillSeed's error only sets
// g.err). This is defense in depth: internal/serveapp's residentPath() (openai.go) is the primary guard, and this
// exercises the decoder package independent of any caller's enforcement.
func TestGenerate_residentContextCapDeclinesToCPUInsteadOfErroring(t *testing.T) {
	m, be := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}
	be.rf.capPos = 3
	prompt := []int{1, 2, 3, 4, 5} // longer than capPos, well under the tiny fixture's MaxPositions

	before := prefillEnters.Load()
	ch, g := m.Generate(context.Background(), prompt, 2, SamplingParams{Temperature: 0})
	var got []int
	for tok := range ch {
		got = append(got, tok)
	}
	if g.Err() != nil {
		t.Fatalf("Generate with a prompt past the resident cap: %v (want a clean CPU fallback, not an error)", g.Err())
	}
	if len(got) == 0 {
		t.Fatal("Generate produced no tokens — the CPU fallback did not run to completion")
	}
	if delta := prefillEnters.Load() - before; delta != 1 {
		t.Errorf("prefillEnters advanced by %d, want exactly 1 — the cap decline must fall through "+
			"to the CPU path (lazily allocating its cache), not attempt the resident prefill at all", delta)
	}
	if be.rf.forwards != 0 {
		t.Errorf("resident Forward called %d times, want 0 — the cap check must decline BEFORE "+
			"ever calling into the resident, not attempt it and recover from an error", be.rf.forwards)
	}
}

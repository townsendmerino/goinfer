package decoder

import (
	"context"
	"sync/atomic"
	"testing"
)

// A1 (docs/tasks/task-audit-followups-2026-10-06.md): every exclusive resident claim must refuse while an MC3
// generation holds a batch place. MC3's own claim reads resBusy but never sets it, so a bare CAS on resBusy succeeds
// mid-batch; these tests go through the two callers that used one (GenerateSpeculative and BlockSpec.generate), not
// through tryClaimResident, whose own test passing is what let the callers drift.

// TestBlockSpecGenerate_declinesWhileAnMC3HolderIsActive: with one MC3 place held and resBusy still 0, block-spec
// generation returns ErrBlockSpecResidentBusy before any device call.
func TestBlockSpecGenerate_declinesWhileAnMC3HolderIsActive(t *testing.T) {
	host := &blockspecStubHost{}
	m := &Model{resident: host, resIDs: []int{1, 2, 3}}
	m.batcher = &residentBatcher{maxHolders: 2}
	m.batcher.init()
	if !m.batcher.claim(&m.resBusy) {
		t.Fatal("setup: the MC3 claim failed")
	}
	s := &BlockSpec{m: m, host: host, dw: blockspecStubWeights{blockSize: 8}}
	_, _, err := s.generate([]int{1, 2, 3}, BlockSpecOptions{VerifyWidth: 4}, nil)
	if !ErrBlockSpecResidentBusy(err) {
		t.Fatalf("generate() with an MC3 holder active: err = %v, want ErrBlockSpecResidentBusy", err)
	}
	if n := atomic.LoadInt32(&host.setBatchedCaptureCalls); n != 0 {
		t.Errorf("SetBatchedCapture called %d times while an MC3 generation held the resident", n)
	}
	if atomic.LoadInt32(&m.resBusy) != 0 {
		t.Error("resBusy changed by a call that should have declined")
	}
	m.batcher.release()
}

// TestGenerateSpeculative_leavesTheResidentToAnMC3Holder: with one MC3 place held on the target, speculative decoding
// runs on the CPU (the existing loser's fallback) and never calls the target's resident.
func TestGenerateSpeculative_leavesTheResidentToAnMC3Holder(t *testing.T) {
	m, rf := loadWithMC3Fake(t, 2)
	if m.EnableResidentConcurrency(2) < 2 {
		t.Skip("the fake resident did not enable MC3")
	}
	draft, err := Load(tinyFixture(t), Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Close()
	if !m.batcher.claim(&m.resBusy) {
		t.Fatal("setup: the MC3 claim failed")
	}
	before := rf.soloFwds
	ch, g, err := m.GenerateSpeculative(context.Background(), []int{1, 2, 3, 4}, 8, draft, 2, SamplingParams{})
	if err != nil {
		t.Fatalf("GenerateSpeculative: %v", err)
	}
	n := 0
	for range ch {
		n++
	}
	if g.Err() != nil {
		t.Fatalf("generation: %v", g.Err())
	}
	if n == 0 {
		t.Fatal("no tokens generated")
	}
	if rf.soloFwds != before {
		t.Errorf("the target's resident ran %d forwards while an MC3 generation held a place", rf.soloFwds-before)
	}
	m.batcher.release()
}

package decoder

import "testing"

// blockspecReuseStubHost is a ResidentDrafterHost that records every PrefillLastNArgmax call's
// startPos and embedding count, and always answers with a fixed anchor — enough to drive
// generate() through its seed step and straight to the fully-completed exit with
// opt.MaxTokens == 1 (the round loop never runs), matching blockspecCommitStubHost's shape but
// tracking calls instead of assuming there is only one.
type blockspecReuseStubHost struct {
	anchor    int
	seedCalls []struct{ startPos, n int }
}

func (h *blockspecReuseStubHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	panic("not reached")
}
func (h *blockspecReuseStubHost) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	h.seedCalls = append(h.seedCalls, struct{ startPos, n int }{startPos, len(embeddings)})
	return []int{h.anchor}, nil
}
func (h *blockspecReuseStubHost) SetBatchedCapture(taps []int) error { return nil }
func (h *blockspecReuseStubHost) BatchedCapture() [][]float32        { return nil }

var _ ResidentDrafterHost = (*blockspecReuseStubHost)(nil)

// blockspecReuseStubDrafter records TruncateContext/FuseContext calls instead of assuming a
// fixed argument like blockspecCommitStubDrafter does — this test drives generate() TWICE on the
// SAME instance and needs to observe a NONZERO truncate on the second call.
type blockspecReuseStubDrafter struct {
	truncateCalls []int
	fuseRowCounts []int
}

func (d *blockspecReuseStubDrafter) FuseContext(rows [][]float32) ([][]float32, error) {
	d.fuseRowCounts = append(d.fuseRowCounts, len(rows))
	return rows, nil
}
func (d *blockspecReuseStubDrafter) ExtendContext(fused [][]float32) error { return nil }
func (d *blockspecReuseStubDrafter) ContextLen() int                       { panic("not reached") }
func (d *blockspecReuseStubDrafter) TruncateContext(n int) {
	d.truncateCalls = append(d.truncateCalls, n)
}
func (d *blockspecReuseStubDrafter) DraftBlock(blockIn [][]float32) ([][]float32, error) {
	panic("not reached")
}
func (d *blockspecReuseStubDrafter) DraftTokens(trunk [][]float32) ([]int, error) {
	panic("not reached")
}

var _ ResidentBlockDrafter = (*blockspecReuseStubDrafter)(nil)

// TestBlockSpecGenerate_reusesDrafterContextOnExtension is P-05's deferred half (audit-2026-09-10):
// a SECOND generate() call on the SAME *BlockSpec instance, whose prompt is a strict extension of
// the first call's committed sequence, must reuse both the target's resident KV (reuseFrom > 0)
// AND the drafter's own context (TruncateContext(reuseFrom), not TruncateContext(0)) — seeding
// and fusing only the new suffix, not re-embedding the whole prompt.
func TestBlockSpecGenerate_reusesDrafterContextOnExtension(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}

	host := &blockspecReuseStubHost{anchor: 7}
	drafter := &blockspecReuseStubDrafter{}
	s := &BlockSpec{m: m, host: host, rd: drafter, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}

	prompt1 := []int{1, 2, 3}
	out1, _, err := s.generate(prompt1, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 1}, nil)
	if err != nil {
		t.Fatalf("first generate: %v", err)
	}
	committed := append(append([]int{}, prompt1...), out1...) // [1,2,3,7]

	// Second call: an extension by exactly one new token — the shape a real agent turn takes.
	prompt2 := append(append([]int{}, committed...), 5)
	if _, _, err := s.generate(prompt2, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 1}, nil); err != nil {
		t.Fatalf("second generate: %v", err)
	}

	if len(host.seedCalls) != 2 {
		t.Fatalf("seedCalls = %v, want 2 entries", host.seedCalls)
	}
	first, second := host.seedCalls[0], host.seedCalls[1]
	if first.startPos != 0 || first.n != len(prompt1) {
		t.Errorf("first seed call: startPos=%d n=%d, want startPos=0 n=%d (cold, whole prompt)",
			first.startPos, first.n, len(prompt1))
	}
	wantReuseFrom := len(committed) // == len(prompt2) - 1
	if second.startPos != wantReuseFrom || second.n != 1 {
		t.Errorf("second seed call: startPos=%d n=%d, want startPos=%d n=1 — only the new suffix "+
			"token should be embedded/seeded, not the whole prompt again",
			second.startPos, second.n, wantReuseFrom)
	}

	if len(drafter.truncateCalls) != 2 {
		t.Fatalf("truncateCalls = %v, want 2 entries", drafter.truncateCalls)
	}
	if drafter.truncateCalls[0] != 0 {
		t.Errorf("first TruncateContext = %d, want 0 (cold start)", drafter.truncateCalls[0])
	}
	if drafter.truncateCalls[1] != wantReuseFrom {
		t.Errorf("second TruncateContext = %d, want %d — the drafter's own context must be kept "+
			"for the reused prefix, not wiped to 0", drafter.truncateCalls[1], wantReuseFrom)
	}

	if len(drafter.fuseRowCounts) != 2 {
		t.Fatalf("fuseRowCounts = %v, want 2 entries", drafter.fuseRowCounts)
	}
	if drafter.fuseRowCounts[0] != len(prompt1) {
		t.Errorf("first fuse row count = %d, want %d (cold, whole prompt)", drafter.fuseRowCounts[0], len(prompt1))
	}
	if drafter.fuseRowCounts[1] != 1 {
		t.Errorf("second fuse row count = %d, want 1 — only the new suffix should be fused into "+
			"the drafter's context", drafter.fuseRowCounts[1])
	}
}

// TestBlockSpecGenerate_declinesDrafterReuseAfterOtherWriter is the safety half of P-05's
// deferred fix: resIDs matching alone is not enough to trust the drafter's own context — a
// DIFFERENT writer (a plain Generate turn, or another BlockSpec instance) can commit a
// token-identical resIDs without ever touching THIS BlockSpec's drafter, leaving rd's context
// stale relative to what resIDs now claims. Simulates that by committing resIDs directly
// (bypassing s.generate, exactly like a plain Generate turn would) between two calls into the
// SAME BlockSpec instance with an otherwise-reusable extension — the second call must still
// TruncateContext(0), not trust the mismatched drafter state.
func TestBlockSpecGenerate_declinesDrafterReuseAfterOtherWriter(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}

	host := &blockspecReuseStubHost{anchor: 7}
	drafter := &blockspecReuseStubDrafter{}
	s := &BlockSpec{m: m, host: host, rd: drafter, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}

	prompt1 := []int{1, 2, 3}
	out1, _, err := s.generate(prompt1, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 1}, nil)
	if err != nil {
		t.Fatalf("first generate: %v", err)
	}
	committed := append(append([]int{}, prompt1...), out1...)

	// A different writer (e.g. a plain Generate turn) commits the SAME ids — token-identical,
	// but this BlockSpec's own drafter never saw it.
	m.residentForgetIDs()
	m.residentCommitIDs(committed, nil, nil, nil)

	prompt2 := append(append([]int{}, committed...), 5)
	if _, _, err := s.generate(prompt2, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 1}, nil); err != nil {
		t.Fatalf("second generate: %v", err)
	}

	if len(drafter.truncateCalls) != 2 {
		t.Fatalf("truncateCalls = %v, want 2 entries", drafter.truncateCalls)
	}
	if drafter.truncateCalls[1] != 0 {
		t.Errorf("second TruncateContext = %d, want 0 — resIDs matched, but the last commit was a "+
			"DIFFERENT writer, so this drafter's own context must not be trusted", drafter.truncateCalls[1])
	}
	if len(host.seedCalls) != 2 || host.seedCalls[1].startPos != 0 || host.seedCalls[1].n != len(prompt2) {
		t.Errorf("second seed call = %+v, want startPos=0 n=%d (cold, whole prompt) after a "+
			"foreign commit", host.seedCalls[1], len(prompt2))
	}
}

package decoder

import "testing"

// blockspecCommitStubHost is a ResidentDrafterHost whose PrefillLastNArgmax always answers with a
// fixed anchor token, non-EOS by construction (the tiny fixture's own EOS set is empty) — enough
// to drive generate() through its SEED step and straight to the fully-completed exit with
// opt.MaxTokens == 1, never entering the round loop at all.
type blockspecCommitStubHost struct{ anchor int }

func (h *blockspecCommitStubHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	panic("not reached")
}
func (h *blockspecCommitStubHost) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	return []int{h.anchor}, nil
}
func (h *blockspecCommitStubHost) SetBatchedCapture(taps []int) error { return nil }
func (h *blockspecCommitStubHost) BatchedCapture() [][]float32        { return nil }

var _ ResidentDrafterHost = (*blockspecCommitStubHost)(nil)

// blockspecCommitStubDrafter answers the seed's own fuse() call (FuseContext + ExtendContext);
// every other method is unreached with opt.MaxTokens == 1 (the round loop never runs).
type blockspecCommitStubDrafter struct{}

func (blockspecCommitStubDrafter) FuseContext(rows [][]float32) ([][]float32, error) {
	return rows, nil
}
func (blockspecCommitStubDrafter) ExtendContext(fused [][]float32) error { return nil }
func (blockspecCommitStubDrafter) ContextLen() int                       { panic("not reached") }

// TruncateContext(0) is called unconditionally at the top of generate() ("fresh sequence: the
// previous generation's context must not leak in"), so it must be a no-op here, not a panic.
func (blockspecCommitStubDrafter) TruncateContext(n int) {
	if n != 0 {
		panic("not reached")
	}
}
func (blockspecCommitStubDrafter) DraftBlock(blockIn [][]float32) ([][]float32, error) {
	panic("not reached")
}
func (blockspecCommitStubDrafter) DraftTokens(trunk [][]float32) ([]int, error) { panic("not reached") }

var _ ResidentBlockDrafter = blockspecCommitStubDrafter{}

// blockspecCommitStubWeights adds a MaskTokenID to blockspecStubWeights's BlockSize — generate()
// reads MaskTokenID unconditionally right after the seed, before checking opt.MaxTokens at all.
type blockspecCommitStubWeights struct{ blockspecStubWeights }

func (blockspecCommitStubWeights) MaskTokenID() int { return 0 }

var _ BlockDrafterWeights = blockspecCommitStubWeights{}

// TestBlockSpecGenerate_commitsResIDsOnFullCompletion is P-05's blockspec.go half (audit-2026-09-10):
// BlockSpec.generate claimed resBusy and forgot resIDs (R-00) but never committed them back on a
// completed generation, so a --drafter turn always left the resident cache cold for whatever ran
// next (a plain Generate turn, or another BlockSpec turn). Uses a real loaded model (embedResident
// needs real weights) with a stubbed drafter host/trunk so the seed step runs for real and
// opt.MaxTokens == 1 ends the generation immediately after it, at the ONLY exit that commits.
func TestBlockSpecGenerate_commitsResIDsOnFullCompletion(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}

	host := &blockspecCommitStubHost{anchor: 7}
	s := &BlockSpec{m: m, host: host, rd: blockspecCommitStubDrafter{}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}

	prompt := []int{1, 2, 3}
	out, _, err := s.generate(prompt, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 1}, nil)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(out) != 1 || out[0] != 7 {
		t.Fatalf("out = %v, want the single seeded anchor [7] — test setup assumption broke", out)
	}

	want := append(append([]int{}, prompt...), out...)
	if !equalIntSlices(m.resIDs, want) {
		t.Errorf("resIDs = %v after a fully-completed generation, want %v — the completed exit must "+
			"commit prompt+generated exactly like generateInto's own natural-completion commit", m.resIDs, want)
	}
}

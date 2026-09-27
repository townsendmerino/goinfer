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

// blockspecWriteHost records every position the target is asked to write (a PrefillLastNArgmax call writes K/V at
// startPos..startPos+n-1) and answers every row with the same id, so a drafter that drafts that id is fully accepted.
type blockspecWriteHost struct {
	id      int
	written map[int]bool
}

func (h *blockspecWriteHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	panic("not reached")
}
func (h *blockspecWriteHost) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	ids := make([]int, len(embeddings))
	for i := range embeddings {
		h.written[startPos+i] = true
		ids[i] = h.id
	}
	return ids, nil
}
func (h *blockspecWriteHost) SetBatchedCapture(taps []int) error { return nil }
func (h *blockspecWriteHost) BatchedCapture() [][]float32        { return nil }

// blockspecEchoDrafter drafts `id` at every position of every block.
type blockspecEchoDrafter struct{ id int }

func (blockspecEchoDrafter) FuseContext(rows [][]float32) ([][]float32, error) { return rows, nil }
func (blockspecEchoDrafter) ExtendContext(fused [][]float32) error             { return nil }
func (blockspecEchoDrafter) ContextLen() int                                   { panic("not reached") }
func (blockspecEchoDrafter) TruncateContext(n int)                             {}
func (blockspecEchoDrafter) DraftBlock(blockIn [][]float32) ([][]float32, error) {
	return blockIn, nil
}
func (d blockspecEchoDrafter) DraftTokens(trunk [][]float32) ([]int, error) {
	ids := make([]int, len(trunk))
	for i := range ids {
		ids[i] = d.id
	}
	return ids, nil
}

// TestBlockSpecGenerate_commitsOnlyWrittenPositions: a completed block-drafter generation may record in resIDs only
// positions whose K/V the target actually wrote, since the next turn reuses them as they stand. The token a round (or
// the seed) ends on is the target's own output, emitted but not yet forwarded: it becomes the next round's anchor.
// Committing prompt+out at a max_tokens exit therefore claimed one position the cache never received — a rejected
// draft's K/V, or nothing at all — and the next turn attended to it. Plain decode forwards every token it emits, so
// the fix forwards that trailing token before committing, and the commit is prompt+out exactly as plain decode's.
// Covered: the seed-only exit (MaxTokens 1), and exits after one and after two full rounds.
func TestBlockSpecGenerate_commitsOnlyWrittenPositions(t *testing.T) {
	for _, maxTok := range []int{1, 5, 9} {
		m, _ := loadWithFakeResident(t)
		if !m.ResidentActive() {
			t.Skip("fixture is not resident-eligible")
		}
		host := &blockspecWriteHost{id: 7, written: map[int]bool{}}
		s := &BlockSpec{m: m, host: host, rd: blockspecEchoDrafter{id: 7}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}
		prompt := []int{1, 2, 3}
		out, rounds, err := s.generate(prompt, BlockSpecOptions{VerifyWidth: 4, MaxTokens: maxTok}, nil)
		if err != nil {
			t.Fatalf("maxTokens %d: generate: %v", maxTok, err)
		}
		if len(out) != maxTok {
			t.Fatalf("maxTokens %d: emitted %d tokens (%d rounds) — test setup assumption broke", maxTok, len(out), rounds)
		}
		want := append(append([]int{}, prompt...), out...)
		if !equalIntSlices(m.resIDs, want) {
			t.Errorf("maxTokens %d: resIDs = %v, want prompt+out %v, as plain decode commits", maxTok, m.resIDs, want)
		}
		for p := range m.resIDs {
			if !host.written[p] {
				t.Errorf("maxTokens %d (%d rounds): resIDs claims position %d (token %d) but the target never wrote its K/V",
					maxTok, rounds, p, m.resIDs[p])
			}
		}
	}
}

package decoder

import (
	"context"
	"testing"
)

// widthHost is a dense drafter host that records the rows of every batched call and answers 7 for each.
type widthHost struct{ calls []int }

func (h *widthHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	panic("not reached")
}
func (h *widthHost) PrefillLastNArgmax(embs [][]float32, startPos int) ([]int, error) {
	h.calls = append(h.calls, len(embs))
	ids := make([]int, len(embs))
	for i := range ids {
		ids[i] = 7
	}
	return ids, nil
}
func (h *widthHost) SetBatchedCapture([]int) error { return nil }
func (h *widthHost) BatchedCapture() [][]float32   { return nil }

// TestBlockSpec_setVerifyWidthReachesGenerateStream: the width serve sets (-drafter-width) is the one GenerateStream verifies at.
// A drafter that drafts the target's own token is fully accepted, so every round verifies the full width: the seed's call
// aside, every batched call is 4 rows.
func TestBlockSpec_setVerifyWidthReachesGenerateStream(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Fatal("the fake-resident fixture is not resident-eligible, so this test cannot reach the batched verify path")
	}
	host := &widthHost{}
	s := &BlockSpec{m: m, host: host, rd: blockspecEchoDrafter{id: 7}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 16}}}
	if err := s.SetVerifyWidth(4); err != nil {
		t.Fatal(err)
	}
	ch, gen, err := s.GenerateStream(context.Background(), []int{1, 2, 3}, 13, SamplingParams{})
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for range ch {
		n++
	}
	if gen.Err() != nil {
		t.Fatal(gen.Err())
	}
	if n != 13 {
		t.Fatalf("%d tokens, want 13", n)
	}
	// the seed (3 prompt rows), three rounds of 4 (1 + 3 x 4 = 13 tokens), and the trailing token's forward
	if want := []int{3, 4, 4, 4, 1}; !equalIntSlices(host.calls, want) {
		t.Errorf("batched calls of %v rows, want %v", host.calls, want)
	}
}

// TestBlockSpec_setVerifyWidthRefuses: a width under 2, and on a recurrent target a width past its fold path's rows.
func TestBlockSpec_setVerifyWidthRefuses(t *testing.T) {
	dense := &BlockSpec{m: &Model{w: &Weights{arch: &Architecture{}}}, dw: blockspecStubWeights{blockSize: 16}}
	for _, c := range []struct {
		w, inUse int // inUse 0: refused
	}{{0, 8}, {1, 0}, {-3, 0}, {2, 2}, {16, 16}, {64, 16}} { // the default is 8; past the block is capped by it
		dense.width = 0
		err := dense.SetVerifyWidth(c.w)
		if (err == nil) != (c.inUse > 0) {
			t.Errorf("dense, width %d: err %v, want ok %v", c.w, err, c.inUse > 0)
		}
		if c.inUse > 0 && dense.VerifyWidthInUse() != c.inUse {
			t.Errorf("dense, width %d: VerifyWidthInUse %d, want %d", c.w, dense.VerifyWidthInUse(), c.inUse)
		}
	}
	rec := &BlockSpec{m: &Model{w: &Weights{arch: deltaNetArch(8)}}, host: prepHost{recurrentSpecStub{rows: 8}, &blockspecWriteHost{}, new(int)},
		dw: blockspecStubWeights{blockSize: 16}}
	if err := rec.SetVerifyWidth(4); err != nil {
		t.Errorf("recurrent, width 4 of the fold's 8 rows: %v", err)
	}
	if err := rec.SetVerifyWidth(16); err == nil {
		t.Error("recurrent, width 16 past the fold's 8 rows: accepted")
	}
	if err := rec.SetVerifyWidth(0); err != nil { // the default, 8, fits the 8 rows
		t.Errorf("recurrent, the default width: %v", err)
	}
}

// TestBlockSpec_setVerifyWidthReachesGenerate: Generate with default options verifies at SetVerifyWidth's width too, not at 8.
func TestBlockSpec_setVerifyWidthReachesGenerate(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Fatal("the fake-resident fixture is not resident-eligible, so this test cannot reach the batched verify path")
	}
	host := &widthHost{}
	s := &BlockSpec{m: m, host: host, rd: blockspecEchoDrafter{id: 7}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 16}}}
	if err := s.SetVerifyWidth(4); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Generate([]int{1, 2, 3}, BlockSpecOptions{MaxTokens: 13}); err != nil {
		t.Fatal(err)
	}
	if want := []int{3, 4, 4, 4, 1}; !equalIntSlices(host.calls, want) {
		t.Errorf("batched calls of %v rows, want %v", host.calls, want)
	}
}

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
		t.Skip("fixture is not resident-eligible")
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
	for w, ok := range map[int]bool{0: true, 1: false, -3: false, 2: true, 16: true, 64: true} { // past the block is capped by it
		if err := dense.SetVerifyWidth(w); (err == nil) != ok {
			t.Errorf("dense, width %d: err %v, want ok %v", w, err, ok)
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

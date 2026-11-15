package decoder

import (
	"errors"
	"testing"
)

// splitResident is a drafter-hosting resident built as the real backends build theirs: it is the ResidentSubsystemProvider, and the
// subsystems it hands out are separate values that are NOT providers themselves. The other fakes in this package return themselves
// from ResidentSubsystems (selfSubsystems, runtimeSubsystems), so a call that hands a subsystem to SubsystemsOf resolves right on
// them and to the no-op subsystems on a backend.
type splitResident struct {
	prepared int
	seeds    int
}

func (f *splitResident) Forward([]float32, int) ([]float32, error)      { return nil, nil }
func (f *splitResident) ForwardN([][]float32, int) ([][]float32, error) { return nil, nil }
func (f *splitResident) UploadKV(int, int, []float32, []float32) error  { return nil }
func (f *splitResident) TruncateTo(int)                                 {}
func (f *splitResident) Reset()                                         {}
func (f *splitResident) Close() error                                   { return nil }

func (f *splitResident) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	return blockspecEchoDrafter{id: 7}, nil
}
func (f *splitResident) PrefillLastNArgmax(e [][]float32, _ int) ([]int, error) {
	return nil, errors.New("splitResident: PrefillLastNArgmax: the seed took the logits-free path's fallback")
}
func (f *splitResident) SetBatchedCapture([]int) error { return nil }
func (f *splitResident) BatchedCapture() [][]float32   { return nil }

func (f *splitResident) ResidentSubsystems() ResidentSubsystems {
	out := noopResidentSubsystems
	out.Runtime = splitRuntime{r: f}
	return out
}

// splitRuntime serves the drafter-host, seed and fold capabilities of a recurrent target.
type splitRuntime struct {
	NoopResidentRuntime
	r *splitResident
}

func (s splitRuntime) DrafterHostAvailable() bool   { return true }
func (s splitRuntime) SeedArgmaxAvailable() bool    { return true }
func (s splitRuntime) RecurrentSpecAvailable() bool { return true }
func (s splitRuntime) RecurrentSpecRows() int       { return RecurrentSpecMaxRows }
func (s splitRuntime) PrepareRecurrentSpec(int) error {
	s.r.prepared++
	return nil
}
func (s splitRuntime) PrefillSeedArgmax([][]float32, int) (int, error) {
	s.r.seeds++
	return 42, nil
}
func (s splitRuntime) AttachBlockDrafter(w BlockDrafterWeights) (ResidentBlockDrafter, error) {
	return s.r.AttachBlockDrafter(w)
}

var (
	_ ResidentForward           = (*splitResident)(nil)
	_ ResidentSubsystemProvider = (*splitResident)(nil)
	_ ResidentDrafterHost       = (*splitResident)(nil)
)

// TestBlockSpec_hostIsTheResident: a BlockSpec on a recurrent target resolves its fold path, its seed and its step shapes from the
// resident's subsystems, not from the value it was built from. A BlockSpec whose host was a subsystem declined every recurrent
// model in production (no RecurrentSpecAvailable) and fell back from the logits-free seed.
func TestBlockSpec_hostIsTheResident(t *testing.T) {
	res := &splitResident{}
	m := &Model{w: &Weights{arch: deltaNetArch(8)}, resident: res}
	s, err := m.newBlockSpec(blockspecStubWeights{blockSize: 16}, nil)
	if err != nil {
		t.Fatalf("newBlockSpec on a recurrent target whose resident serves the fold path: %v", err)
	}
	if res.prepared != 1 {
		t.Errorf("the resident was asked to prepare its fold path %d times, want 1", res.prepared)
	}
	if s.recurrentFold(8) == nil {
		t.Error("recurrentFold(8) is nil: the fold path the resident serves was not found")
	}
	id, err := s.seedAnchor([][]float32{{1}}, 0)
	if err != nil || id != 42 || res.seeds != 1 {
		t.Errorf("seedAnchor = (%d, %v) after %d logits-free seeds, want (42, nil) after 1", id, err, res.seeds)
	}
}

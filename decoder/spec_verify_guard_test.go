package decoder

import (
	"context"
	"errors"
	"testing"
)

// errFakeDivergence is what the diverging fake resident reports. Every assertion below checks for
// THIS error by identity (errors.Is), never by message, so a refusal caused by some OTHER
// precondition — rollback safety, an empty prompt, an unsupported host — cannot pass for this one.
var errFakeDivergence = errors.New("fake resident: decode and batched verify use different trees")

// divergingResident is the resident seam's fakeResident plus the optional DecodeVerifyDiverger
// method, switchable so the SAME model can be driven with the divergence on and off.
type divergingResident struct {
	*fakeResident
	diverge bool
}

func (d *divergingResident) DecodeVerifyDivergence() error {
	if d.diverge {
		return errFakeDivergence
	}
	return nil
}

type divergingResidencyBackend struct {
	Backend
	rf *divergingResident
}

func (b *divergingResidencyBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &divergingResident{fakeResident: &fakeResident{vocab: vocab}}
	return b.rf, true, nil
}

func (b *divergingResidencyBackend) Close() error { return nil }

func loadWithDivergingResident(t *testing.T) (*Model, *divergingResident) {
	t.Helper()
	be := &divergingResidencyBackend{}
	name := "fake-resident-diverging"
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: name})
	if err != nil {
		t.Fatalf("Load with diverging fake resident: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if be.rf == nil || !m.ResidentActive() {
		t.Fatal("the diverging fake resident was not installed — every assertion below would be " +
			"testing a model with no resident, where SpecDecodeConflict is nil by construction")
	}
	return m, be.rf
}

// drain consumes a generation's stream so a control-arm run that DID start does not leak its
// goroutine into the next assertion.
func drain(ch <-chan int) {
	if ch == nil {
		return
	}
	for range ch {
	}
}

// TestSpecDecodeConflict_refusesEveryResidentVerifyEntry drives the three speculative entry points
// that verify on the resident — not the predicate alone. CLAUDE.md: a component whose contract
// depends on WHERE it is called must be tested through its callers, or the test only asserts the
// predicate works when someone remembers to ask it.
func TestSpecDecodeConflict_refusesEveryResidentVerifyEntry(t *testing.T) {
	m, rf := loadWithDivergingResident(t)
	ctx := context.Background()
	prompt := []int{1, 2, 3, 4, 5}
	greedy := SamplingParams{}

	// Precondition the refusal must not be confused with: if the tiny fixture were rollback-UNsafe,
	// GenerateSpeculative would refuse for that reason first and a sloppier assertion would pass.
	if !m.specRollbackSafe() {
		t.Fatal("fixture is not specRollbackSafe — GenerateSpeculative would refuse for a different " +
			"reason, so this test could not tell the two refusals apart")
	}

	rf.diverge = true

	if err := m.SpecDecodeConflict(); !errors.Is(err, errFakeDivergence) {
		t.Fatalf("SpecDecodeConflict = %v, want it to wrap the resident's divergence", err)
	}

	ch, _, err := m.GenerateNgramSpeculative(ctx, prompt, 4, &NgramDrafter{}, 4, greedy)
	drain(ch)
	if !errors.Is(err, errFakeDivergence) {
		t.Errorf("GenerateNgramSpeculative err = %v, want the divergence refusal — the n-gram "+
			"path verifies on the resident's batched tree while the prompt is seeded through decode", err)
	}

	ch, _, err = m.GenerateSpeculative(ctx, prompt, 4, m, 4, greedy)
	drain(ch)
	if !errors.Is(err, errFakeDivergence) {
		t.Errorf("GenerateSpeculative err = %v, want the divergence refusal", err)
	}

	if _, err := m.NewBlockSpec(nil, nil); !errors.Is(err, errFakeDivergence) {
		t.Errorf("NewBlockSpec err = %v, want the divergence refusal BEFORE any drafter is attached", err)
	}

	// CONTROL — the same model with the divergence off. Without this arm every assertion above could
	// be passing because the entry points refuse for some unrelated reason on this fixture.
	rf.diverge = false

	if err := m.SpecDecodeConflict(); err != nil {
		t.Fatalf("control: SpecDecodeConflict = %v with no divergence, want nil", err)
	}
	ch, _, err = m.GenerateNgramSpeculative(ctx, prompt, 4, &NgramDrafter{}, 4, greedy)
	drain(ch)
	if errors.Is(err, errFakeDivergence) {
		t.Errorf("control: GenerateNgramSpeculative refused for divergence with none reported")
	}
	ch, _, err = m.GenerateSpeculative(ctx, prompt, 4, m, 4, greedy)
	drain(ch)
	if errors.Is(err, errFakeDivergence) {
		t.Errorf("control: GenerateSpeculative refused for divergence with none reported")
	}
	if _, err := m.NewBlockSpec(nil, nil); errors.Is(err, errFakeDivergence) {
		t.Errorf("control: NewBlockSpec refused for divergence with none reported (it may still " +
			"decline — this fake is no drafter host — but not for THIS reason)")
	}
}

// TestSpecDecodeConflict_sessionNgramIsNotRefused pins the deliberate EXCLUSION. A Session verifies
// on its CPU cache and never reaches the resident, so a resident divergence cannot affect it —
// refusing it would block serve's constrained/grammar requests for nothing. If someone later moves
// the check into validateNgramSpec (which Sessions share), this fails.
func TestSpecDecodeConflict_sessionNgramIsNotRefused(t *testing.T) {
	m, rf := loadWithDivergingResident(t)
	rf.diverge = true

	s := m.NewSession(0)
	ch, _, err := s.GenerateNgramSpeculative(context.Background(), []int{1, 2, 3, 4, 5}, 4, &NgramDrafter{}, 4, SamplingParams{})
	drain(ch)
	if errors.Is(err, errFakeDivergence) {
		t.Fatalf("Session n-gram refused for a RESIDENT divergence (%v), but a Session verifies on its "+
			"CPU cache and never touches the resident", err)
	}
}

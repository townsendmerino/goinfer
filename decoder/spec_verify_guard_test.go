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

// namedStubBackend is a minimal Backend whose only job is to report a Name() — SpecDecodeConflict's
// M-09 check (docs/audit-2026-09-10.md) keys on isWebGPUBackend(m.be.Name()), independent of
// m.resident, so unlike the resident-divergence tests above this needs no RegisterBackend/Load
// machinery at all: any Model with be set to this and quant set to "int4"/"int4mix" reproduces the
// staged condition directly.
type namedStubBackend struct{ name string }

func (b namedStubBackend) Name() string                               { return b.name }
func (b namedStubBackend) MatmulBT(a, bb, dst []float32, M, K, N int) {}
func (b namedStubBackend) Close() error                               { return nil }

// TestSpecDecodeConflict_refusesStagedWebGPUInt4 is M-09's own gate: gpu/backend.go's MatmulW4A8
// declines every M != 1, so a staged (non-resident) int4 model on webgpu decodes M=1 on the device
// and verifies M>1 on the CPU kernel — two different kernels, no measured tolerance between them.
// Mirrors TestSpecDecodeConflict_refusesEveryResidentVerifyEntry's discipline (CLAUDE.md: test
// through the callers, not the predicate alone) for the staged half of SpecDecodeConflict.
func TestSpecDecodeConflict_refusesStagedWebGPUInt4(t *testing.T) {
	m, err := Load(tinyFixture(t), Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if m.resident != nil {
		t.Fatal("fixture came up resident on a plain cpu Load — this test needs the staged path, " +
			"which only exists when m.resident is nil")
	}
	if !m.specRollbackSafe() {
		t.Fatal("fixture is not specRollbackSafe — every entry point would refuse for a different " +
			"reason, so this test could not tell the two refusals apart")
	}

	ctx := context.Background()
	prompt := []int{1, 2, 3, 4, 5}
	greedy := SamplingParams{}

	for _, quant := range []string{"int4", "int4mix"} {
		m.be = namedStubBackend{name: "webgpu"}
		m.quant = quant

		if err := m.SpecDecodeConflict(); err == nil {
			t.Errorf("quant=%s: SpecDecodeConflict = nil, want the staged webgpu int4 refusal", quant)
		}

		ch, _, err := m.GenerateNgramSpeculative(ctx, prompt, 4, &NgramDrafter{}, 4, greedy)
		drain(ch)
		if err == nil {
			t.Errorf("quant=%s: GenerateNgramSpeculative err = nil, want the staged refusal", quant)
		}

		ch, _, err = m.GenerateSpeculative(ctx, prompt, 4, m, 4, greedy)
		drain(ch)
		if err == nil {
			t.Errorf("quant=%s: GenerateSpeculative err = nil, want the staged refusal", quant)
		}

		// NewBlockSpec deliberately NOT checked here: it requires m.resident to implement
		// ResidentDrafterHost, so on this staged (m.resident == nil) model it already returns a
		// DIFFERENT, unrelated error (errBlockSpecUnsupported) regardless of this fix — asserting
		// "err != nil" there would pass in both the fixed and the reverted state, testing nothing.
		// Block drafting is resident-only by construction; M-09's staged path cannot reach it.
	}

	// CONTROL — the same model, same webgpu backend, a quant the M-09 gap does not apply to.
	// Without this arm every refusal above could be the entry point declining for some unrelated
	// reason on this fixture, not the staged-int4 check this test is actually about.
	m.be = namedStubBackend{name: "webgpu"}
	m.quant = "int8int8"
	if err := m.SpecDecodeConflict(); err != nil {
		t.Errorf("control (int8int8 on webgpu): SpecDecodeConflict = %v, want nil — M-09 is int4-specific", err)
	}

	// CONTROL — int4 quant, but NOT a webgpu backend (this repo's staged int4 CPU kernel is
	// M-independent on its own; M-09 is specifically about the webgpu MatmulW4A8 intercept).
	m.be = namedStubBackend{name: "cpu"}
	m.quant = "int4"
	if err := m.SpecDecodeConflict(); err != nil {
		t.Errorf("control (int4 on cpu): SpecDecodeConflict = %v, want nil — the CPU int4 kernel is M-independent on its own", err)
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

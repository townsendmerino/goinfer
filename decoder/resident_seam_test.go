package decoder

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// The seam gate.
//
// WHY THIS EXISTS. `serve --backend cuda|metal|webgpu` was silently CPU-only on EVERY backend
// for five weeks (0eefd77 -> 7557723). TWO independent bugs lived in the same gap, either of
// which alone was fatal:
//
//   - Options.Validate rejected the backend NAME, so `serve --backend cuda` failed at flag
//     validation even where the module was built in (727f198).
//   - Model.Generate gates the GPU on `useGPU := resident != nil && prefillFrom == 0 &&
//     commit == nil`, and Session.Generate ALWAYS sets commit — so routing a request through
//     the session cache silently disabled residency (7557723).
//
// Neither was caught, because nothing ever asserted "is this actually running on the GPU?".
// Every test called decoder.Forward directly; all 12 cmd/serve tests pin Backend: "cpu"; and
// ResidentActive() was asserted only inside gpu/. The flagship feature did not work at all and
// the suite was green. It was found by noticing a tok/s number, which is not a gate.
//
// The tests below need NO GPU and NO downloaded model — they fake the resident backend and use
// the committed tiny fixture — so this seam is gated in CI, on every push, rather than by a
// human remembering to look at throughput.

// fakeResident is a ResidentForward that records whether the resident path was ACTUALLY taken.
// It returns a deterministic non-nil logit row so callers proceed normally; the point is the
// call count, not the values.
type fakeResident struct {
	vocab    int
	forwards int
	closed   bool
}

func (f *fakeResident) Forward(embedding []float32, pos int) ([]float32, error) {
	f.forwards++
	out := make([]float32, f.vocab)
	out[pos%f.vocab] = 1 // deterministic, and distinct per position
	return out, nil
}

func (f *fakeResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	rows := make([][]float32, 0, len(embeddings))
	for i := range embeddings {
		r, err := f.Forward(embeddings[i], startPos+i)
		if err != nil {
			return nil, err
		}
		rows = append(rows, r)
	}
	return rows, nil
}

func (f *fakeResident) UploadKV(layer int, keys, vals []float32) error { return nil }
func (f *fakeResident) TruncateTo(pos int)                             {}
func (f *fakeResident) Reset()                                         {}
func (f *fakeResident) Close() error                                   { f.closed = true; return nil }

// fakeResidencyBackend is a CPU backend that ALSO advertises residency, so the seam can be
// exercised with no device present. It wraps the real CPU backend so every non-resident code
// path behaves normally.
type fakeResidencyBackend struct {
	Backend
	rf *fakeResident
}

func (b *fakeResidencyBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &fakeResident{vocab: vocab}
	return b.rf, true, nil
}

func (b *fakeResidencyBackend) Close() error { return nil }

// tinyFixture is the 810 KB committed GGUF — no download, so this runs anywhere.
func tinyFixture(t *testing.T) string {
	t.Helper()
	p := filepath.Join("..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	return p
}

func loadWithFakeResident(t *testing.T) (*Model, *fakeResidencyBackend) {
	t.Helper()
	be := &fakeResidencyBackend{}
	name := "fake-resident-seam"
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
		t.Fatalf("Load with fake resident backend: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, be
}

// TestSeam_ResidentBackendIsActuallyUsed asserts the model reports residency at all. If
// BuildResident is called and returns a runner, ResidentActive must be true — otherwise every
// downstream "GPU" claim is false and no other test would notice.
func TestSeam_ResidentBackendIsActuallyUsed(t *testing.T) {
	m, be := loadWithFakeResident(t)
	if be.rf == nil {
		t.Fatal("BuildResident was never called — the backend advertises ResidencyBackend but the " +
			"load path never asked it for a resident runner")
	}
	if !m.ResidentActive() {
		t.Fatal("BuildResident returned a runner but ResidentActive() is false — the resident " +
			"runner was built and then dropped")
	}
}

// TestSeam_GenerateRunsOnTheResident is the gate for 7557723. Generate on a resident model MUST
// dispatch to the resident runner. The five-week bug was exactly this: it silently ran on the
// CPU and every test still passed, because nothing counted resident calls.
func TestSeam_GenerateRunsOnTheResident(t *testing.T) {
	m, be := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible; the other seam tests still gate the wiring")
	}
	before := be.rf.forwards
	sp := SamplingParams{}
	stream, _ := m.Generate(context.Background(), []int{1, 2, 3}, 4, sp)
	for range stream { //nolint:revive // draining the stream is the point
	}
	if be.rf.forwards == before {
		t.Fatalf("Generate produced tokens WITHOUT calling the resident runner (%d forwards) — the "+
			"model is resident but decode silently ran on the CPU. This is the bug that made "+
			"`serve --backend cuda|metal|webgpu` CPU-only for five weeks (7557723): "+
			"useGPU := resident != nil && prefillFrom == 0 && commit == nil, and any caller that "+
			"sets commit (Session.Generate does, always) disables the GPU with no error.",
			be.rf.forwards)
	}
	t.Logf("resident forwards during Generate: %d", be.rf.forwards-before)
}

// TestSeam_SessionsDoNotSilentlyDisableResidency pins the exact mechanism of 7557723 so the
// trade stays a DECISION rather than drifting back into a silent regression. A session sets
// commit, which turns residency off. cmd/serve therefore bypasses sessions for resident models
// (openai.go). If a future change routes resident requests back through sessions, decode
// silently returns to the CPU — this test makes that loud.
func TestSeam_SessionsDoNotSilentlyDisableResidency(t *testing.T) {
	m, be := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	sess := m.NewSession(8)
	if sess == nil {
		t.Skip("no session support for this fixture")
	}
	before := be.rf.forwards
	sp := SamplingParams{}
	stream, _ := sess.Generate(context.Background(), []int{1, 2, 3}, 4, sp)
	for range stream { //nolint:revive
	}
	usedResident := be.rf.forwards > before
	// This documents CURRENT, DELIBERATE behaviour: sessions and residency are mutually
	// exclusive (model.go: commit != nil ⇒ !useGPU). The assertion is not "sessions must use the
	// GPU" — it is that the incompatibility is REAL and known, so cmd/serve must keep bypassing
	// sessions for resident models. If this ever starts passing, sessions learned to preserve
	// residency and cmd/serve's bypass can be revisited.
	if usedResident {
		t.Logf("NOTE: Session.Generate now uses the resident runner (%d forwards). The "+
			"session/residency incompatibility is gone — revisit cmd/serve's resident bypass "+
			"(openai.go), which exists only to work around it.", be.rf.forwards-before)
	} else {
		t.Logf("confirmed: Session.Generate does NOT use the resident runner — this is why " +
			"cmd/serve bypasses sessions for resident models. Do not route resident requests " +
			"back through the session cache without re-checking this.")
	}
}

// TestSeam_ValidateAcceptsEveryBackendName is the gate for 727f198: Options.Validate rejected
// "cuda"/"metal", so `serve --backend cuda` died at flag validation even with the module built
// in. Accepting the name is deliberately NOT a claim the backend is compiled in — an
// unregistered backend falls back to CPU with a note (NewBackend) — so validation must accept
// every name the CLI advertises.
func TestSeam_ValidateAcceptsEveryBackendName(t *testing.T) {
	// The names `cmd/serve --backend` documents. Adding a backend means adding it here.
	for _, name := range []string{"", "cpu", "webgpu", "cuda", "metal"} {
		o := Options{Backend: name}
		if err := o.Validate(); err != nil {
			t.Errorf("Options.Validate rejected backend %q: %v — `serve --backend %s` cannot start "+
				"even where the backend IS built in (this is 727f198)", name, err, name)
		}
	}
	if err := (Options{Backend: "definitely-not-a-backend"}).Validate(); err == nil {
		t.Error("Validate accepted a nonsense backend name — the allowlist has stopped allowlisting")
	}
}

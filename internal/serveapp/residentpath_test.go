package serveapp

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// fakeResident is a minimal decoder.ResidentForward + decoder.ResidentCapped, built only from
// decoder's exported API (mirroring decoder/resident_seam_test.go's own unexported fakeResident,
// which internal/serveapp cannot import) — enough to make ResidentActive()/ResidentContextCap()
// report real values without a GPU, so M-01's fix (docs/audit-2026-09-10.md) can be tested
// end-to-end instead of only by source-text assertion.
type fakeResident struct {
	vocab  int
	capPos int
}

func (f *fakeResident) Forward(embedding []float32, pos int) ([]float32, error) {
	if f.capPos > 0 && pos >= f.capPos {
		return nil, fmt.Errorf("fakeResident: pos %d >= context cap %d", pos, f.capPos)
	}
	out := make([]float32, f.vocab)
	out[pos%f.vocab] = 1
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
func (f *fakeResident) UploadKV(layer, base int, keys, vals []float32) error { return nil }
func (f *fakeResident) TruncateTo(pos int)                                   {}
func (f *fakeResident) Reset()                                               {}
func (f *fakeResident) Close() error                                         { return nil }
func (f *fakeResident) ContextCap() int                                      { return f.capPos }

// fakeResidencyBackend wraps the real cpu backend and additionally advertises residency, so
// decoder.Load builds a resident runner with no GPU present — the same seam
// decoder/resident_seam_test.go exercises from inside package decoder, reproduced here with only
// exported API since internal/serveapp cannot import that file's unexported types.
type fakeResidencyBackend struct {
	decoder.Backend
	rf *fakeResident
}

func (b *fakeResidencyBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &fakeResident{vocab: vocab}
	return b.rf, true, nil
}

func (b *fakeResidencyBackend) Close() error { return nil }

// residentServed loads the committed tiny fixture (tinyServed's own fixture,
// prefillpath_test.go) through a fake backend that reports residency — giving
// lm.model.ResidentActive() == true and, via the returned *fakeResident, a settable
// ContextCap() (rf.capPos), without a GPU.
func residentServed(t *testing.T) (*loadedModel, *fakeResident) {
	t.Helper()
	p := filepath.Join("..", "..", "testdata", "glm-tiny.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no committed tiny fixture at %s", p)
	}
	be := &fakeResidencyBackend{}
	name := "fake-resident-m01"
	decoder.RegisterBackend(name, func() (decoder.Backend, error) {
		cpu, err := decoder.NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := decoder.Load(p, decoder.Options{Backend: name, Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load tiny fixture with fake resident backend: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Fatal("fake resident backend did not activate — ResidentActive() = false")
	}
	return &loadedModel{name: "tiny-resident", model: m}, be.rf
}

// TestLoadedModel_residentPathTrueForAdapterWhenResidentActive is M-01's own gate
// (docs/audit-2026-09-10.md): residentPath (and prepare's context-window enforcement with it) must
// key off ResidentActive(), not "no adapter" — an adapter's own first turn still runs resident GPU
// decode when one is active (decoder/model.go's generateInto useGPU condition doesn't exclude
// adapters), so `lm.adapter == ""` wrongly reported false for exactly the case that needed the
// tighter resident cap enforced.
func TestLoadedModel_residentPathTrueForAdapterWhenResidentActive(t *testing.T) {
	lm, _ := residentServed(t)
	lm.adapter = "ft" // an adapter model — the case the old `lm.adapter == ""` predicate missed

	if !lm.residentPath() {
		t.Error("residentPath() = false for an adapter model with an active resident — want true: " +
			"ResidentActive() is what determines whether the resident cap applies, not the adapter name")
	}

	lm.adapter = ""
	if !lm.residentPath() {
		t.Error("residentPath() = false with no adapter and an active resident — want true")
	}
}

// TestLoadedModel_residentPathNilModel guards the nil-safety this method adds over a bare
// lm.model.ResidentActive() call — an embedding-only loadedModel (lm.model == nil, per
// s.pathFields' own guard) must not panic.
func TestLoadedModel_residentPathNilModel(t *testing.T) {
	lm := &loadedModel{name: "embed-only"}
	if lm.residentPath() {
		t.Error("residentPath() = true with lm.model == nil, want false")
	}
}

// TestServe_prepareEnforcesResidentCapForAdapter reproduces M-01's exact failure scenario end to
// end: an adapter model whose resident backend has a smaller context cap than MaxPositions must
// have prepare enforce the SMALLER (resident) cap, matching what generateInto's useGPU condition
// actually admits for an adapter's first turn — not the old behavior, which enforced the uncapped
// MaxPositions and let a resident prefill past the cap die mid-request instead of being rejected
// cleanly up front.
func TestServe_prepareEnforcesResidentCapForAdapter(t *testing.T) {
	lm, rf := residentServed(t)
	lm.adapter = "ft"

	maxPos := lm.model.Config().MaxPositions
	capPos := 4
	if maxPos <= capPos {
		t.Skipf("tiny fixture's MaxPositions (%d) is not larger than the test cap (%d)", maxPos, capPos)
	}
	rf.capPos = capPos

	one := 1
	sm := sampling{MaxTokens: &one}

	// A prompt inside the resident cap must be accepted.
	if _, err := lm.prepare(sm, make([]int, capPos-1), lm.residentPath()); err != nil && strings.Contains(err.Error(), "context_length_exceeded") {
		t.Errorf("a prompt under the resident cap (%d) was rejected: %v", capPos-1, err)
	}
	// A prompt past the resident cap, but still under MaxPositions, must now be rejected cleanly —
	// the old `lm.adapter == ""` predicate let this through to die mid-prefill instead (M-01).
	if _, err := lm.prepare(sm, make([]int, capPos+1), lm.residentPath()); err == nil || !strings.Contains(err.Error(), "context_length_exceeded") {
		t.Errorf("a prompt past the resident cap (%d) but under MaxPositions (%d) was not rejected as context_length_exceeded: %v", capPos+1, maxPos, err)
	}
}

package decoder

import (
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// The load path used to check only DecodeRunnerEligible before asking the backend to build, and replaced any
// backend decline with one generic string. It now runs every gate ResidentEligible applies (residentAdmission)
// for a backend that declares its capabilities, and records WHICH gate refused. These drive that through Load
// with a fake residency backend that reports a real backend's name.

type namedResidencyBackend struct {
	Backend
	name  string
	built atomic.Int32
}

func (b *namedResidencyBackend) Name() string { return b.name }
func (b *namedResidencyBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	b.built.Add(1)
	_, _, _, _, _, _, vocab := m.Dims()
	return &fakeResident{vocab: vocab}, true, nil
}
func (b *namedResidencyBackend) Close() error { return nil }

var admissionBackendSeq atomic.Int32

func loadWithNamedBackend(t *testing.T, dir, name string) (*Model, *namedResidencyBackend) {
	t.Helper()
	be := &namedResidencyBackend{name: name}
	reg := "fake-admission-" + name + "-" + string(rune('a'+admissionBackendSeq.Add(1)%26))
	RegisterBackend(reg, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(dir, Options{Backend: reg, Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, be
}

func TestResidentAdmission_declinesBeforeBuildAndNamesTheGate(t *testing.T) {
	var needsMore, supported string // a fixture Metal's declared features do not cover, and one they do
	dirs, _ := filepath.Glob(filepath.Join("..", "testdata", "*", "model.safetensors"))
	for _, st := range dirs {
		dir := filepath.Dir(st)
		m, err := Load(dir, Options{Quant: "int4"})
		if err != nil {
			continue
		}
		eligible := m.w.arch.decodeRunnerEligible()
		missing := m.MissingResidentFeatures(ResidentBackendFeatures("metal"))
		m.Close()
		switch {
		case eligible && len(missing) > 0 && needsMore == "":
			needsMore = dir
		case eligible && len(missing) == 0 && residentGateReason(m.w.arch, "metal") == "" && supported == "":
			supported = dir
		}
	}
	if needsMore == "" || supported == "" {
		t.Skipf("fixtures do not cover both outcomes (needs-more=%q supported=%q)", needsMore, supported)
	}

	m, be := loadWithNamedBackend(t, needsMore, "metal")
	if n := be.built.Load(); n != 0 {
		t.Errorf("%s: BuildResident was called %d times for a model the admission gate refuses", filepath.Base(needsMore), n)
	}
	if m.ResidentActive() {
		t.Errorf("%s: resident despite a missing feature", filepath.Base(needsMore))
	}
	why := m.ResidentDecline()
	want := m.MissingResidentFeatures(ResidentBackendFeatures("metal"))
	if !strings.Contains(why, "metal does not implement") || !strings.Contains(why, string(want[0])) {
		t.Errorf("%s: decline reason %q does not name the missing feature(s) %v", filepath.Base(needsMore), why, want)
	}

	m2, be2 := loadWithNamedBackend(t, supported, "metal")
	if be2.built.Load() != 1 || !m2.ResidentActive() {
		t.Errorf("%s: a model every gate admits was not built resident (built=%d, decline %q)",
			filepath.Base(supported), be2.built.Load(), m2.ResidentDecline())
	}

	// A backend with no declared capability set keeps the old contract: its own BuildResident decides.
	_, be3 := loadWithNamedBackend(t, needsMore, "some-out-of-tree-backend")
	if be3.built.Load() != 1 {
		t.Errorf("an undeclared backend was not asked to build (built=%d) — only declared backends take the full gate", be3.built.Load())
	}
}

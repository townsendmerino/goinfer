package decoder

import (
	"strings"
	"testing"
)

// TestSeams_ownLoopFamiliesStillRefuse: OP3 keeps each pre-OP3 own-loop family's seam refusals exactly (seam_refusals.go), over
// every registered model type, so a family is covered the day it is registered.
func TestSeams_ownLoopFamiliesStillRefuse(t *testing.T) {
	refused := 0
	for _, mt := range strings.Split(knownModelTypes(), ", ") {
		cfg := representativeConfig(mt)
		if cfg == nil {
			continue
		}
		arch, _, err := resolveArchitecture(cfg)
		if err != nil {
			t.Fatalf("%s: %v", mt, err)
		}
		if !arch.ownLoopFamily() {
			continue
		}
		refused++
		m := &Model{w: &Weights{arch: arch}}
		if _, _, _, _, err := m.ForwardSubCapture(0, NewKVCache(1, 1, 1, 0, 4, nil)); err == nil || !strings.Contains(err.Error(), "not wired") {
			t.Errorf("%s: ForwardSubCapture did not refuse (%v)", mt, err)
		}
		if _, err := m.HiddenLast([]int{0}); err == nil || !strings.Contains(err.Error(), "not wired") {
			t.Errorf("%s: HiddenLast did not refuse (%v)", mt, err)
		}
		_, _, err = m.ForwardCapture(0, NewKVCache(1, 1, 1, 0, 4, nil), nil)
		wired := err == nil || !strings.Contains(err.Error(), "seam not wired")
		if wired != arch.ownLoopCaptured() {
			t.Errorf("%s: ForwardCapture wired=%v, its loop captured=%v (%v)", mt, wired, arch.ownLoopCaptured(), err)
		}
	}
	if refused == 0 {
		t.Fatal("no own-loop family among the registered model types: the test would test nothing")
	}
}

package decoder

import (
	"os"
	"slices"
	"testing"
)

// TestGemma4EModel_realDeclinesResident confirms on the REAL E2B GGUF (not a synthetic arch) that a Gemma-4 E-model resolves
// to the E-model shape and is DECLINED by a resident backend that does not implement it, so it falls back to the CPU/staged
// path there instead of being admitted to a runner that skips its PLE branch and silently mis-runs. Metal, CUDA and WebGPU
// implement the shape (docs/tasks/task-multimodal-support-2026-10.md, S1 and S12.1) and must NOT report it missing; the
// backend that lacks it here is a feature set without it.
// Heavy (loads a multi-GB checkpoint) — opt in with GOINFER_HEAVY_TESTS=1.
func TestGemma4EModel_realDeclinesResident(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/gemma-4-E2B_q4_0-it.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no E2B GGUF at %s", path)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1") // even with the bring-up gate ON, the E-model must decline
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load E2B: %v", err)
	}
	defer m.Close()

	// It must resolve to the E-model shape (PLE present).
	if g := m.w.arch.gemma4; g == nil || g.HiddenSizePerLayerInput == 0 {
		t.Fatalf("E2B did not resolve to a Gemma-4 E-model (gemma4=%v ple=%d) — the fixture/loader changed",
			m.w.arch.gemma4 != nil, func() int {
				if m.w.arch.gemma4 != nil {
					return m.w.arch.gemma4.HiddenSizePerLayerInput
				}
				return -1
			}())
	}
	t.Logf("E2B resolved: hidden_size_per_layer_input=%d (E-model)", m.w.arch.gemma4.HiddenSizePerLayerInput)

	// A resident backend without the E-model bridge must report FeatGemma4EModel missing ⇒ decline. Every real backend has
	// it now, so the backend without it is each real one's feature set with that one feature removed.
	for _, be := range []string{"metal", "cuda", "webgpu"} {
		without := map[ResidentFeature]bool{}
		for f, ok := range residentBackendFeatures[be] {
			without[f] = ok && f != FeatGemma4EModel
		}
		missing := m.MissingResidentFeatures(without)
		if !slices.Contains(missing, FeatGemma4EModel) {
			t.Errorf("%s without the E-model feature: real E2B is NOT declined for the E-model shape (missing=%v) — it would be admitted and mis-run", be, missing)
		}
	}
	for _, be := range []string{"metal", "cuda", "webgpu"} {
		if missing := m.MissingResidentFeatures(residentBackendFeatures[be]); slices.Contains(missing, FeatGemma4EModel) {
			t.Errorf("%s reports the E-model shape missing (missing=%v) although it declares FeatGemma4EModel", be, missing)
		}
	}
}

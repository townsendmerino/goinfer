package decoder

import (
	"slices"
	"strings"
	"testing"
)

// gatedActBackends are the resident backends whose gated kernels select the activation by GatedActResident's ordinal.
var gatedActBackends = []string{"cuda", "metal", "webgpu"}

// plainGatedArch is a dense gated block that needs no feature beyond its activation's, so a decline is the activation's.
func plainGatedArch(act ActKind) *Architecture {
	return &Architecture{NumLayers: 1, HeadDim: 128, NormPlacement: NormPre2, Act: act}
}

// TestGatedAct_onlyTanhAndSiLUAdmitted pins what GatedActResident may hand a resident kernel for a gated MLP: 0 (GELU-tanh) and 1
// (SiLU), the two ordinals the kernels know. Every other ordinal, including one past the last ActKind, must fail the feature check
// on every backend, and through Model.MissingResidentFeatures too, which is the check each backend's BuildResident runs itself.
func TestGatedAct_onlyTanhAndSiLUAdmitted(t *testing.T) {
	for act := ActKind(0); act <= ActGelu+2; act++ {
		a := plainGatedArch(act)
		if !a.decodeRunnerEligible() {
			t.Fatalf("act %d: the plain gated arch fails the runner shape, so this test would pass for the wrong reason", act)
		}
		m := &Model{w: &Weights{arch: a}}
		for _, be := range gatedActBackends {
			admitted := ResidentEligible(a, be)
			missing := m.MissingResidentFeatures(ResidentBackendFeatures(be))
			if admitted != (len(missing) == 0) {
				t.Errorf("act %d on %s: ResidentEligible=%v but MissingResidentFeatures=%v", act, be, admitted, missing)
			}
			if kernelKnows := act == ActGeluTanh || act == ActSiLU; admitted != kernelKnows {
				t.Errorf("act %d (%v) on %s: admitted=%v, want %v (GatedActResident=%d reaches a kernel that knows 0 and 1)",
					act, act, be, admitted, kernelKnows, m.GatedActResident())
			}
		}
	}
	for be, set := range residentBackendFeatures {
		if set[FeatGatedActUnknown] {
			t.Errorf("%s declares %q, which names no kernel", be, FeatGatedActUnknown)
		}
	}
}

// TestGatedGELUErf_declinesByName: a gated exact-erf GELU arch is refused by every backend, and the reason names the feature.
func TestGatedGELUErf_declinesByName(t *testing.T) {
	a := plainGatedArch(ActGelu)
	if got := a.residentFeatures(); !slices.Equal(got, []ResidentFeature{FeatGatedGELUErf}) {
		t.Fatalf("gated ActGelu derives %v, want only %q", got, FeatGatedGELUErf)
	}
	for _, be := range gatedActBackends {
		if ResidentEligible(a, be) {
			t.Errorf("%s admits a gated exact-erf GELU arch; its gated kernel would run tanh-GELU or SiLU", be)
		}
		if why := residentGateReason(a, be); !strings.Contains(why, string(FeatGatedGELUErf)) {
			t.Errorf("%s: residentGateReason = %q, want it to name %q", be, why, FeatGatedGELUErf)
		}
	}
}

// TestGatedGELUErf_sparkNamesIt: Spark-X2.5 fails the runner shape and lacks the attention output gate, and neither may hide that
// its activation is also missing. Read on the registry's representative config and on the tiny checkpoint's loaded Model.
func TestGatedGELUErf_sparkNamesIt(t *testing.T) {
	a, _, err := resolveArchitecture(representativeConfig("spark2_5"))
	if err != nil {
		t.Fatal(err)
	}
	if a.NonGatedMLP || a.Act != ActGelu {
		t.Fatalf("spark2_5 is no longer a gated exact-GELU arch (NonGatedMLP=%v Act=%v); this test needs another family", a.NonGatedMLP, a.Act)
	}
	m, err := Load("testdata/spark2-5-tiny", Options{})
	if err != nil {
		t.Fatalf("load spark2-5-tiny: %v", err)
	}
	for _, be := range gatedActBackends {
		if ResidentEligible(a, be) {
			t.Errorf("%s admits spark2_5", be)
		}
		why := residentGateReason(a, be)
		for _, f := range []ResidentFeature{FeatGatedGELUErf, FeatAttnOutputGate} {
			if !strings.Contains(why, string(f)) {
				t.Errorf("%s: residentGateReason = %q, want it to name %q", be, why, f)
			}
		}
		missing := m.MissingResidentFeatures(ResidentBackendFeatures(be))
		if !slices.Contains(missing, FeatGatedGELUErf) {
			t.Errorf("%s: the loaded spark2-5-tiny is missing %v, want it to include %q", be, missing, FeatGatedGELUErf)
		}
	}
}

// TestGatedGELU_gemmaUnchanged: GELU-tanh keeps FeatGatedGELU and nothing else for its activation, and the Gemma families every
// backend ran before the exact-GELU split still pass.
func TestGatedGELU_gemmaUnchanged(t *testing.T) {
	a := plainGatedArch(ActGeluTanh)
	if got := a.residentFeatures(); !slices.Equal(got, []ResidentFeature{FeatGatedGELU}) {
		t.Fatalf("gated ActGeluTanh derives %v, want only %q", got, FeatGatedGELU)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	for _, name := range []string{"gemma", "gemma3", "gemma3_text", "gemma4", "gemma4_text", "gemma4_unified_text"} {
		g, _, err := resolveArchitecture(representativeConfig(name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		feats := g.residentFeatures()
		if !slices.Contains(feats, FeatGatedGELU) || slices.Contains(feats, FeatGatedGELUErf) || slices.Contains(feats, FeatGatedActUnknown) {
			t.Errorf("%s derives %v, want %q and neither %q nor %q", name, feats, FeatGatedGELU, FeatGatedGELUErf, FeatGatedActUnknown)
		}
		for _, be := range gatedActBackends {
			if !ResidentEligible(g, be) {
				t.Errorf("%s is no longer resident on %s: %s", name, be, residentGateReason(g, be))
			}
		}
	}
}

// TestNonGatedGELUErf_declinesByName: a GPT-2 checkpoint declaring HF's exact "gelu" passes every other feature Metal declares, and
// Metal's non-gated kernel has GELU-tanh and SiLU only, so the activation must be what refuses it. The default "gelu_new" stays.
func TestNonGatedGELUErf_declinesByName(t *testing.T) {
	cfg := representativeConfig("gpt2")
	tanh, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !ResidentEligible(tanh, "metal") {
		t.Fatalf("precondition: gpt2 (gelu_new) should be resident on metal: %s", residentGateReason(tanh, "metal"))
	}
	cfg.ActivationFunction = "gelu"
	erf, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !erf.NonGatedMLP || erf.Act != ActGelu {
		t.Fatalf("gpt2 with activation_function \"gelu\" resolved NonGatedMLP=%v Act=%v", erf.NonGatedMLP, erf.Act)
	}
	for _, be := range gatedActBackends {
		if ResidentEligible(erf, be) {
			t.Errorf("%s admits a non-gated exact-erf GELU arch", be)
		}
		if why := residentGateReason(erf, be); !strings.Contains(why, string(FeatNonGatedGELUErf)) {
			t.Errorf("%s: residentGateReason = %q, want it to name %q", be, why, FeatNonGatedGELUErf)
		}
	}
}

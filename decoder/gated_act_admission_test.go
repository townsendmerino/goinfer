package decoder

import (
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/program"
)

// gatedActBackends are the resident backends whose gated kernels select the activation by GatedActResident's ordinal.
var gatedActBackends = []string{"cuda", "metal", "webgpu"}

// plainGatedArch is a dense gated block that needs no feature beyond its activation's, so a decline is the activation's.
func plainGatedArch(act ActKind) *Architecture {
	return &Architecture{NumLayers: 1, HeadDim: 128, NormPlacement: NormPre2, Act: act}
}

// TestGatedAct_onlyTanhAndSiLUAdmitted pins what GatedActResident may hand a resident kernel for a gated MLP: 0 (GELU-tanh) and 1
// (SiLU), the two ordinals the kernels know. Every other ordinal, including one past the last ActKind, must fail the feature check
// (or a table backend's operation table) on every backend, and the check each backend's BuildResident runs itself.
func TestGatedAct_onlyTanhAndSiLUAdmitted(t *testing.T) {
	for act := ActKind(0); act <= ActGelu+2; act++ {
		a := plainGatedArch(act)
		if !a.decodeRunnerEligible() {
			t.Fatalf("act %d: the plain gated arch fails the runner shape, so this test would pass for the wrong reason", act)
		}
		m := &Model{w: &Weights{arch: a}}
		for _, be := range gatedActBackends {
			admitted := ResidentEligible(a, be)
			if residentBackendTables[be] != nil {
				// A table backend's build checks the table, not a feature set (metal/backend.go's residentPreBuild).
				if covered := residentCovers(a, be); admitted != covered {
					t.Errorf("act %d on %s: ResidentEligible=%v but the table covers it %v", act, be, admitted, covered)
				}
			} else if missing := m.MissingResidentFeatures(ResidentBackendFeatures(be)); admitted != (len(missing) == 0) {
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

// TestGatedAct_metalModelPathAndGemma4MoE (Gemini on #548): what Metal's build checks on a Model (metal/backend.go's
// residentPreBuild: the table over Model.Program, and Model.FeaturesOutsideProgram) agrees with ResidentEligible for every gated
// activation; and Gemma 4's MoE block, whose operation carries no activation, declines an exact-erf or unknown GELU through the
// features outside the program while GELU-tanh stays admitted.
func TestGatedAct_metalModelPathAndGemma4MoE(t *testing.T) {
	for act := ActKind(0); act <= ActGelu+2; act++ {
		a := plainGatedArch(act)
		m := &Model{w: &Weights{arch: a}}
		build := program.Metal.Check(m.Program()) == nil && len(m.FeaturesOutsideProgram()) == 0
		if eligible := ResidentEligible(a, "metal"); build != eligible {
			t.Errorf("act %d: Metal's build check on the Model admits %v, ResidentEligible %v", act, build, eligible)
		}
	}
	base, _, err := resolveArchitecture(representativeConfig("gemma4_text"))
	if err != nil {
		t.Fatal(err)
	}
	if base.gemma4 == nil || base.MoE == nil {
		t.Fatalf("representativeConfig(gemma4_text) is not Gemma 4's MoE variant (gemma4=%v MoE=%v)", base.gemma4 != nil, base.MoE != nil)
	}
	for _, c := range []struct {
		act  ActKind
		want ResidentFeature // "" when admitted
	}{{ActGeluTanh, ""}, {ActGelu, FeatGatedGELUErf}, {ActGelu + 1, FeatGatedActUnknown}} {
		a := *base
		a.Act = c.act
		m := &Model{w: &Weights{arch: &a}}
		out := m.FeaturesOutsideProgram()
		var reasons []string
		for _, g := range residentGateReasons(&a, "metal", false) {
			reasons = append(reasons, g.Reason)
		}
		switch {
		case c.want == "" && (len(out) > 0 || !ResidentEligible(&a, "metal")):
			t.Errorf("Gemma 4 MoE with %v: outside the program %v, gates %q; want admitted", c.act, out, reasons)
		case c.want != "" && !slices.Contains(out, c.want):
			t.Errorf("Gemma 4 MoE with %v: outside the program %v, want %q", c.act, out, c.want)
		case c.want != "" && (ResidentEligible(&a, "metal") || !strings.Contains(strings.Join(reasons, "; "), string(c.want))):
			t.Errorf("Gemma 4 MoE with %v: gates %q, want a decline naming %q", c.act, reasons, c.want)
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
		want := string(FeatGatedGELUErf)
		if residentBackendTables[be] != nil {
			want = tableGatedGELUGap // a table names the operation's parameter
		}
		if why := residentGateReason(a, be); !strings.Contains(why, want) {
			t.Errorf("%s: residentGateReason = %q, want it to name %q", be, why, want)
		}
	}
}

// tableGatedGELUGap is how an operation table names a gated FFN whose exact-erf GELU it lacks.
const tableGatedGELUGap = "no GELU activation in its gated block"

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
		if tb := residentBackendTables[be]; tb != nil {
			// A table names each gap by its operation: the output gate and the gated FFN's activation, neither hiding the other,
			// on the representative config and on the loaded checkpoint.
			for _, w := range []string{"(out-gate)", tableGatedGELUGap} {
				if !strings.Contains(why, w) {
					t.Errorf("%s: residentGateReason = %q, want it to name %q", be, why, w)
				}
				if gaps := strings.Join(tb.Gaps(m.Program()), "; "); !strings.Contains(gaps, w) {
					t.Errorf("%s: the loaded spark2-5-tiny's gaps %q, want them to name %q", be, gaps, w)
				}
			}
			continue
		}
		for _, f := range []ResidentFeature{FeatGatedGELUErf, FeatAttnOutputGateSigmoid} {
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
		want := string(FeatNonGatedGELUErf)
		if residentBackendTables[be] != nil {
			want = "no GELU activation in its non-gated block" // a table names the operation's parameter
		}
		if why := residentGateReason(erf, be); !strings.Contains(why, want) {
			t.Errorf("%s: residentGateReason = %q, want it to name %q", be, why, want)
		}
	}
}

package decoder

import (
	"strings"
	"testing"
)

// admitArch resolves a representative arch for model type mt, or the config of the tiny fixture dir when mt is a path.
func admitArch(t *testing.T, mt string) *Architecture {
	t.Helper()
	var cfg *Config
	if strings.Contains(mt, "/") {
		c, err := censusConfig(mt, true)
		if err != nil {
			t.Fatalf("%s: %v", mt, err)
		}
		cfg = c
	} else if cfg = representativeConfig(mt); cfg == nil {
		t.Fatalf("model_type %q has no representativeConfig", mt)
	}
	a, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("%s: %v", mt, err)
	}
	return a
}

// TestAdmit_table pins Admit's gate order on facts given directly, with no Model: each row's facts would also fail a later gate,
// so the expected refusal is the earliest one. want is a substring of Why, or "" for an admission.
func TestAdmit_table(t *testing.T) {
	// Admit reads no environment: with the knob set in the process, a fact that says it is unset must still close Granite's gate.
	t.Setenv("GOINFER_SSM_RESIDENT", "1")
	const nemoMoE = "../testdata/nemotron3nano-tiny"
	rows := []struct {
		name string
		mt   string
		f    AdmitFacts
		want string
	}{
		{"no residency first", "llama4_text", AdmitFacts{Backend: "cpu", Quant: "q4k"}, "backend does not implement residency"},
		{"q4k on webgpu before the shape", "llama4_text", AdmitFacts{Backend: "webgpu", Residency: true, Quant: "q4k"}, "--quant q4k (native Q4_K) has no resident kernel on webgpu"},
		{"q4k on cuda passes", "llama", AdmitFacts{Backend: "cuda", Residency: true, Quant: "q4k", ProjsInt4: true}, ""},
		{"auto metal keeps f32 on the CPU", "llama4_text", AdmitFacts{Backend: "metal", Residency: true, BackendAuto: true, QuantLabel: "f32"}, "-backend auto keeps this f32 model on the CPU"},
		{"auto metal int4 passes", "llama", AdmitFacts{Backend: "metal", Residency: true, BackendAuto: true, QuantLabel: "int4"}, ""},
		{"nemotron MoE on cuda named", nemoMoE, AdmitFacts{Backend: "cuda", Residency: true, ProjsInt4: true}, "Nemotron-H MoE FFN block has no GPU resident implementation"},
		{"nemotron MoE on webgpu overrides the shape", nemoMoE, AdmitFacts{Backend: "webgpu", Residency: true, ProjsInt4: true}, ""},
		{"nemotron MoE on webgpu-subgroup overrides the shape", nemoMoE, AdmitFacts{Backend: "webgpu-subgroup", Residency: true, ProjsInt4: true}, ""},
		{"own forward declines on the shape", "lfm2", AdmitFacts{Backend: "cuda", Residency: true, ActGroup: 32}, "arch is not eligible for the resident decode runner: own forward"},
		{"granite closed without the knob fact", "granitemoehybrid", AdmitFacts{Backend: "cuda", Residency: true}, "bring-up gate (GOINFER_SSM_RESIDENT unset)"},
		{"granite opened by the knob fact", "granitemoehybrid", AdmitFacts{Backend: "cuda", Residency: true, SSMResident: true}, ""},
		{"nemotron int8 needs int4 or the knob", "nemotron_h", AdmitFacts{Backend: "cuda", Residency: true}, "Nemotron-H runs resident with int4 projections only by default"},
		{"nemotron int8 with the knob", "nemotron_h", AdmitFacts{Backend: "cuda", Residency: true, SSMResident: true}, ""},
		{"activation groups before features", "bailing_hybrid", AdmitFacts{Backend: "webgpu", Residency: true, ActGroup: 32}, "ActQuantGroup 32: webgpu's resident projections"},
		{"activation groups pass on cuda", "llama", AdmitFacts{Backend: "cuda", Residency: true, ActGroup: 32}, ""},
		{"features last", "bailing_hybrid", AdmitFacts{Backend: "cuda", Residency: true}, "cuda does not implement [kda]"},
		{"an undeclared backend stops at the shape", "bailing_hybrid", AdmitFacts{Backend: "out-of-tree", Residency: true, ActGroup: 32}, ""},
	}
	for _, r := range rows {
		f := r.f
		f.Arch = admitArch(t, r.mt)
		got := Admit(f).Why
		if r.want == "" && got != "" || r.want != "" && !strings.Contains(got, r.want) {
			t.Errorf("%s: Admit(%s on %s).Why = %q, want %q", r.name, r.mt, f.Backend, got, r.want)
		}
	}
}

// TestAdmit_shapeAndGatesAreTheArchs pins Admission's split: Shape and Gates do not read the precision policy or a backend's
// override of the runner shape, which is why a model-free caller reads them and not Why.
func TestAdmit_shapeAndGatesAreTheArchs(t *testing.T) {
	ad := Admit(AdmitFacts{Arch: admitArch(t, "nemotron_h"), Backend: "cuda", Residency: true})
	if ad.Why == "" || ad.Shape != "" || len(ad.Gates) != 0 {
		t.Errorf("dense nemotron_h at int8: Why %q, Shape %q, Gates %v; want a precision refusal in Why only", ad.Why, ad.Shape, ad.Gates)
	}
	ad = Admit(AdmitFacts{Arch: admitArch(t, "../testdata/nemotron3nano-tiny"), Backend: "webgpu", Residency: true, ProjsInt4: true})
	if ad.Why != "" || ad.Shape == "" {
		t.Errorf("nemotron MoE on webgpu: Why %q, Shape %q; want admitted with the arch's shape decline kept in Shape", ad.Why, ad.Shape)
	}
}

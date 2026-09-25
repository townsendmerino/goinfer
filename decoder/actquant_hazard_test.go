package decoder

import (
	"strings"
	"testing"
)

// TestActivationQuantHazard_declinesGPUResidency: a Phi-3 model must not be admitted as resident on any
// GPU backend, because every resident projection quantizes activations to int8 per row and Phi-3's
// activation outliers do not survive that (queue-engineering.md H2). Checked through ResidentEligible,
// the one gate both the runtime (residentAdmission) and the generated hardware matrix use, on a real
// load of the committed tiny Phi-3 fixture.
func TestActivationQuantHazard_declinesGPUResidency(t *testing.T) {
	m, err := Load("../testdata/phi3-tiny", Options{})
	if err != nil {
		t.Fatalf("Load phi3-tiny: %v", err)
	}
	defer m.Close()
	if m.w.arch.Name != "phi3" {
		t.Fatalf("arch = %q, want phi3", m.w.arch.Name)
	}
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		if ResidentEligible(m.w.arch, be) {
			t.Errorf("phi3 admitted as resident on %s; its activation outliers break int8 activations (H2)", be)
		}
		if why := residentGateReason(m.w.arch, be); !strings.Contains(why, "Phi-3's activation outliers") {
			t.Errorf("%s decline reason %q does not name the activation-quantization hazard", be, why)
		}
	}
	if QuantizesActivations("int8") || QuantizesActivations("f32") || !QuantizesActivations("int4") {
		t.Error("QuantizesActivations: int8/f32 keep f32 activations; int4 does not")
	}
	if ActivationQuantHazard("qwen2") != "" {
		t.Error("a family with no measured hazard must not be flagged")
	}
}

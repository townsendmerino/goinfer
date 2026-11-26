package decoder

import (
	"strings"
	"testing"
)

// TestActivationQuantHazard_declinesGPUResidency: a Phi-3 model must not be admitted as resident on any
// GPU backend, because every resident projection quantizes activations to int8 per row and Phi-3's
// activation outliers do not survive that (queue-engineering.md H2). Checked through ResidentEligible,
// the one gate both the runtime (Admit) and the generated hardware matrix use, on a real
// load of the committed tiny Phi-3 fixture.
func TestActivationQuantHazard_declinesGPUResidency(t *testing.T) {
	m, err := Load("../testdata/phi3-tiny", Options{})
	if err != nil {
		t.Fatalf("Load phi3-tiny: %v", err)
	}
	defer m.Close()
	if m.w.Arch.Name != "phi3" {
		t.Fatalf("arch = %q, want phi3", m.w.Arch.Name)
	}
	for _, be := range []string{"cuda", "metal", "webgpu"} {
		if ResidentEligible(m.w.Arch, be) {
			t.Errorf("phi3 admitted as resident on %s; its activation outliers break int8 activations (H2)", be)
		}
		if why := residentGateReason(m.w.Arch, be); !strings.Contains(why, "Phi-3's activation outliers") {
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

// TestActivationQuantHazard_metalQ4KLane: the one Metal configuration that clears the hazard is the q4k lane, which
// quantizes no activation at all. residentGateReasonAct admits Phi-3 on Metal
// with actSafe, and on WebGPU, whose actSafe is per-32 activation scales over int8 weights; the default (actSafe false)
// still declines both.
func TestActivationQuantHazard_metalQ4KLane(t *testing.T) {
	m, err := Load("../testdata/phi3-tiny", Options{})
	if err != nil {
		t.Fatalf("Load phi3-tiny: %v", err)
	}
	defer m.Close()
	if why := residentGateReasonAct(m.w.Arch, "metal", true); strings.Contains(why, "Phi-3's activation outliers") {
		t.Errorf("the q4k lane (actSafe) on metal still declined for the hazard: %q", why)
	}
	if why := residentGateReasonAct(m.w.Arch, "metal", false); !strings.Contains(why, "Phi-3's activation outliers") {
		t.Errorf("metal without the q4k lane must keep the hazard decline, got %q", why)
	}
	if why := residentGateReasonAct(m.w.Arch, "webgpu", true); strings.Contains(why, "Phi-3's activation outliers") {
		t.Errorf("webgpu with per-32 activation scales (actSafe) still declined for the hazard: %q", why)
	}
	if why := residentGateReasonAct(m.w.Arch, "webgpu", false); !strings.Contains(why, "Phi-3's activation outliers") {
		t.Errorf("webgpu with per-vector scales must keep the hazard decline, got %q", why)
	}
}

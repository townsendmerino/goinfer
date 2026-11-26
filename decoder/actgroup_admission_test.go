package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"strings"
	"sync/atomic"
	"testing"
)

// TestActGroupResidentDecline_table is the gate's rule: a set activation group declines every resident
// backend but CUDA and WebGPU at int8int8 (its per-32 GEMV takes int8 weights), except Metal's q4k lane, which
// quantizes no activation.
func TestActGroupResidentDecline_table(t *testing.T) {
	for _, c := range []struct {
		backend, quant string
		group          int
		declines       bool
	}{
		{"metal", "int8int8", 32, true},
		{"webgpu", "int8int8", 32, false},
		{"metal", "int4", 32, true},
		{"cuda", "int8int8", 32, false},
		{"metal", "q4k", 32, false}, // modelFromOptions stamps 32 on every q4k load
		{"webgpu", "q4k", 32, true},
		{"webgpu", "int4", 32, true},
		{"webgpu", "f32", 32, true},
		{"metal", "int8int8", 0, false},
		{"webgpu", "int4", 0, false},
	} {
		got := actGroupResidentDecline(c.backend, c.group, c.quant)
		if (got != "") != c.declines {
			t.Errorf("%s %s group %d: decline %q, want declines=%v", c.backend, c.quant, c.group, got, c.declines)
		}
		if got != "" && !strings.Contains(got, "ActQuantGroup") {
			t.Errorf("%s %s: the decline does not name the option: %q", c.backend, c.quant, got)
		}
	}
}

var actGroupBackendSeq atomic.Int32

// TestResidentAdmission_actGroupDeclinesOffCUDA drives the gate through Load with a fake residency backend
// reporting each real backend's name (resident_admission_test.go's harness): with ActQuantGroup set, Metal is
// refused before the build and the decline names the option; CUDA and WebGPU build; Metal without the option
// builds, so the refusal is the option's and not the fixture's.
func TestResidentAdmission_actGroupDeclinesOffCUDA(t *testing.T) {
	const dir = "../testdata/llama-tiny"
	load := func(name string, group int) (*Model, *namedResidencyBackend) {
		t.Helper()
		be := &namedResidencyBackend{name: name}
		reg := "fake-actgroup-" + name + "-" + string(rune('a'+actGroupBackendSeq.Add(1)%26))
		RegisterBackend(reg, func() (Backend, error) {
			cpu, err := NewBackend("cpu")
			if err != nil {
				return nil, err
			}
			be.Backend = cpu
			return be, nil
		})
		m, err := Load(dir, Options{Backend: reg, Quant: "int8int8", ActQuantGroup: group})
		if err != nil {
			t.Fatalf("load %s: %v", name, err)
		}
		t.Cleanup(func() { _ = m.Close() })
		return m, be
	}
	for _, name := range []string{"metal"} {
		m, be := load(name, 32)
		if n := be.built.Load(); n != 0 {
			t.Errorf("%s: BuildResident called %d times with ActQuantGroup 32", name, n)
		}
		if m.ResidentActive() {
			t.Errorf("%s: resident with ActQuantGroup 32", name)
		}
		if !strings.Contains(m.ResidentDecline(), "ActQuantGroup") {
			t.Errorf("%s: decline does not name ActQuantGroup: %q", name, m.ResidentDecline())
		}
	}
	for _, name := range []string{"cuda", "webgpu"} {
		if _, be := load(name, 32); be.built.Load() == 0 {
			t.Errorf("%s: not built with ActQuantGroup 32; its residency implements per-group scales", name)
		}
	}
	if _, be := load("metal", 0); be.built.Load() == 0 {
		t.Error("metal without ActQuantGroup was not built: the fixture does not isolate the option")
	}
}

// TestWebGPUActGroupShapeDecline_layerNorm: WebGPU's per-32 activations have no LayerNorm quantize twin, so a LayerNorm
// family (Cohere) is declined at admission, naming the feature, while a dense RMSNorm family of the same shape class is not.
func TestWebGPUActGroupShapeDecline_layerNorm(t *testing.T) {
	for _, tc := range []struct {
		dir     string
		decline bool
	}{{"../testdata/cohere-tiny", true}, {"../testdata/llama-tiny", false}} {
		m, err := Load(tc.dir, Options{Quant: "int8int8"})
		if err != nil {
			t.Fatal(err)
		}
		why := webgpuActGroupShapeDecline("webgpu", 32, m.w.Arch)
		_ = m.Close()
		if got := why != ""; got != tc.decline || (tc.decline && !strings.Contains(why, string(darch.FeatLayerNorm))) {
			t.Errorf("%s: decline %q, want a decline naming %s: %v", tc.dir, why, darch.FeatLayerNorm, tc.decline)
		}
	}
}

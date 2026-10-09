package decoder

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

// TestAutoBackend_picksTheFirstGPUThatAnswers (R17, docs/tasks/task-first-hour.md): "auto" is the first GPU backend this
// binary links whose device answers, cuda then metal, else cpu, with a reason the CLIs print as-is. autoBackend is the
// selection over its inputs, so the cases that need a GPU backend linked run here without registering one: a fake
// registered under "cuda" would outlive this test in the process-wide registry and change what NewBackend("cuda")
// returns to every other test in the package.
func TestAutoBackend_picksTheFirstGPUThatAnswers(t *testing.T) {
	answers := func(ok ...string) func(string) (int64, bool) {
		return func(b string) (int64, bool) {
			if slices.Contains(ok, b) {
				return 1 << 30, true
			}
			return 0, false
		}
	}
	for _, tc := range []struct {
		name     string
		linked   []string
		probe    func(string) (int64, bool)
		goarch   string
		want     string
		skipped  bool
		reasonIn string
	}{
		{"no GPU backend linked", []string{"cpu"}, answers(), "amd64", "cpu", false, "this binary has no GPU backend"},
		{"cuda with a device", []string{"cpu", "cuda"}, answers("cuda"), "amd64", "cuda", false, "a CUDA device answered"},
		{"cuda without a device", []string{"cpu", "cuda"}, answers(), "amd64", "cpu", true, "cuda is built in, but no CUDA device answered"},
		{"metal on Apple silicon", []string{"cpu", "metal"}, answers("metal"), "arm64", "metal", false, "Apple silicon; metal runs int4 models"},
		// The darwin/amd64 release asset links metal, and its probe answers from the RAM size alone.
		{"metal on an Intel Mac", []string{"cpu", "metal"}, answers("metal"), "amd64", "cpu", true, "only been run on Apple silicon; -backend metal asks for it"},
		{"cuda before metal", []string{"cpu", "cuda", "metal"}, answers("cuda", "metal"), "arm64", "cuda", false, "a CUDA device answered"},
		{"metal after a cuda with no device", []string{"cpu", "cuda", "metal"}, answers("metal"), "arm64", "metal", false, "Apple silicon; metal runs int4 models"},
		{"webgpu alone is never picked", []string{"cpu", "webgpu"}, answers("webgpu"), "amd64", "cpu", true, "-backend webgpu asks for it"},
		{"webgpu does not outrank cuda", []string{"cpu", "cuda", "webgpu"}, answers("cuda", "webgpu"), "amd64", "cuda", false, "a CUDA device answered"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := autoBackend(tc.linked, tc.probe, tc.goarch)
			if c.Backend != tc.want || c.Skipped != tc.skipped || !strings.Contains(c.Reason, tc.reasonIn) {
				t.Errorf("autoBackend(%v, %s) = %+v; want backend %q, Skipped %v, a reason containing %q",
					tc.linked, tc.goarch, c, tc.want, tc.skipped, tc.reasonIn)
			}
			if !strings.HasPrefix(c.Reason, "auto: ") {
				t.Errorf("reason %q does not start with \"auto: \", which the CLIs' one line relies on", c.Reason)
			}
		})
	}
}

// TestLoad_autoResolvesBeforeTheLoad: Options{Backend: "auto"} reaches the load as the backend that will run, so the
// name checks inside a load (the int4 layout, the fit guard) and the banner's requested-vs-running split see that
// backend, not "auto". This test binary links no GPU backend, so auto is the CPU, and the banner must read as a plain
// cpu load, not "requested auto → running on cpu".
func TestLoad_autoResolvesBeforeTheLoad(t *testing.T) {
	if c := AutoBackend(); c.Backend != "cpu" {
		t.Skipf("this test binary links %v, and auto chose %q", CompiledBackends(), c.Backend)
	}
	if err := (Options{Backend: "auto"}).Validate(); err != nil {
		t.Fatalf("Options{Backend: \"auto\"}.Validate() = %v", err)
	}
	if be, err := NewBackend("auto"); err != nil || be == nil || be.Name() != "cpu" {
		t.Fatalf("NewBackend(\"auto\") = %v, %v; want the cpu backend", be, err)
	}
	m, err := Load(tinyFixture(t), Options{Backend: "auto"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if req, eff, rep := m.RequestedBackend(), m.EffectiveBackend(), m.BackendReport(); req != "cpu" || eff != "cpu" || rep != "cpu" {
		t.Errorf("Load(auto): requested %q, running %q, banner %q; want cpu for all three", req, eff, rep)
	}
}

// fakeNamedResidency is fakeResidencyBackend under another backend's name, so a by-name rule in withResidency can be
// driven without registering that name: the registry entry is unique per test, only Name() says "metal".
type fakeNamedResidency struct {
	fakeResidencyBackend
	name   string
	report string // non-empty: the built resident reports it as ResidentQuant
}

// quantReporting is a fake resident that reports the precision it runs at (ResidentQuantReporter).
type quantReporting struct {
	ResidentForward
	q string
}

func (r quantReporting) ResidentQuant() string { return r.q }

func (b *fakeNamedResidency) BuildResident(m *Model) (ResidentForward, bool, error) {
	rf, ok, err := b.fakeResidencyBackend.BuildResident(m)
	if b.report != "" {
		return quantReporting{rf, b.report}, ok, err
	}
	return rf, ok, err
}

func (b *fakeNamedResidency) Name() string { return b.name }

// TestAutoMetalPrecision_keepsTheLoadedPrecision (R17, then slice 5 of docs/tasks/task-metal-int8-2026-10.md): when
// auto chose metal, a model Metal would run only re-quantized (int8, int4mix) or not at all (f32) stays on the CPU at the
// precision it loaded at, and says so. int8int8 goes to BuildResident, since Metal runs dense int8int8 natively, and is
// kept only when the resident reports that it ran at int8int8; one Metal re-quantized (MoE, DeltaNet) is dropped for
// the CPU. int4, and any model on a metal the user named, go resident as before. Without the guard, the model-included
// 0.5B goinfer-chat (an int8int8 bundle) went resident on Metal at int4 and loaded in 1.7 s against 0.17 s on the CPU
// (exploratory runs, docs/measurements/r17-auto-backend-2026-10-01/).
func TestAutoMetalPrecision_keepsTheLoadedPrecision(t *testing.T) {
	for _, tc := range []struct {
		auto        bool
		quant       string
		report      string // what the fake resident reports as ResidentQuant ("" reports nothing: a re-quant)
		built, kept bool
		why         string
	}{
		{true, "int4", "", true, true, ""},
		{true, "int8int8", "int8int8", true, true, ""}, // native int8: kept
		{true, "int8int8", "", true, false, "keeps this int8int8 model on the CPU: metal would re-quantize it to int4"},
		{true, "int8", "", false, false, "keeps this int8 model on the CPU"},
		{true, "", "", false, false, "keeps this f32 model on the CPU: metal runs only int4 resident"},
		{false, "int8int8", "", true, true, ""}, // -backend metal named: runs re-quantized, as before
	} {
		t.Run(fmt.Sprintf("auto=%v quant=%q report=%q", tc.auto, tc.quant, tc.report), func(t *testing.T) {
			be := &fakeNamedResidency{name: "metal", report: tc.report}
			reg := fmt.Sprintf("fake-auto-metal-%v-%q-%q", tc.auto, tc.quant, tc.report)
			RegisterBackend(reg, func() (Backend, error) {
				cpu, err := NewBackend("cpu")
				be.Backend = cpu
				return be, err
			})
			m, err := Load(tinyFixture(t), Options{Backend: reg, BackendAuto: tc.auto, Quant: tc.quant})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if got := be.rf != nil; got != tc.built {
				t.Fatalf("BuildResident called = %v, want %v (decline %q)", got, tc.built, m.ResidentDecline())
			}
			if got := m.resident != nil; got != tc.kept {
				t.Fatalf("resident kept = %v, want %v (decline %q)", got, tc.kept, m.ResidentDecline())
			}
			if !tc.kept && !strings.Contains(m.ResidentDecline(), tc.why) {
				t.Errorf("decline %q, want it to contain %q", m.ResidentDecline(), tc.why)
			}
			// The banner names auto as the request: the user did not ask for metal by name.
			if dp := m.DecodePath(); !tc.kept && !strings.Contains(dp, "requested auto (metal) → running on cpu") {
				t.Errorf("DecodePath() = %q, want it to say auto chose metal and the model runs on the CPU", dp)
			}
		})
	}
}

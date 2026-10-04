package decoder

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestCompareLogits_judgment(t *testing.T) {
	want := []float32{1, -2, 3, 0.5, -0.25}
	if cos, rel := compareLogits(want, want); cos < 0.9999999 || rel != 0 {
		t.Errorf("identical vectors: cosine %v, relative L2 %v", cos, rel)
	}
	// A uniform scale is invisible to the cosine; the relative L2 is the half that sees it, so a missing logit scale cannot pass on the cosine alone.
	scaled := make([]float32, len(want))
	for i, v := range want {
		scaled[i] = 2 * v
	}
	cos, rel := compareLogits(scaled, want)
	if cos < probeCosineFloor || rel <= probeRelL2Ceil {
		t.Errorf("a 2x scale: cosine %v (should pass), relative L2 %v (should exceed %v)", cos, rel, probeRelL2Ceil)
	}
	// An all-zero row has no direction; it must read as a defect, not as a match.
	if cos, rel := compareLogits(make([]float32, len(want)), want); cos >= probeCosineFloor || !math.IsInf(rel, 1) {
		t.Errorf("all-zero logits: cosine %v, relative L2 %v; want a failing pair", cos, rel)
	}
	if cos, _ := compareLogits(want[:3], want); cos >= probeCosineFloor {
		t.Errorf("length mismatch passed (cosine %v)", cos)
	}
	if !nonFinite([]float32{1, float32(math.NaN())}) || !nonFinite([]float32{float32(math.Inf(-1))}) || nonFinite(want) {
		t.Error("nonFinite misjudges NaN, Inf or a finite row")
	}
}

// The embedded fixtures are copies of testdata/ ones; the copy exists only so a released binary can carry them, and must not drift from the originals the parity gates use.
func TestSelfTestFixtures_matchTestdataAndFitTheirWindow(t *testing.T) {
	for _, fx := range selfTestFixtures {
		ents, err := selfTestFS.ReadDir("selftestdata/" + fx.name)
		if err != nil || len(ents) == 0 {
			t.Fatalf("%s: no embedded files (%v)", fx.name, err)
		}
		hasWeights := false
		for _, e := range ents {
			emb, err := selfTestFS.ReadFile("selftestdata/" + fx.name + "/" + e.Name())
			if err != nil {
				t.Fatal(err)
			}
			orig, err := os.ReadFile(filepath.Join("..", "testdata", fx.name, e.Name()))
			if err != nil {
				t.Skipf("no testdata original for %s/%s: %v", fx.name, e.Name(), err)
			}
			if !bytes.Equal(emb, orig) {
				t.Errorf("%s/%s drifted from testdata/%s/%s; re-copy it", fx.name, e.Name(), fx.name, e.Name())
			}
			if e.Name() == "model.safetensors" {
				hasWeights = true
			}
			if e.Name() == "config.json" {
				var c map[string]any
				if err := json.Unmarshal(emb, &c); err != nil {
					t.Fatal(err)
				}
				if tc, ok := c["text_config"].(map[string]any); ok {
					c = tc
				}
				win, _ := c["max_position_embeddings"].(float64)
				if win > 0 && float64(fx.positions) > win {
					t.Errorf("%s: probes %d positions past its %v-position window; the resident refuses a position beyond its cap", fx.name, fx.positions, win)
				}
			}
		}
		if !hasWeights {
			t.Errorf("%s: no model.safetensors embedded (a .gitignore rule dropped it?)", fx.name)
		}
	}
}

// The calling convention under test is withResidency's: the probe is reached from a real Load, runs once per backend and quant, and a decline drops that model's resident. A resident that
// answers with a one-hot row (fakeResident) cannot agree with the CPU's logits on any fixture.
func TestGPUProbe_declinesAWrongResidentThroughLoad(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	var warns []string
	var mu sync.Mutex
	prevWarn := selfTestWarn
	selfTestWarn = func(m string) { mu.Lock(); warns = append(warns, m); mu.Unlock() }
	t.Cleanup(func() { selfTestWarn = prevWarn })
	resetSelfTestCaches()

	var builds atomic.Int64
	const name = "fake-probe-gpu"
	probeMeasured[name] = true
	t.Cleanup(func() { delete(probeMeasured, name) })
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be := &fakeNamedResidency{name: name}
		be.Backend = cpu
		return &countingBuild{fakeNamedResidency: be, n: &builds}, nil
	})

	m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.resident != nil {
		t.Fatal("a resident whose logits disagree with the CPU's everywhere stayed attached")
	}
	if d := m.ResidentDecline(); !strings.Contains(d, "startup self-test declined") {
		t.Errorf("decline reason %q does not name the self-test", d)
	}
	if got := int64(len(selfTestFixtures)) + 1; builds.Load() != got {
		t.Errorf("%d resident builds on first load; want the real model's plus one per fixture (%d)", builds.Load(), got)
	}
	var rec *SelfTestResult
	for _, r := range SelfTestResults() {
		if r.Backend == name {
			r := r
			rec = &r
		}
	}
	if rec == nil || rec.Status != SelfTestDeclined {
		t.Fatalf("recorded result %+v; want a declined one for %s", rec, name)
	}
	mu.Lock()
	nw := len(warns)
	mu.Unlock()
	if nw == 0 {
		t.Error("no WARN line for a declined backend")
	}

	// Cached: a second model on the same backend and quant adds exactly its own resident build and is declined from the cache.
	before := builds.Load()
	m2, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	if builds.Load()-before != 1 || m2.resident != nil {
		t.Errorf("second load: %d builds (want 1: no re-probe), resident attached = %v", builds.Load()-before, m2.resident != nil)
	}

	// -no-selftest: nothing is checked, so the (wrong) resident stays, which is exactly what the flag trades away.
	t.Cleanup(func() { selfTestSkipped.Store(false) })
	SkipSelfTests()
	m3, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int8"})
	if err != nil {
		t.Fatal(err)
	}
	defer m3.Close()
	if m3.resident == nil {
		t.Error("with self-tests skipped the resident was still dropped")
	}
}

type countingBuild struct {
	*fakeNamedResidency
	n *atomic.Int64
}

func (c *countingBuild) BuildResident(m *Model) (ResidentForward, bool, error) {
	c.n.Add(1)
	return c.fakeNamedResidency.BuildResident(m)
}

// With no resident on the backend (a CPU-only binary, or a name that is not compiled in) nothing was checked, and the result must SAY so rather than read as a pass.
func TestGPUProbe_aBackendWithNoResidentIsSkippedNotPassed(t *testing.T) {
	r := probeBackend("not-compiled-in", "int4")
	if r.Status != SelfTestSkipped || r.Note == "" {
		t.Errorf("status %s, note %q; want skipped with the reasons", r.Status, r.Note)
	}
	for _, fx := range selfTestFixtures {
		if !strings.Contains(r.Note, fx.name) {
			t.Errorf("note does not name fixture %s: %q", fx.name, r.Note)
		}
	}
}

// A backend whose margins nobody measured is not probed at all: a wrong resident stays attached and nothing is recorded, rather than the probe judging it against bars it was never seen to clear.
func TestGPUProbe_anUnmeasuredBackendIsNotProbed(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	var builds atomic.Int64
	const name = "fake-unmeasured-gpu"
	RegisterBackend(name, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be := &fakeNamedResidency{name: name}
		be.Backend = cpu
		return &countingBuild{fakeNamedResidency: be, n: &builds}, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.resident == nil || builds.Load() != 1 {
		t.Errorf("resident attached = %v, %d builds; want the one resident kept and no probe builds", m.resident != nil, builds.Load())
	}
	for _, r := range SelfTestResults() {
		if r.Backend == name {
			t.Errorf("a result was recorded for an unmeasured backend: %s", r.Summary())
		}
	}
	// The report path says so instead of staying silent.
	if r := probeStandalone(name); r.Status != SelfTestSkipped || !strings.Contains(r.Note, "no self-test for this backend yet") {
		t.Errorf("standalone probe of an unmeasured backend: %s", r.Summary())
	}
}

// A backend that names itself "webgpu:vulkan" is probed under its registry name, and one that declares itself ineligible (a software adapter) is skipped with its reason, keeps its resident, and is never recorded as a pass.
type ineligibleBackend struct {
	*fakeNamedResidency
	why string
}

func (b *ineligibleBackend) SelfTestEligible() (bool, string) { return false, b.why }

func TestGPUProbe_normalizesTheBackendNameAndHonoursIneligibility(t *testing.T) {
	t.Cleanup(resetSelfTestCaches)
	resetSelfTestCaches()
	if got := probeBackendName(&fakeNamedResidency{name: "webgpu:vulkan"}); got != "webgpu" {
		t.Errorf("probeBackendName(webgpu:vulkan) = %q, want webgpu", got)
	}
	const reg = "fake-soft-gpu"
	probeMeasured[reg] = true
	t.Cleanup(func() { delete(probeMeasured, reg) })
	RegisterBackend(reg, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be := &fakeNamedResidency{name: reg + ":software"}
		be.Backend = cpu
		return &ineligibleBackend{fakeNamedResidency: be, why: "software adapter: test"}, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: reg, Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.resident == nil {
		t.Error("an ineligible backend's resident was dropped; it should be left alone, unprobed")
	}
	var rec *SelfTestResult
	for _, r := range SelfTestResults() {
		if r.Backend == reg {
			r := r
			rec = &r
		}
	}
	if rec == nil || rec.Status != SelfTestSkipped || !strings.Contains(rec.Note, "software adapter") {
		t.Errorf("recorded %+v; want skipped with the adapter reason", rec)
	}
	if r := probeStandalone(reg); r.Status != SelfTestSkipped || !strings.Contains(r.Note, "software adapter") {
		t.Errorf("standalone probe of an ineligible backend: %s", r.Summary())
	}
}

// A resident that computes another precision than the load quant is not judged at the load quant (H2 on Metal: Metal
// re-quantizes int8int8 DeltaNet and MoE models to int4). Through Load, a fake GPU whose resident reports int4 for an
// int8int8 load is skipped fixture by fixture, recorded as skipped, not passed, and keeps its resident; the control, the
// same wrong resident reporting the load quant, is declined. residentPrecision also reads Metal's re-quantizing labels.
func TestGPUProbe_aReQuantizingResidentIsNotJudgedAtTheLoadQuant(t *testing.T) {
	for _, c := range []struct {
		backend, quant, report, want string
	}{
		{"metal", "int8int8", "", "int4"}, {"metal", "int8", "", "int4"}, {"metal", "int4mix", "", "int4"},
		{"metal", "int8int8", "int8int8", "int8int8"}, {"metal", "int4", "", "int4"}, {"cuda", "int8int8", "", "int8int8"},
	} {
		var rf ResidentForward = &fakeResident{}
		if c.report != "" {
			rf = quantReporting{rf, c.report}
		}
		if got := residentPrecision(c.backend, c.quant, rf); got != c.want {
			t.Errorf("residentPrecision(%s, %s, reports %q) = %q, want %q", c.backend, c.quant, c.report, got, c.want)
		}
	}
	for _, c := range []struct {
		report string
		want   string
		kept   bool
	}{{"int4", SelfTestSkipped, true}, {"int8int8", SelfTestDeclined, false}} {
		t.Run("reports "+c.report, func(t *testing.T) {
			t.Cleanup(resetSelfTestCaches)
			resetSelfTestCaches()
			prevWarn := selfTestWarn
			selfTestWarn = func(string) {}
			t.Cleanup(func() { selfTestWarn = prevWarn })
			name := "fake-requant-gpu-" + c.report
			probeMeasured[name] = true
			t.Cleanup(func() { delete(probeMeasured, name) })
			RegisterBackend(name, func() (Backend, error) {
				cpu, err := NewBackend("cpu")
				if err != nil {
					return nil, err
				}
				be := &fakeNamedResidency{name: name, report: c.report}
				be.Backend = cpu
				return be, nil
			})
			m, err := Load(tinyFixture(t), Options{Backend: name, Quant: "int8int8"})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if (m.resident != nil) != c.kept {
				t.Errorf("resident attached = %v, want %v (%s)", m.resident != nil, c.kept, m.ResidentDecline())
			}
			var rec *SelfTestResult
			for _, r := range SelfTestResults() {
				if r.Backend == name {
					r := r
					rec = &r
				}
			}
			if rec == nil || rec.Status != c.want {
				t.Fatalf("recorded %+v; want status %s", rec, c.want)
			}
			if c.want == SelfTestSkipped {
				for _, fx := range selfTestFixtures {
					if !strings.Contains(rec.Note, fx.name+": re-quantized to int4") {
						t.Errorf("note does not say %s was re-quantized and not probed: %q", fx.name, rec.Note)
					}
				}
			}
		})
	}
}

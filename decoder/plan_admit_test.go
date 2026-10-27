package decoder

import (
	"fmt"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// planVerdict is a Plan eligibility verdict on backend be for a loaded model: true when Plan would decline before pricing.
type planVerdict func(m *Model, be string) bool

// planAgreement compares verdict with the decoder's admission on layer D's fixtures and option cells (decoder's TestAdmissionCensus).
// The verdict comes from a CPU load with the cell's options and the KV request dropped, since admission does not read the KV. The
// admission is whether a load under a fake backend of that name reached the backend's build. It returns one line per disagreement.
// A cell whose CPU load fails has no model for Plan to judge; it is counted in noModel, and its fake loads must not reach a build.
func planAgreement(t *testing.T, verdict planVerdict) (diffs []string, cells, noModel int) {
	t.Helper()
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "")
	t.Setenv("GOINFER_SSM_RESIDENT", "")
	useProbe(t, &probe.Fake{PhysicalBytes: 64 << 30, AvailableBytes: 48 << 30})
	backends := []string{"cuda", "metal", "webgpu"}
	fakes := map[string]*namedResidencyBackend{}
	for _, be := range backends {
		reg := "plan-admit-fake-" + be
		RegisterBackend(reg, func() (Backend, error) {
			cpu, err := NewBackend("cpu")
			if err != nil {
				return nil, err
			}
			fakes[be] = &namedResidencyBackend{Backend: cpu, name: be}
			return fakes[be], nil
		})
		t.Cleanup(func() {
			backendMu.Lock()
			defer backendMu.Unlock()
			delete(backendRegistry, reg)
		})
	}
	for _, f := range sb2Fixtures(t) {
		for _, cell := range sb2Cells {
			opts := func(be string) Options {
				o := Options{Backend: be, Quant: cell.quant, ActQuantGroup: cell.actGroup, noSelfTest: true}
				if cell.ssm != "" {
					o.Knobs = &Knobs{knobSSMResident: cell.ssm}
				}
				return o
			}
			cpuM, cpuErr := Load(f.path, opts("cpu"))
			if cpuErr != nil {
				noModel++
			}
			for _, be := range backends {
				cells++
				o := opts("plan-admit-fake-" + be)
				o.KVPrecision, o.KVQuant = cell.kvPrec, cell.kvQuant
				fakes[be] = nil
				m, err := Load(f.path, o)
				if err == nil {
					m.Close()
				}
				if fakes[be] == nil {
					t.Fatalf("%s %s %s: the fake backend was never made", f.name, cell.name, be)
				}
				admitted := fakes[be].built.Load() > 0
				if cpuErr != nil {
					if admitted {
						diffs = append(diffs, fmt.Sprintf("%s %s %s: no CPU model for Plan (%v), but the load reached the build", f.name, cell.name, be, cpuErr))
					}
					continue
				}
				if declined := verdict(cpuM, be); declined == admitted {
					diffs = append(diffs, fmt.Sprintf("%s %s %s: Plan declines=%v, the load's admission admits=%v", f.name, cell.name, be, declined, admitted))
				}
			}
			if cpuErr == nil {
				cpuM.Close()
			}
		}
	}
	return diffs, cells, noModel
}

// TestPlanAgreesWithAdmission holds Plan's eligibility to the decoder's admission on every census cell (SB2,
// docs/tasks/task-subsystems-2026-10.md, "SB2 — Plan asks Admit"), and shows the comparison goes red on the verdict Plan used
// before it asked Admit.
func TestPlanAgreesWithAdmission(t *testing.T) {
	slowTest(t)
	// Plan's admission step on every backend; on cuda and metal it is the whole eligibility check. webgpu's model-only declines,
	// which Plan mirrors on top, are held to the load in gpu's TestPlanAgreesWithLoad_webgpu.
	diffs, cells, noModel := planAgreement(t, func(m *Model, be string) bool {
		why := m.planAdmitDecline(be)
		if full := m.planIneligible(be, PlanRequest{}); be != "webgpu" && full != why {
			t.Errorf("on %s: Plan's eligibility %q is not its admission step %q", be, full, why)
		}
		return why != ""
	})
	if len(diffs) > 0 {
		t.Errorf("Plan disagrees with the load's admission in %d of %d cells:\n%s", len(diffs), cells, strings.Join(diffs[:min(len(diffs), 30)], "\n"))
	}
	t.Logf("%d cells: Plan and the load's admission agree (%d with no CPU model, compared as declined)", cells, noModel*3)

	// The plant: the arch-only verdict Plan read before (features, then ResidentEligible).
	old, _, _ := planAgreement(t, func(m *Model, be string) bool {
		return len(m.MissingResidentFeatures(ResidentBackendFeatures(be))) > 0 || !ResidentEligible(m.w.arch, be)
	})
	if len(old) == 0 {
		t.Fatal("the old arch-only verdict agrees everywhere: the comparison cannot go red")
	}
	nemo := 0
	for _, d := range old {
		if strings.HasPrefix(d, "testdata/nemotron3nano-tiny ") && strings.Contains(d, " webgpu:") {
			nemo++
		}
	}
	if nemo == 0 {
		t.Errorf("the old verdict's %d disagreements do not include Nemotron 3 Nano on webgpu", len(old))
	}
	t.Logf("planted old verdict: %d disagreements, %d of them Nemotron 3 Nano on webgpu:\n%s", len(old), nemo, strings.Join(old, "\n"))
}

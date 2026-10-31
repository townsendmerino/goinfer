package decoder

import (
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/program"
)

// TestMetalPricing_followsThePathThatRuns (OP4m-c2; the coordinator's approval of 2026-10-31 07:58): the fit guard prices a Metal
// load as resident only when BuildQuery.FeaturesCovered, which reads Metal's table. Llama 4 and Nemotron 3 Nano run on the CPU
// under Metal (the runner shape, and Nemotron's MoE block), and the table declines their operations, so they are priced as the
// CPU path (f32 KV, the whole context), not as Metal-resident (f16 KV, a default 4096). Plain Llama stays priced resident.
func TestMetalPricing_followsThePathThatRuns(t *testing.T) {
	opts := Options{Backend: "metal"}
	nemo, err := loadConfig(os.DirFS("../testdata/nemotron3nano-tiny"), "config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name     string
		cfg      *Config
		resident bool
	}{
		{"llama (control)", representativeConfig("llama"), true},
		{"llama4_text", representativeConfig("llama4_text"), false},
		{"nemotron3nano-tiny", nemo, false},
	} {
		q := BuildQueryFor(c.cfg, opts, 1<<20, 0, false)
		if q.FeaturesCovered != c.resident {
			t.Errorf("%s on Metal: FeaturesCovered %v, want %v (priced as %s)", c.name, q.FeaturesCovered, c.resident,
				map[bool]string{true: "Metal-resident", false: "the CPU path it runs"}[c.resident])
		}
		if d := defaultMetalBuildDemand(q); d.Resident && !c.resident {
			t.Errorf("%s on Metal: the build demand prices it resident, but its load runs on the CPU", c.name)
		}
	}
}

// TestMetalTable_plantedGapGoesRed is OP4m-c2's C2-3 (docs/tasks/task-forward-architecture-2026-10.md §4.9): Metal's
// admission reads program.Metal, so a copy of the table without GELU-tanh on the gated FFN form, put in place for Metal, must
// decline every admitted Gemma row (a gated GELU-tanh family the golden admits on Metal) and leave a SiLU row admitted.
func TestMetalTable_plantedGapGoesRed(t *testing.T) {
	planted := *program.Metal
	planted.FFN = map[program.FFNForm][]program.ActKind{}
	for f, acts := range program.Metal.FFN {
		planted.FFN[f] = acts
	}
	planted.FFN[program.FfnGated] = []program.ActKind{program.ActSiLU}
	prev := residentBackendTables["metal"]
	residentBackendTables["metal"] = &planted
	t.Cleanup(func() { residentBackendTables["metal"] = prev })

	red, silu := 0, 0
	for _, name := range slices.Sorted(mapKeys(admissionGolden)) {
		if !slices.Contains(admissionGolden[name], "metal") {
			continue
		}
		a, _, err := resolveArchitecture(representativeConfig(name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		gated := !a.NonGatedMLP && a.MoE == nil && a.gptoss == nil && a.gemma4 == nil
		switch {
		case gated && a.Act == ActGeluTanh:
			if ResidentEligible(a, "metal") {
				t.Errorf("%s: admitted on Metal by a table without GELU-tanh on the gated form", name)
			} else {
				red++
				t.Logf("%s: declined under the plant: %s", name, residentGateReason(a, "metal"))
			}
		case gated && a.Act == ActSiLU && silu == 0:
			if !ResidentEligible(a, "metal") {
				t.Errorf("%s: a SiLU row declined under a plant that removes only GELU-tanh: %s", name, residentGateReason(a, "metal"))
			}
			silu++
		}
	}
	if red == 0 || silu == 0 {
		t.Fatalf("plant red on %d Gemma rows, %d SiLU control rows: the plant must reach both", red, silu)
	}
}

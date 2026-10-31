package decoder

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/program"
)

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

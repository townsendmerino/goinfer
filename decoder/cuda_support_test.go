package decoder

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/program"
)

// configFixtureArchs is every tracked fixture config under ../testdata that resolves, by its directory relative to the repo
// root: the configs TestProgramFingerprint_table builds programs for.
func configFixtureArchs(t *testing.T) map[string]*Architecture {
	t.Helper()
	out := map[string]*Architecture{}
	err := filepath.WalkDir("../testdata", func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || d.Name() != "config.json" {
			return err
		}
		if exec.Command("git", "check-ignore", "-q", path).Run() == nil {
			return nil
		}
		dir := filepath.Dir(path)
		cfg, err := loadConfig(os.DirFS(dir), "config.json")
		if err != nil {
			return nil
		}
		a, _, err := resolveArchitecture(cfg)
		if err != nil {
			return nil
		}
		if tied, ok := configTie(dir); ok {
			a.TiedLMHead = tied
		}
		out[strings.TrimPrefix(filepath.ToSlash(dir), "../")] = a
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) < 20 {
		t.Fatalf("only %d fixture configs resolved: a comparison over few configs proves little", len(out))
	}
	return out
}

// TestCUDATable_plantedGapGoesRed is OP4cu-c2's C2-3 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): CUDA's
// admission reads program.CUDA (c2 replaced c1's C1-2, which compared the table with the feature set it now stands in for), so a
// copy of the table without the tanh GELU on the gated FFN form, put in place for CUDA, must decline every admitted Gemma row (a
// gated GELU-tanh family the golden admits on CUDA) and leave a SiLU row admitted.
func TestCUDATable_plantedGapGoesRed(t *testing.T) {
	planted := *program.CUDA
	planted.FFN = map[program.FFNForm][]program.ActKind{}
	for f, acts := range program.CUDA.FFN {
		planted.FFN[f] = acts
	}
	planted.FFN[program.FfnGated] = []program.ActKind{program.ActSiLU}
	prev := residentBackendTables["cuda"]
	residentBackendTables["cuda"] = &planted
	t.Cleanup(func() { residentBackendTables["cuda"] = prev })

	red, silu := 0, 0
	for _, name := range slices.Sorted(mapKeys(admissionGolden)) {
		if !slices.Contains(admissionGolden[name], "cuda") {
			continue
		}
		a, _, err := resolveArchitecture(representativeConfig(name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		gated := !a.NonGatedMLP && a.MoE == nil && a.gptoss == nil && a.gemma4 == nil
		switch {
		case gated && a.Act == ActGeluTanh:
			if ResidentEligible(a, "cuda") {
				t.Errorf("%s: admitted on CUDA by a table without GELU-tanh on the gated form", name)
			} else {
				red++
				t.Logf("%s: declined under the plant: %s", name, residentGateReason(a, "cuda"))
			}
		case gated && a.Act == ActSiLU && silu == 0:
			if !ResidentEligible(a, "cuda") {
				t.Errorf("%s: a SiLU row declined under a plant that removes only GELU-tanh: %s", name, residentGateReason(a, "cuda"))
			}
			silu++
		}
	}
	if red == 0 || silu == 0 {
		t.Fatalf("plant red on %d Gemma rows, %d SiLU control rows: the plant must reach both", red, silu)
	}
}

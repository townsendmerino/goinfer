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

// metalAdmitsWithTable is Metal's model-free admission with its feature gate replaced by the table s: every refusal Admit makes
// before or beside the feature gate stands, and the program must pass s.
func metalAdmitsWithTable(a *Architecture, s *program.Support) (bool, string) {
	ad := Admit(archAdmitFacts(a, "metal"))
	why := ad.Why
	if len(ad.Gates) > 0 && why == ad.Gates[0].Reason {
		why = ""
		for _, g := range ad.Gates {
			if g.Kind != residentGateFeatures {
				why = g.Reason
				break
			}
		}
	}
	if why != "" {
		return false, why
	}
	if err := s.Check(buildProgram(a, configExperts(a))); err != nil {
		return false, err.Error()
	}
	return true, ""
}

// TestMetalSupport_replacesTheFeatures is OP4m-c1's C1-2 (docs/tasks/task-forward-architecture-2026-10.md §4.9): over every
// fixture config, Metal's admission with its feature gate replaced by program.Metal admits exactly what today's admits, except
// Olmo Hybrid, which the table declines for DeltaNet's widened write gate. The plant (the table admits the exact-erf GELU on a
// gated FFN) must turn it red on at least one admitted SiLU-gated config read with that activation.
func TestMetalSupport_replacesTheFeatures(t *testing.T) {
	archs := configFixtureArchs(t)
	changed := map[string]bool{"olmo_hybrid": true}
	planted := *program.Metal
	planted.FFN = map[program.FFNForm][]program.ActKind{}
	for f, acts := range program.Metal.FFN {
		planted.FFN[f] = acts
	}
	planted.FFN[program.FfnGated] = append(slices.Clone(program.Metal.FFN[program.FfnGated]), program.ActGelu)
	agree, red, erf := 0, 0, 0
	for dir, a := range archs {
		today := Admit(archAdmitFacts(a, "metal")).Why == ""
		got, why := metalAdmitsWithTable(a, program.Metal)
		switch {
		case changed[a.Name]:
			if !today || got {
				t.Errorf("%s (%s): admitted today %v, by the table %v; want admitted today and declined by the table", dir, a.Name, today, got)
			} else {
				t.Logf("%s (%s): admitted today, declined by the table: %s", dir, a.Name, why)
			}
		case got != today:
			t.Errorf("%s (%s): admitted today %v, by the table %v (%s)", dir, a.Name, today, got, why)
		default:
			agree++
		}
		// No fixture has a gated exact-erf GELU, so each admitted SiLU-gated config is also read with one: today's features and
		// the table must both decline it, and the planted table admits it.
		if today && got && a.Act == ActSiLU && !a.NonGatedMLP {
			b := *a
			b.Act = ActGelu
			erf++
			erfToday := Admit(archAdmitFacts(&b, "metal")).Why == ""
			erfGot, _ := metalAdmitsWithTable(&b, program.Metal)
			if erfToday || erfGot {
				t.Errorf("%s with a gated exact-erf GELU: admitted today %v, by the table %v; want both to decline", dir, erfToday, erfGot)
			}
			if p, _ := metalAdmitsWithTable(&b, &planted); p {
				red++
			}
		}
	}
	t.Logf("%d of %d configs agree (beside the named change)", agree, len(archs))
	if t.Logf("plant (the table admits the exact-erf GELU on a gated FFN): red on %d of %d configs read with one", red, erf); red == 0 {
		t.Errorf("plant (the table admits the exact-erf GELU on a gated FFN): no config goes red")
	}
}

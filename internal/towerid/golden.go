package towerid

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/kernelprint"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// Golden is one backend's committed record on one platform, with the anchor its bits depend on (SB9's goldens,
// docs/tasks/task-subsystems-2026-10.md, "SB9's gates").
type Golden struct {
	Schema int    `json:"schema"`
	Anchor string `json:"anchor"`
	Record Record `json:"record"`
}

// OnOtherAnchor is what CheckGolden does when this machine's anchor is not the golden's.
type OnOtherAnchor int

const (
	// SkipOtherAnchor: a different anchor is a configuration nobody has recorded yet (the CPU's kernel set on a CI runner).
	SkipOtherAnchor OnOtherAnchor = iota
	// FailOtherAnchor: a different anchor is re-anchored deliberately, never carried forward (a device's driver or adapter).
	FailOtherAnchor
	// FailOtherAnchorSkipOnCI is FailOtherAnchor, except on CI (GITHUB_ACTIONS=true), where a different anchor skips: a CI
	// runner's virtual adapter has no golden of its own (docs/tasks/task-subsystems-2026-10.md, "SB9 step S on the Mac").
	FailOtherAnchorSkipOnCI
)

// CPUAnchor is what the CPU record's bits depend on: GOARCH and linalg's kernel selection (kernelprint.Config).
func CPUAnchor() string { return kernelprint.CurrentConfig().String() }

// GoldenPath is backend's golden on this platform, under the repository root.
func GoldenPath(root, backend string) string {
	return filepath.Join(root, "internal", "towerid", "testdata", fmt.Sprintf("towers.%s-%s-%s.json", backend, runtime.GOOS, runtime.GOARCH))
}

// CheckGolden grades rec, recorded under anchor, against its backend's golden case for case, exactly (G-SB9-1). With update
// it first writes rec as the golden. A platform without a golden skips, printing the record to commit; a different anchor
// skips or fails as onOther says. It returns the golden's record.
func CheckGolden(t testing.TB, root string, rec Record, anchor string, onOther OnOtherAnchor, update bool) Record {
	t.Helper()
	path := GoldenPath(root, rec.Backend)
	if update {
		j, err := json.MarshalIndent(Golden{Schema: 1, Anchor: anchor, Record: rec}, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(j, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s (review the diff)", path)
	}
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("no golden %s for this platform yet: a session on such a machine records it with -towerid.update and reviews it", path)
	}
	if err != nil {
		t.Fatal(err)
	}
	var g Golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	if g.Anchor != anchor {
		msg := fmt.Sprintf("%s is anchored to %q; this machine is %q", path, g.Anchor, anchor)
		switch {
		case onOther == SkipOtherAnchor:
			t.Skipf("%s: an unrecorded configuration", msg)
		case onOther == FailOtherAnchorSkipOnCI && os.Getenv("GITHUB_ACTIONS") == "true":
			t.Skipf("%s: a CI runner's adapter, which has no golden of its own (SB9 option (a))", msg)
		}
		t.Fatalf("%s: re-anchor it deliberately (-towerid.update, reviewed), never carry it forward", msg)
	}
	for _, d := range diffCases(g.Record.Cases, rec.Cases) {
		t.Errorf("%s against %s: %s", rec.Backend, path, d)
	}
	return g.Record
}

// diffCases lists how got differs from want, by family and input.
func diffCases(want, got []Case) []string {
	w, g := map[string]Case{}, map[string]Case{}
	for _, c := range want {
		w[caseKey(c.Family, c.Input)] = c
	}
	for _, c := range got {
		g[caseKey(c.Family, c.Input)] = c
	}
	var out []string
	for k, wc := range w {
		gc, ok := g[k]
		switch {
		case !ok:
			out = append(out, fmt.Sprintf("%s %s: missing", wc.Family, wc.Input))
		case gc != wc:
			out = append(out, fmt.Sprintf("%s %s: got device=%v values=%d %s, want device=%v values=%d %s", wc.Family, wc.Input,
				gc.Device, gc.Values, gc.SHA256, wc.Device, wc.Values, wc.SHA256))
		}
	}
	for k, gc := range g {
		if _, ok := w[k]; !ok {
			out = append(out, fmt.Sprintf("%s %s: not in the golden", gc.Family, gc.Input))
		}
	}
	slices.Sort(out)
	return out
}

// CheckPlants is G-SB9-2: for each family in golden, a one-ulp plant in that family's output (PlantForTest) turns exactly
// that family's cases red against golden and leaves every other case equal.
func CheckPlants(t testing.TB, root, backend string, golden Record) {
	t.Helper()
	defer func() { PlantForTest = "" }()
	var families []string
	for _, c := range golden.Cases {
		if !slices.Contains(families, c.Family) {
			families = append(families, c.Family)
		}
	}
	for _, f := range families {
		PlantForTest = f
		rec, err := Run(root, backend)
		PlantForTest = ""
		if err != nil {
			t.Fatalf("%s, planted in %s: %v", backend, f, err)
		}
		red := map[string]bool{}
		for _, c := range golden.Cases {
			for _, r := range rec.Cases {
				if r.Family == c.Family && r.Input == c.Input && r != c {
					red[caseKey(c.Family, c.Input)] = true
				}
			}
		}
		for _, c := range golden.Cases {
			if want := c.Family == f; red[caseKey(c.Family, c.Input)] != want {
				t.Errorf("%s, planted in %s: case %s %s red=%v, want %v", backend, f, c.Family, c.Input, !want, want)
			}
		}
	}
	t.Logf("%s: each of %d families' plants turned exactly its own cases red", backend, len(families))
}

// programFamilies are the families whose blocks run each tower program, by the program's name, on whichever backend
// runs them.
var programFamilies = map[string][]string{
	"siglip":       {"siglip", "siglip2-naflex"},
	"qwen3":        {"qwen3", "qwen3-deepstack"},
	"glm-ocr":      {"glm-ocr"},
	"gemma4":       {"gemma4-vision"},
	"qwen2.5":      {"qwen2.5-vl"},
	"pixtral":      {"pixtral"},
	"gemma4-audio": {"gemma4-audio"},
	"qwen3-asr":    {"qwen3-asr"},
}

func isActivation(op vision.TowerOp) bool {
	return op.Kind == vision.TowerGELUTanh || op.Kind == vision.TowerSiLUMul || op.Kind == vision.TowerGELUMul ||
		op.Kind == vision.TowerSiLU || op.Kind == vision.TowerGELU
}

// programPlants are G-SB9M-2's plants in a tower program (docs/tasks/task-subsystems-2026-10.md §8, "SB9 step M"). The
// epsilon is scaled by 2^20, exact in float32 and float64, because a one-ulp move is lost in variance+eps. The
// activation swap reads a register its linear has not yet written (on SigLIP and Qwen3 it equals the activation
// dropped), so proj's swap with the second norm is the order plant that reads only written registers.
var programPlants = []struct {
	name  string
	plant func(vision.TowerBlock) vision.TowerBlock
}{
	{"the activation dropped", func(p vision.TowerBlock) vision.TowerBlock {
		return slices.DeleteFunc(p, isActivation)
	}},
	{"the activation swapped with the linear before it", func(p vision.TowerBlock) vision.TowerBlock {
		for i := 1; i < len(p); i++ {
			if isActivation(p[i]) && p[i-1].Kind == vision.TowerLinear {
				p[i-1], p[i] = p[i], p[i-1]
				return p
			}
		}
		panic("towerid: the program has no activation after a linear")
	}},
	{"proj's residual add swapped with the second norm", func(p vision.TowerBlock) vision.TowerBlock {
		// proj's residual add is its linear-add, or, where a norm sits between proj and its add (Gemma 4's sandwich), the
		// block's first add.
		firstAdd := slices.IndexFunc(p, func(op vision.TowerOp) bool { return op.Kind == vision.TowerAdd })
		for i := 1; i < len(p); i++ {
			projAdd := (p[i-1].Kind == vision.TowerLinearAdd && p[i-1].W == vision.TowerWO) || i-1 == firstAdd
			if projAdd && p[i].Kind == vision.TowerNorm {
				p[i-1], p[i] = p[i], p[i-1]
				return p
			}
		}
		panic("towerid: the program has no proj residual add before a norm")
	}},
	{"the first norm's epsilon scaled by 2^20", func(p vision.TowerBlock) vision.TowerBlock {
		for i := range p {
			if p[i].Kind == vision.TowerNorm {
				p[i].Eps *= 1 << 20
				return p
			}
		}
		panic("towerid: the program has no norm")
	}},
}

// CheckProgramPlants is G-SB9M-2: each plant, in one of the tower programs the backend walks at a time
// (vision.TowerProgramPlantForTest), turns every case of a family that walks that program red against golden and
// leaves every other case equal (programPlantVerdict). programs names them (programFamilies' keys); none is an error.
func CheckProgramPlants(t testing.TB, root, backend string, golden Record, programs ...string) {
	t.Helper()
	defer func() { vision.TowerProgramPlantForTest = nil }()
	if len(programs) == 0 {
		t.Fatalf("%s: CheckProgramPlants names no program", backend)
	}
	for _, prog := range programs {
		fams, ok := programFamilies[prog]
		if !ok {
			t.Fatalf("%s: no tower program %q", backend, prog)
		}
		var covered []string
		for _, pl := range programPlants {
			vision.TowerProgramPlantForTest = func(name string, p vision.TowerBlock) vision.TowerBlock {
				if name != prog {
					return p
				}
				return pl.plant(p)
			}
			rec, err := Run(root, backend)
			vision.TowerProgramPlantForTest = nil
			if err != nil {
				t.Fatalf("%s, %s in %s: %v", backend, pl.name, prog, err)
			}
			var errs []string
			errs, covered = programPlantVerdict(backend, fams, golden, rec)
			for _, e := range errs {
				t.Errorf("%s, %s in %s: %s", backend, pl.name, prog, e)
			}
		}
		t.Logf("%s: each of %d plants in program %s turned exactly the %v cases red", backend, len(programPlants), prog, covered)
	}
}

// programPlantVerdict compares one planted run's record with golden: every case of one of fams (the planted program's
// families) must be red and every other case equal; golden must hold a case of at least one of fams; off the CPU, each
// such family needs a red case that ran on the device. covered lists the families of fams golden has cases of, which is
// what the check vouches for on this backend.
func programPlantVerdict(backend string, fams []string, golden, rec Record) (errs, covered []string) {
	got := map[string]Case{}
	for _, r := range rec.Cases {
		got[caseKey(r.Family, r.Input)] = r
	}
	redOnDevice := map[string]bool{} // the program's families with a red case that ran on the device
	for _, c := range golden.Cases {
		r, ok := got[caseKey(c.Family, c.Input)]
		red := !ok || r != c
		want := slices.Contains(fams, c.Family)
		if red != want {
			errs = append(errs, fmt.Sprintf("case %s %s red=%v, want %v", c.Family, c.Input, red, want))
		}
		if want && !slices.Contains(covered, c.Family) {
			covered = append(covered, c.Family)
		}
		if want && red && c.Device {
			redOnDevice[c.Family] = true
		}
	}
	if len(covered) == 0 {
		errs = append(errs, fmt.Sprintf("the golden has no case of %v, so no plant can show the program was read", fams))
	}
	if backend != "cpu" {
		for _, f := range covered {
			if !redOnDevice[f] {
				errs = append(errs, fmt.Sprintf("no %s case that ran on the device went red, so no device walker read the program for it", f))
			}
		}
	}
	return errs, covered
}

// The bound G-SB9-3 registered for a device output against the CPU's: an expectation, not a measured floor.
const (
	BackendMinCosine = 0.999
	BackendMaxRelErr = 0.05
)

// CompareBackends is G-SB9-3: each of backend's device cases against the CPU's output on the same input, by cosine and by
// max abs difference over the CPU output's RMS. A case outside the registered bound is a finding (DECISION NEEDED).
func CompareBackends(t testing.TB, root, backend string) {
	t.Helper()
	_, cpu, err := RunOutputs(root, "cpu")
	if err != nil {
		t.Fatal(err)
	}
	rec, dev, err := RunOutputs(root, backend)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range rec.Cases {
		if !c.Device {
			continue
		}
		k := caseKey(c.Family, c.Input)
		a, b := cpu[k], dev[k]
		if len(a) != len(b) || len(a) == 0 {
			t.Errorf("%s %s: %d device values against %d on the CPU", c.Family, c.Input, len(b), len(a))
			continue
		}
		var dot, na, nb, maxAbs float64
		for i := range a {
			x, y := float64(a[i]), float64(b[i])
			dot, na, nb = dot+x*y, na+x*x, nb+y*y
			maxAbs = max(maxAbs, math.Abs(x-y))
		}
		cos := dot / math.Sqrt(na*nb)
		rel := maxAbs / math.Sqrt(na/float64(len(a)))
		t.Logf("%s %-18s %-28s cosine %.9f  max|d|/rms %.3g", backend, c.Family, c.Input, cos, rel)
		if !(cos >= BackendMinCosine) || !(rel <= BackendMaxRelErr) {
			t.Errorf("DECISION NEEDED: %s %s %s differs from the CPU beyond the registered bound (cosine %.6f against %.3f, "+
				"max|d|/rms %.3g against %.2f); record it, do not move the bound", backend, c.Family, c.Input, cos,
				BackendMinCosine, rel, BackendMaxRelErr)
		}
	}
}

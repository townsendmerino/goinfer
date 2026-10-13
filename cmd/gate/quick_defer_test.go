package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// cellsRunning counts, for each of the root's tests, the cells that run it among those the day loop runs (deferred
// false) or leaves (deferred true).
func cellsRunning(cells []*quickCell, test string, deferred bool) int {
	n := 0
	for _, c := range cells {
		if (c.Deferred != "") != deferred {
			continue
		}
		if c.Run != "" && !regexp.MustCompile(c.Run).MatchString(test) {
			continue
		}
		if c.Skip != "" && regexp.MustCompile(c.Skip).MatchString(test) {
			continue
		}
		n++
	}
	return n
}

// Rule 1: the [checkpoint] cell is deferred and the rest of the package is not; a package too small to split keeps
// its loaders in the one cell, which still runs.
func TestQuick_deferCells_checkpointCell(t *testing.T) {
	qc, r := ckFixture(t)
	g := &quickGraph{roots: []*testRoot{r}}
	sel := &quickSelection{Changed: []string{"ck/arch.go"}, Affected: map[*testRoot][]string{r: {"compiles ck/arch.go"}}}
	cells := deferCells(qc, sel, buildCells(qc, g, sel, false, false))
	if len(cells) != 2 {
		t.Fatalf("%d cells, want the [checkpoint] cell and the rest", len(cells))
	}
	for _, c := range cells {
		if want := c.Name == "ck [checkpoint]"; (c.Deferred != "") != want {
			t.Errorf("cell %q: deferred %q, want deferred = %v", c.Name, c.Deferred, want)
		}
	}
	loader := map[string]bool{}
	for _, l := range r.loaders {
		loader[l] = true
	}
	for _, n := range r.tests {
		run, left := cellsRunning(cells, n, false), cellsRunning(cells, n, true)
		if run+left != 1 || (left == 1) != loader[n] {
			t.Errorf("%s: runs in %d cell(s), deferred in %d; want one, deferred only for a loader", n, run, left)
		}
	}

	qc2, r2 := ckFixture(t)
	small := &quickCell{Name: "ck", Root: r2, Tier: "affected", Mem: true}
	if out := deferCells(qc2, sel, []*quickCell{small}); len(out) != 1 || out[0].Deferred != "" {
		t.Errorf("an unsplit package with loaders was deferred: %+v", out)
	}
}

// Rule 2: a device module the diff reaches only from outside defers its Slow tests in a [slow] cell and runs the
// rest; touched, or only cache-checked, it is left whole. Every test still belongs to exactly one cell.
func TestQuick_deferCells_slowOfAnUntouchedDeviceModule(t *testing.T) {
	build := func(changed, tier string) ([]*quickCell, *testRoot) {
		qc, r := ckFixture(t)
		dev := &quickModule{Name: "dev", Dir: "ck", Device: true, Slow: `^(TestPlain00|TestPlain01)$`}
		qc.Modules = append(qc.Modules, dev)
		r.Mod = dev
		sel := &quickSelection{Changed: []string{changed}}
		if tier == "affected" {
			sel.Affected = map[*testRoot][]string{r: {"imports elsewhere"}}
		}
		return deferCells(qc, sel, buildCells(qc, &quickGraph{roots: []*testRoot{r}}, sel, false, false)), r
	}
	names := func(cells []*quickCell) (out []string) {
		for _, c := range cells {
			out = append(out, c.Name)
		}
		return out
	}

	cells, r := build("elsewhere/x.go", "affected")
	var slow *quickCell
	for _, c := range cells {
		if c.Name == "ck [slow]" {
			slow = c
		}
	}
	if slow == nil || slow.Deferred == "" {
		t.Fatalf("no deferred [slow] cell for a device module reached from outside: %v", names(cells))
	}
	for _, n := range r.tests {
		run, left := cellsRunning(cells, n, false), cellsRunning(cells, n, true)
		isSlow := n == "TestPlain00" || n == "TestPlain01"
		isLoader := false
		for _, l := range r.loaders {
			isLoader = isLoader || l == n
		}
		if run+left != 1 || (left == 1) != (isSlow || isLoader) {
			t.Errorf("%s: runs in %d cell(s), deferred in %d; want one, deferred only for a slow test or a loader", n, run, left)
		}
	}

	if cells, _ := build("ck/arch.go", "affected"); strings.Contains(strings.Join(names(cells), ","), "[slow]") {
		t.Errorf("a touched device module had its slow tests deferred: %v", names(cells))
	}
	if cells, _ := build("elsewhere/x.go", "cache-checked"); strings.Contains(strings.Join(names(cells), ","), "[slow]") {
		t.Errorf("a cache-checked device cell was split: %v", names(cells))
	}
}

// Through quickMain: a change outside a device module that breaks only one of its Slow tests is green by day, with
// the deferral printed, and red under -full and once the module itself is touched. So a deferred cell is neither
// run by day nor lost.
func TestQuick_deferredSlowTestRunsUnderFullAndWhenTouched(t *testing.T) {
	qc := fxRepo(t)
	qc.Modules[1].Device, qc.Modules[1].Slow = true, `^TestLeafFromSub$`
	run := func(full bool) (int, string) {
		var out bytes.Buffer
		rc := quickMain(qc, quickOpts{Base: "HEAD", Jobs: 4, Timeout: "5m", LogDir: t.TempDir(), NoLint: true, Full: full,
			Progress: io.Discard}, &out)
		return rc, regexp.MustCompile("\x1b\\[[0-9;]*m").ReplaceAllString(out.String(), "")
	}
	if rc, out := run(true); rc != 0 {
		t.Fatalf("baseline must be GREEN under -full, rc=%d:\n%s", rc, out)
	}
	// Wrong only for Add(1, 1), which the sub module's test is the one to call.
	mutate(t, qc, "leaf/leaf.go", "a + b", "a + b + map[bool]int{true: 1}[a == 1 && b == 1]")
	rc, out := run(false)
	if rc != 0 {
		t.Fatalf("day tier: rc=%d, want 0 (the only failing test is deferred):\n%s", rc, out)
	}
	for _, want := range []string{"DEFERRED sub [slow]", "DEFERRED to the pre-tag tier: 1 cell(s), 1 test(s) not run here", "1 deferred to the pre-tag tier"} {
		if !strings.Contains(out, want) {
			t.Errorf("the day report does not say %q:\n%s", want, out)
		}
	}
	if rc, out := run(true); rc != 1 || !strings.Contains(cellLine(t, out, "sub"), "FAIL") {
		t.Fatalf("-full: rc=%d, want 1 with the sub cell failing:\n%s", rc, out)
	}
	mutate(t, qc, "sub/s.go", "package sub\n", "package sub\n\n// touched\n")
	if rc, out := run(false); rc != 1 || strings.Contains(out, "[slow]") {
		t.Fatalf("day tier with the module touched: rc=%d, want 1 and no [slow] cell:\n%s", rc, out)
	}
}

// The real configuration's Slow patterns name tests that exist. A renamed test would silently stop being deferred
// (it would run on every push again), and a pattern matching nothing defers nothing.
func TestQuick_slowPatternsNameRealTests(t *testing.T) {
	root, _ := filepath.Abs("../..")
	qc, err := realQuickConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	defer func() {
		if checked == 0 {
			t.Error("no module has a Slow pattern; the test checked nothing")
		}
	}()
	for _, m := range qc.Modules {
		if m.Slow == "" {
			continue
		}
		if !m.Device {
			t.Errorf("module %s has a Slow pattern and is not a device module", m.Name)
		}
		body := strings.TrimSuffix(strings.TrimPrefix(m.Slow, "^("), ")$")
		if body == m.Slow {
			t.Fatalf("module %s: Slow %q is not an anchored list of names", m.Name, m.Slow)
		}
		checked++
		src := moduleTestSource(t, qc, m)
		for _, name := range strings.Split(body, "|") {
			if !strings.Contains(src, "func "+name+"(t *testing.T)") {
				t.Errorf("module %s: Slow names %s, which no test file in %s declares", m.Name, name, m.Dir)
			}
		}
	}
}

// moduleTestSource is the module's top-level test files, concatenated.
func moduleTestSource(t *testing.T, qc *quickConfig, m *quickModule) string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(qc.modDir(m), "*_test.go"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no test files in %s (%v)", qc.modDir(m), err)
	}
	var b strings.Builder
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(src)
	}
	return b.String()
}

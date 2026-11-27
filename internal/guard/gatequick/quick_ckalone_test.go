package gatequick

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testpath"
)

// A loader the module names in CheckpointAlone gets a [checkpoint <name>] cell of its own, a Checkpoint cell marked Mem, which
// the day loop defers like the package's [checkpoint] cell; the package's cells still run every test exactly once.
func TestQuick_checkpointAloneGetsItsOwnCell(t *testing.T) {
	qc, r := ckFixture(t)
	qc.Modules[0].CheckpointAlone = []string{"TestTable", "TestNotALoader"}
	cells := buildCells(qc, &quickGraph{roots: []*testRoot{r}}, &quickSelection{}, false, false)
	byName := map[string]*quickCell{}
	for _, c := range cells {
		byName[c.Name] = c
	}
	alone, ck, rest := byName["ck [checkpoint TestTable]"], byName["ck [checkpoint]"], byName["ck"]
	if alone == nil || ck == nil || rest == nil || len(cells) != 3 {
		names := make([]string, 0, len(cells))
		for _, c := range cells {
			names = append(names, c.Name)
		}
		t.Fatalf("want ck [checkpoint TestTable], ck [checkpoint] and ck; got %v", names)
	}
	if alone.Run != "^(TestTable)$" || !alone.Checkpoint || !alone.Mem {
		t.Errorf("the alone cell: Run %q Checkpoint %v Mem %v, want ^(TestTable)$, true, true", alone.Run, alone.Checkpoint, alone.Mem)
	}
	if regexp.MustCompile(ck.Run).MatchString("TestTable") {
		t.Errorf("ck [checkpoint] still runs TestTable: %s", ck.Run)
	}
	for _, c := range deferCells(qc, &quickSelection{}, cells) {
		if c.Checkpoint && c.Deferred != deferCheckpoint {
			t.Errorf("%s is not deferred to the pre-tag tier: %q", c.Name, c.Deferred)
		}
	}
	for _, n := range r.tests {
		in := 0
		for _, c := range cells {
			if (c.Run == "" || regexp.MustCompile(c.Run).MatchString(n)) && (c.Skip == "" || !regexp.MustCompile(c.Skip).MatchString(n)) {
				in++
			}
		}
		if in != 1 {
			t.Errorf("%s runs in %d cells, want exactly 1", n, in)
		}
	}
}

// Every name the real config sets alone is a test the root module declares, so a rename cannot quietly put it back.
func TestQuick_checkpointAloneNamesExist(t *testing.T) {
	root := testpath.Root(t)
	qc, err := realQuickConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, m := range qc.Modules {
		for _, name := range m.CheckpointAlone {
			n++
			files, _ := filepath.Glob(filepath.Join(root, m.Dir, "*", "*_test.go"))
			found := false
			for _, f := range files {
				b, err := os.ReadFile(f)
				if err == nil && strings.Contains(string(b), "func "+name+"(t *testing.T)") {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("%s: CheckpointAlone names %s, which no test file under %s declares", m.Name, name, m.Dir)
			}
		}
	}
	if n == 0 {
		t.Fatal("the real config sets no test alone")
	}
}

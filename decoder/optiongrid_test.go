package decoder

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// Tests for the option grid (optiongrid.go) and the generated page that shows it beside the
// cache-state grid (docs/option-state-grid.md).

const optionStateGridPath = "../docs/option-state-grid.md"

// TestOptionGrid_everyOptionClassified is the registration test for load options: a decoder.Options
// field lands only with a declaration on every path (or a load-only reason), the way
// TestOptions_everyFlagReachesOptions makes every CLI flag reach Options. A-C01 (-kv i8 into Metal's
// batched prefill) was an option no path's admission had been asked about.
func TestOptionGrid_everyOptionClassified(t *testing.T) {
	ty := reflect.TypeFor[Options]()
	seen := map[string]bool{}
	for field := range ty.Fields() {
		name := field.Name
		seen[name] = true
		row, inGrid := ogGrid[name]
		why, loadOnly := ogLoadOnly[name]
		switch {
		case inGrid && loadOnly:
			t.Errorf("Options.%s is both load-only and in ogGrid; pick one", name)
		case !inGrid && !loadOnly:
			t.Errorf("Options.%s is not classified: add it to ogGrid with a cell for every path, or to "+
				"ogLoadOnly with the reason no path reads it after Load (optiongrid.go)", name)
		case loadOnly && why == "":
			t.Errorf("Options.%s is load-only with no reason", name)
		case inGrid:
			for _, p := range ogPaths {
				if _, ok := row[p]; !ok {
					t.Errorf("Options.%s has no cell for %q", name, p)
				}
			}
			if len(row) != len(ogPaths) {
				t.Errorf("Options.%s declares %d cells for %d paths", name, len(row), len(ogPaths))
			}
		}
	}
	for name := range ogGrid {
		if !seen[name] {
			t.Errorf("ogGrid names Options.%s, which no longer exists", name)
		}
	}
	for name := range ogLoadOnly {
		if !seen[name] {
			t.Errorf("ogLoadOnly names Options.%s, which no longer exists", name)
		}
	}
}

// TestOptionGrid_cellsCarryEvidence: a tested cell names a test, a declined cell names the function
// that declines, and every name exists in this repository (any module); an n/a cell says why. A cell
// cannot claim coverage the tree does not have.
func TestOptionGrid_cellsCarryEvidence(t *testing.T) {
	tests, funcs := repoFuncs(t)
	for name, row := range ogGrid {
		for p, c := range row {
			where := fmt.Sprintf("Options.%s × %q", name, p)
			switch c.h {
			case ogTested:
				if c.test == "" {
					t.Errorf("%s is tested but names no test", where)
				}
			case ogDeclined:
				if c.decline == "" {
					t.Errorf("%s is declined but names no declining function", where)
				} else if !funcs[c.decline] {
					t.Errorf("%s names decline %q, which is not a function in this repository", where, c.decline)
				}
			case ogUntested:
				if c.test != "" || c.decline != "" {
					t.Errorf("%s is untested but carries evidence (%q, %q): declare it tested or declined", where, c.test, c.decline)
				}
			case ogNA:
				if c.why == "" {
					t.Errorf("%s is n/a with no reason", where)
				}
			default:
				t.Errorf("%s has handling %q", where, c.h)
			}
			if c.test != "" && !tests[c.test] {
				t.Errorf("%s names test %q, which is not a Test function in this repository", where, c.test)
			}
		}
	}
}

// TestOptionGrid_ratchet holds the untested cells at optionGridUntestedCeiling. Above it, a new
// option or path landed without a test; below it, a cell gained one and the ceiling must come down
// so it cannot be lost again unnoticed.
func TestOptionGrid_ratchet(t *testing.T) {
	n := 0
	for _, row := range ogGrid {
		for _, c := range row {
			if c.h == ogUntested {
				n++
			}
		}
	}
	switch {
	case n > optionGridUntestedCeiling:
		t.Errorf("%d option × path cells are admitted untested, above the ceiling of %d: a new cell must name "+
			"a test that drives it, or decline", n, optionGridUntestedCeiling)
	case n < optionGridUntestedCeiling:
		t.Errorf("%d option × path cells are admitted untested, below the ceiling of %d: lower "+
			"optionGridUntestedCeiling to %d so the ratchet holds what was gained", n, optionGridUntestedCeiling, n)
	}
}

// TestOptionStateGrid_fresh keeps docs/option-state-grid.md generated from the two grids.
// Regenerate: go test ./decoder -run OptionStateGrid -update
func TestOptionStateGrid_fresh(t *testing.T) {
	md := renderOptionStateGrid()
	if *updateMatrix {
		if err := os.WriteFile(optionStateGridPath, md, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	got, err := os.ReadFile(optionStateGridPath)
	if err != nil {
		t.Fatalf("read %s: %v (regenerate: go test ./decoder -run OptionStateGrid -update)", optionStateGridPath, err)
	}
	if !bytes.Equal(got, md) {
		t.Fatalf("%s is out of date — regenerate: go test ./decoder -run OptionStateGrid -update", optionStateGridPath)
	}
}

var testFuncRE = regexp.MustCompile(`(?m)^func (Test[A-Za-z0-9_]+)\(`)
var anyFuncRE = regexp.MustCompile(`(?m)^func (?:\([^)]*\) )?([A-Za-z_][A-Za-z0-9_]*)\(`)

// repoFuncs scans every .go file in the repository (all modules) for Test functions and for
// function and method names.
func repoFuncs(t *testing.T) (tests, funcs map[string]bool) {
	t.Helper()
	tests, funcs = map[string]bool{}, map[string]bool{}
	err := filepath.WalkDir("..", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "node_modules", "testdata", "docs", "site":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if strings.HasSuffix(path, "_test.go") {
			for _, m := range testFuncRE.FindAllSubmatch(b, -1) {
				tests[string(m[1])] = true
			}
		}
		for _, m := range anyFuncRE.FindAllSubmatch(b, -1) {
			funcs[string(m[1])] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(tests) < 100 || len(funcs) < 1000 {
		t.Fatalf("the repository scan found %d tests and %d functions — it is not scanning the tree", len(tests), len(funcs))
	}
	return tests, funcs
}

func renderOptionStateGrid() []byte {
	var b bytes.Buffer
	w := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }
	w("# Option × path and state × lifecycle grids\n\n")
	w("**Generated** from `decoder/optiongrid.go` and `decoder/cachestate.go` by `TestOptionStateGrid_fresh`; do not edit.\n")
	w("Regenerate: `go test ./decoder -run OptionStateGrid -update`. Design and history:\n")
	w("[`tasks/task-option-path-admission-2026-10.md`](tasks/task-option-path-admission-2026-10.md) §4.\n\n")

	untested := 0
	for _, row := range ogGrid {
		for _, c := range row {
			if c.h == ogUntested {
				untested++
			}
		}
	}
	w("## Load options × execution paths\n\n")
	w("**tested** names a test that drives the option through the path; **declined** names the function where the path refuses it;\n")
	w("**untested** runs today with no test that drives it (%d cells; `TestOptionGrid_ratchet` lets that number only fall);\n", untested)
	w("**n/a**: the option does not reach the path (reasons below the table).\n\n")
	w("| option |")
	for _, p := range ogPaths {
		w(" %s |", p)
	}
	w("\n|---|")
	for range ogPaths {
		w("---|")
	}
	w("\n")
	names := make([]string, 0, len(ogGrid))
	for n := range ogGrid {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		w("| `%s` |", n)
		for _, p := range ogPaths {
			c := ogGrid[n][p]
			switch c.h {
			case ogTested:
				w(" tested: `%s` |", c.test)
			case ogDeclined:
				if c.test != "" {
					w(" declined at `%s` (`%s`) |", c.decline, c.test)
				} else {
					w(" declined at `%s` |", c.decline)
				}
			case ogUntested:
				w(" untested |")
			case ogNA:
				w(" n/a |")
			}
		}
		w("\n")
	}
	w("\n**n/a reasons.**\n\n")
	for _, n := range names {
		seen := map[string]bool{}
		for _, p := range ogPaths {
			if c := ogGrid[n][p]; c.h == ogNA && !seen[c.why] {
				seen[c.why] = true
				w("- `%s`: %s\n", n, c.why)
			}
		}
	}
	w("\n**Load-only options** (read while loading, by no path afterwards):\n\n")
	lo := make([]string, 0, len(ogLoadOnly))
	for n := range ogLoadOnly {
		lo = append(lo, n)
	}
	sort.Strings(lo)
	for _, n := range lo {
		w("- `%s`: %s\n", n, ogLoadOnly[n])
	}

	w("\n## Kinds of cache state × lifecycle paths\n\n")
	w("Every `KVCache` field belongs to one of these kinds, or is declared geometry, a counter or scratch (`TestKVCache_everyFieldHasAState`).\n")
	w("`Snapshot` and a partial `TruncateTo` read this table.\n\n")
	w("| kind |")
	for _, p := range lifecyclePaths {
		w(" %s |", p)
	}
	w("\n|---|")
	for range lifecyclePaths {
		w("---|")
	}
	w("\n")
	for _, s := range cacheStates {
		w("| %s |", s)
		for _, p := range lifecyclePaths {
			w(" %s |", cacheStateGrid[s][p].h)
		}
		w("\n")
	}
	w("\n**Reasons.**\n\n")
	for _, s := range cacheStates {
		for _, p := range lifecyclePaths {
			if c := cacheStateGrid[s][p]; c.why != "" {
				w("- %s × %s: %s\n", s, p, c.why)
			}
		}
	}
	return b.Bytes()
}

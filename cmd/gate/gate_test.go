package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func needTools(t *testing.T) {
	t.Helper()
	for _, bin := range []string{"go", "git"} {
		if _, err := exec.LookPath(bin); err != nil {
			t.Skipf("%s not on PATH", bin)
		}
	}
}

func shortSHA(s string) string {
	if len(s) > 10 {
		return s[:10]
	}
	return s
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func writeTree(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for p, c := range files {
		full := filepath.Join(root, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func scratchModule(t *testing.T, src string) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module scratch\n\ngo 1.27\n")
	write("scratch_test.go", src)
	return dir
}

func runScratch(t *testing.T, cfg *gateConfig, dirs ...string) (int, string, *results, []cellResult) {
	t.Helper()
	cfg.Cells = nil
	for i, d := range dirs {
		cfg.Cells = append(cfg.Cells, cell{Name: fmt.Sprintf("cell%d", i), Pkgs: []string{"./..."}, Dir: d})
	}
	res := newResults()
	var cells []cellResult
	for _, c := range cfg.Cells {
		cells = append(cells, runCell(c, cfg, res, t.TempDir()))
	}
	var buf bytes.Buffer
	var rc int
	if cfg.Decision == "census" {
		rc = reportCensus(&buf, cfg, res, false)
	} else {
		rc = reportTally(&buf, cfg, res, cells, provenance{Commit: "test", Date: "-", Host: "-"})
	}
	return rc, buf.String(), res, cells
}

// Refusal is a third outcome, not a flavour of red: with no models dir, "green" and "red" are both
// lies. heavy_gate.sh exited 2; so does this.
func TestHeavy_refusesWithoutModelsDir(t *testing.T) {
	t.Setenv("GOINFER_GATE_MODELS", filepath.Join(t.TempDir(), "definitely-not-here"))
	var buf bytes.Buffer
	if rc := run([]string{"heavy"}, &buf); rc != 2 {
		t.Fatalf("rc=%d, want 2 (refused)\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "REFUSED") {
		t.Errorf("refusal not stated:\n%s", buf.String())
	}
}

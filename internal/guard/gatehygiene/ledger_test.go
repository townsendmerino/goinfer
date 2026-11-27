package gatehygiene

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testpath"
)

func TestLedger_realLedgerRoundTripsByteIdentical(t *testing.T) {
	root := testpath.Root(t)
	orig, err := os.ReadFile(ledgerPath(root))
	if err != nil {
		t.Skipf("no ledger: %v", err)
	}
	d, err := loadLedger(root)
	if err != nil {
		t.Fatal(err)
	}
	got, err := encodeLedger(d)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, orig) {
		t.Fatalf("re-encoding %s changed it (%d -> %d bytes)", ledgerRel, len(orig), len(got))
	}
	if got := string(asciiEscape([]byte("caf\u00e9 \u2014 \U0001F600"))); got != "caf\\u00e9 \\u2014 \\ud83d\\ude00" {
		t.Errorf("asciiEscape = %q", got)
	}
}

const ledgerKeyBody = "package pkg\n\nimport \"testing\"\n\nfunc TestAlpha(t *testing.T) {\n\t// déjà vu — a comment\n\tt.Log(\"x\")\n}\n\nfunc TestBeta(t *testing.T) {\n}\n"

func ledgerScratch(t *testing.T, ledger string) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "pkg"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "pkg", "a_test.go"), []byte(ledgerKeyBody), 0o644); err != nil {
		t.Fatal(err)
	}
	if ledger != "" {
		if err := os.MkdirAll(filepath.Join(root, "testdata"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(ledgerPath(root), []byte(ledger), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestLedger_sourceKeyMatchesThePythonImplementation(t *testing.T) {
	root := ledgerScratch(t, "")
	if k, ok := gateSourceKey(root, "TestAlpha"); !ok || k != "bb9dbe85121f4b67" {
		t.Errorf("TestAlpha key = %q (found=%v), want bb9dbe85121f4b67", k, ok)
	}
	if k, ok := gateSourceKey(root, "TestBeta"); !ok || k != "aa6fd3ca834d0f76" {
		t.Errorf("TestBeta key = %q (found=%v), want aa6fd3ca834d0f76", k, ok)
	}
	if _, ok := gateSourceKey(root, "TestAlph"); ok {
		t.Error("a prefix of a gate name matched")
	}
}

func TestLedger_classifyOutcomes(t *testing.T) {
	if got := ClassifyGate(ledgerScratch(t, ""), "TestAlpha"); got != "CONFIRMED" {
		t.Errorf("no ledger at all: %s, want CONFIRMED", got)
	}
	root := ledgerScratch(t, `{"entries": [
		{"gate": "TestAlpha", "value": "PASS", "promoted_by": "a", "date": "2026-09-25", "commit": "x", "source_sha256": "bb9dbe85121f4b67"},
		{"gate": "TestBeta", "value": "PASS", "promoted_by": "a", "date": "2026-09-25", "commit": "x", "source_sha256": "0000000000000000"}
	]}`)
	for gate, want := range map[string]string{
		"TestAlpha":       "CONFIRMED",
		"TestBeta":        "SOURCE-CHANGED",
		"TestGamma":       "UNKNOWN-GATE",
		"TestNeverExists": "UNKNOWN-GATE",
	} {
		if got := ClassifyGate(root, gate); got != want {
			t.Errorf("classify %s = %s, want %s", gate, got, want)
		}
	}
	root2 := ledgerScratch(t, `{"entries": []}`)
	if got := ClassifyGate(root2, "TestAlpha"); got != "FIRST-RUN" {
		t.Errorf("classify TestAlpha with an empty ledger = %s, want FIRST-RUN", got)
	}
}

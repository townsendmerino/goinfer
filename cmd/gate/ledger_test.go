package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// `gate ledger` replaced scripts/gate_ledger.py (2026-09-25). Before the script was deleted the two were run
// side by side: classify agreed on all 83 gates in testdata/gate_ledger.json (plus an unknown name), reconcile
// printed the same 69 lines, and promote + seed wrote the same file bar the seed note's new command name. These
// pin the parts of that equivalence that existing confirmations depend on, now that there is no script to
// compare against.

// The file format: loading the real ledger and writing it back must not change a byte — sorted keys, two-space
// indent, and non-ASCII escaped as \uXXXX the way Python's json.dumps(ensure_ascii) wrote it. A drift here
// would rewrite every entry on the next promote.
func TestLedger_realLedgerRoundTripsByteIdentical(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Skip(err)
	}
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
		t.Fatalf("re-encoding %s changed it (%d → %d bytes) — the format no longer matches the script's", ledgerRel, len(orig), len(got))
	}
	// And the escaping in isolation, including a character above U+FFFF (a surrogate pair).
	if got := string(asciiEscape([]byte("caf\u00e9 \u2014 \U0001F600"))); got != "caf\\u00e9 \\u2014 \\ud83d\\ude00" {
		t.Errorf("asciiEscape = %q", got)
	}
}

// The source key: SHA-256 of the gate function's body, first 16 hex digits. The value below was computed by
// scripts/gate_ledger.py for exactly this body (with non-ASCII in it) before the script was retired; a different
// key would silently turn every existing confirmation into SOURCE-CHANGED.
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
		t.Errorf("TestAlpha key = %q (found=%v), the script computed bb9dbe85121f4b67", k, ok)
	}
	if k, ok := gateSourceKey(root, "TestBeta"); !ok || k != "aa6fd3ca834d0f76" {
		t.Errorf("TestBeta key = %q (found=%v), the script computed aa6fd3ca834d0f76", k, ok)
	}
	if _, ok := gateSourceKey(root, "TestAlph"); ok {
		t.Error("a prefix of a gate name matched — the scan must match `func <name>(` exactly")
	}
}

func TestLedger_classifyOutcomes(t *testing.T) {
	if got := classifyGate(ledgerScratch(t, ""), "TestAlpha"); got != "CONFIRMED" {
		t.Errorf("no ledger at all: %s, want CONFIRMED (inert: a failure still blocks)", got)
	}
	root := ledgerScratch(t, `{"entries": [
		{"gate": "TestAlpha", "value": "PASS", "promoted_by": "a", "date": "2026-09-25", "commit": "x", "source_sha256": "bb9dbe85121f4b67"},
		{"gate": "TestBeta", "value": "PASS", "promoted_by": "a", "date": "2026-09-25", "commit": "x", "source_sha256": "0000000000000000"}
	]}`)
	for gate, want := range map[string]string{
		"TestAlpha":       "CONFIRMED",      // entry, and the body is the one confirmed
		"TestBeta":        "SOURCE-CHANGED", // entry, but the body changed since
		"TestGamma":       "UNKNOWN-GATE",   // no entry and no function anywhere
		"TestNeverExists": "UNKNOWN-GATE",
	} {
		if got := classifyGate(root, gate); got != want {
			t.Errorf("classify %s = %s, want %s", gate, got, want)
		}
	}
	// FIRST-RUN: a gate that exists in source but has no entry.
	root2 := ledgerScratch(t, `{"entries": []}`)
	if got := classifyGate(root2, "TestAlpha"); got != "FIRST-RUN" {
		t.Errorf("classify TestAlpha with an empty ledger = %s, want FIRST-RUN", got)
	}
}

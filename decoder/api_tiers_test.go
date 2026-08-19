package decoder

// The drift guard between docs/api-tiers.md (the PROSE promise) and
// testdata/apidiff/hard_tier.txt (the MACHINE-CHECKED one, read by scripts/apidiff_check.sh).
//
// WHY BOTH FILES EXIST. The document is what a user reads to decide whether to build on a surface;
// the list is what CI can enforce. Neither can replace the other — prose is not checkable and a
// bare identifier list explains nothing — so the risk is that they disagree: a promise nobody
// gates, or a gate for a promise nobody wrote down. This test makes the pair fail loudly instead.
//
// Scope, stated honestly: it verifies every Hard-tier entry is NAMED IN THE HARD SECTION of the
// document. It does not verify the reverse direction (the doc's prose lists Options fields and
// route paths that are not apidiff identifiers), and it does not check placement beyond the
// section boundary.

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAPITiers_hardListMatchesDoc(t *testing.T) {
	const listPath = "../testdata/apidiff/hard_tier.txt"
	const docPath = "../docs/api-tiers.md"

	raw, err := os.ReadFile(filepath.Clean(listPath))
	if err != nil {
		t.Fatalf("read %s: %v", listPath, err)
	}
	doc, err := os.ReadFile(filepath.Clean(docPath))
	if err != nil {
		t.Fatalf("read %s: %v", docPath, err)
	}
	// The Hard section only: a name that appears solely under Experimental is a disagreement, not a
	// match, and matching against the whole file would hide exactly that case.
	text := string(doc)
	start := strings.Index(text, "## Hard —")
	end := strings.Index(text, "## Experimental —")
	if start < 0 || end < 0 || end < start {
		t.Fatalf("%s: could not locate the Hard/Experimental section headings — the drift guard "+
			"cannot check placement if the document's shape changed", docPath)
	}
	hard := text[start:end]

	wantPkgs := map[string]bool{"decoder": true, "tokenizer": true, "chat": true, "constrain": true}
	n := 0
	sc := bufio.NewScanner(strings.NewReader(string(raw)))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		pkg, name, ok := strings.Cut(line, " ")
		if !ok {
			t.Errorf("%s: malformed line %q (want `<pkg> <apidiff-name>`)", listPath, line)
			continue
		}
		if !wantPkgs[pkg] {
			t.Errorf("%s: package %q is not one of the Hard-tier packages", listPath, pkg)
		}
		n++
		// The bare identifier: "(*Model).Close" → "Close", "Options.Quant" → "Options.Quant"
		// (the doc writes the field qualified), plain names unchanged.
		bare := name
		if i := strings.Index(bare, ")."); i >= 0 {
			bare = bare[i+2:]
		}
		if !strings.Contains(hard, bare) {
			t.Errorf("%s lists %s %q but docs/api-tiers.md's HARD section never names %q — the "+
				"gate would enforce a promise the document does not make", listPath, pkg, name, bare)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan %s: %v", listPath, err)
	}
	// A list that silently emptied would pass every check above while gating nothing.
	if n < 50 {
		t.Errorf("only %d Hard-tier entries parsed — the list looks truncated; scripts/apidiff_check.sh "+
			"would then report PASS while checking almost nothing", n)
	}
	t.Logf("%d Hard-tier entries, all named in the document's Hard section", n)
}

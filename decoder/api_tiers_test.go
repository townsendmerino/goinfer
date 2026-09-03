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
	"regexp"
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
		// G-10: a BARE substring match cannot fail. "Load" is inside "LoadGGUFBytes", so
		// deleting `decoder.Load` from the Hard section left this green — and the same held for
		// Model, Session, Config, Segment, Template, Tool, Turn, Meta and Has, every one of them
		// a substring of some longer name the document also lists. A check that cannot go red
		// for the ten most fundamental entries is not gating the document.
		//
		// Match the name as the document actually writes it: inside backticks, optionally
		// package-qualified, and terminated — so `decoder.Load` matches and LoadGGUFBytes does
		// not lend it its letters.
		if !namedInDoc(hard, pkg, bare) {
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

// namedInDoc reports whether doc names ident (optionally qualified by pkg) as a WHOLE name
// rather than as a substring of a longer one. See G-10 at the call site.
//
// The document writes entries inside backticks — `decoder.Load`, `Options.Quant`, **`Model`** —
// so the boundary is the backtick or a character that cannot continue a Go identifier or
// selector. Both the qualified and unqualified spellings are accepted because the doc uses both.
func namedInDoc(doc, pkg, ident string) bool {
	// SCOPED TO THE PACKAGE'S OWN SUBSECTION. Searching the whole Hard section let
	// `tokenizer.Load` satisfy an entry for `decoder.Load` — measured: with the doc's
	// `decoder.Load` deleted, the check still passed on the tokenizer line. The document is
	// organised as "### `<pkg>`" subsections, so that is the unit to search.
	//
	// Within a subsection the TRAILING boundary does the work: it separates `Load` from
	// `LoadGGUFBytes`. The leading one only rejects a name buried inside a longer identifier
	// (Reload), so it excludes word characters but NOT a dot — the document qualifies by
	// whatever reads best, `decoder.Load` in one place and `Model.LoadSession` in another, and
	// both are the name being named.
	sec := pkgSection(doc, pkg)
	if sec == "" {
		return false
	}
	// WHOLE BACKTICKED ENTRIES, not a regex over prose. A word-boundary match still let a TYPE
	// be satisfied by a METHOD that happens to be qualified by it — `Model.LoadSession` answered
	// for `decoder.Model`, so deleting the type from the document changed nothing. Measured: of
	// the ten names G-10 lists, a boundary match caught only Load.
	//
	// An entry names `ident` when some backticked span IS it (bare or package-qualified), or
	// ends in "."+ident — which covers `Model.LoadSession` naming LoadSession and
	// `Options.Quant` naming Quant, while `Model.LoadSession` does NOT name Model.
	for _, span := range regexp.MustCompile("`([^`]+)`").FindAllStringSubmatch(sec, -1) {
		e := strings.TrimSpace(span[1])
		if e == ident || e == pkg+"."+ident || strings.HasSuffix(e, "."+ident) {
			return true
		}
	}
	return false
}

// pkgSection returns the "### `<pkg>`" subsection of doc, or "" when the document has none for
// that package. The heading may carry a trailing description ("### `decoder` — load and
// generate"), so it is matched by its backticked name rather than by the whole line.
func pkgSection(doc, pkg string) string {
	head := regexp.MustCompile("(?m)^### `" + regexp.QuoteMeta(pkg) + "`")
	loc := head.FindStringIndex(doc)
	if loc == nil {
		return ""
	}
	rest := doc[loc[1]:]
	if next := regexp.MustCompile("(?m)^#{2,3} ").FindStringIndex(rest); next != nil {
		return rest[:next[0]]
	}
	return rest
}

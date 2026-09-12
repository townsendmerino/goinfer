package decoder

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// N-42: docs/env-vars.md is declared a Hard-tier contract by docs/api-tiers.md, and it had
// drifted in BOTH directions — 40 variables read by production code were absent from it
// (including the escape hatches for all four default-ON changes, each of which alters greedy
// output), and one it documented was read by nothing at all.
//
// This is the TestAssetRegistry_noDirectReads shape the audit's fix text asks for: the doc and
// the code are compared to each other, so neither can move without the other. It deliberately
// does NOT check what the doc SAYS about a variable — only that every variable exists in both
// places. A wrong description is a different problem; a missing entry is this one.
func TestEnvVars_docAndCodeAgree(t *testing.T) {
	root := ".."
	docPath := filepath.Join(root, "docs", "env-vars.md")
	doc, err := os.ReadFile(docPath)
	if err != nil {
		t.Skipf("no env-vars.md: %v", err)
	}
	documented := map[string]bool{}
	for _, v := range regexp.MustCompile(`GOINFER_[A-Z0-9_]+`).FindAllString(string(doc), -1) {
		documented[v] = true
	}
	// M-56 (audit-2026-09-10.md): everything ABOVE the first "not contract" heading is an
	// operator-facing promise (docs/api-tiers.md's Hard tier); a var documented only there but
	// referenced by nothing except _test.go files is exactly GOINFER_GEMMA4_RESIDENT's shape —
	// a bring-up gate that went silently unread while its doc row kept promising an effect. The
	// phantom check below already tolerates a test-only reference for the diagnostics section
	// (test-only IS what "not contract" means there); operatorDoc is scanned separately so that
	// same tolerance can't hide a dead operator knob again.
	operatorDoc := string(doc)
	if i := strings.Index(operatorDoc, "## Diagnostics and experiment knobs"); i >= 0 {
		operatorDoc = operatorDoc[:i]
	}
	operatorDocumented := map[string]bool{}
	for _, v := range regexp.MustCompile(`GOINFER_[A-Z0-9_]+`).FindAllString(operatorDoc, -1) {
		operatorDocumented[v] = true
	}

	// Every way the tree reads a knob, not just os.Getenv (audit-2026-09-10 G-13(e)): os.LookupEnv,
	// and cmd/gate's env(key, default) helper, whose four knobs were undocumented and invisible here.
	getenv := regexp.MustCompile(`(?:os\.Getenv|os\.LookupEnv|\benv)\("(GOINFER_[A-Z0-9_]+)"`)
	anyRef := regexp.MustCompile(`"(GOINFER_[A-Z0-9_]+)"`)
	readInProd := map[string]string{} // var -> first file that reads it
	referencedAnywhere := map[string]bool{}
	referencedInNonTest := map[string]bool{} // any mention (not just a getenv call) outside _test.go

	err = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".go") {
			return nil
		}
		b, rerr := os.ReadFile(p)
		if rerr != nil {
			return nil
		}
		for _, m := range anyRef.FindAllStringSubmatch(string(b), -1) {
			referencedAnywhere[m[1]] = true
		}
		if strings.HasSuffix(p, "_test.go") {
			return nil
		}
		for _, m := range anyRef.FindAllStringSubmatch(string(b), -1) {
			referencedInNonTest[m[1]] = true
		}
		for _, m := range getenv.FindAllStringSubmatch(string(b), -1) {
			if _, seen := readInProd[m[1]]; !seen {
				readInProd[m[1]] = p
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if len(readInProd) == 0 {
		t.Fatal("found no GOINFER_ env reads at all — the scan is broken, not the docs")
	}

	var missing []string
	for v, file := range readInProd {
		if !documented[v] {
			missing = append(missing, v+"  ("+file+")")
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("%d variable(s) read by NON-TEST code and absent from docs/env-vars.md, which "+
			"api-tiers.md declares a Hard-tier contract (N-42):\n  %s\n\nAdd each to the curated "+
			"table if an operator would set it, or to the diagnostics list if not — the list is "+
			"explicit precisely so this test has somewhere to put things that are not contract.",
			len(missing), strings.Join(missing, "\n  "))
	}

	var phantom []string
	for v := range documented {
		if !referencedAnywhere[v] {
			phantom = append(phantom, v)
		}
	}
	sort.Strings(phantom)
	if len(phantom) > 0 {
		t.Errorf("%d documented variable(s) appear in NO .go file, test or otherwise — the doc "+
			"promises a knob that does not exist (N-42):\n  %s", len(phantom), strings.Join(phantom, "\n  "))
	}

	var operatorPhantom []string
	for v := range operatorDocumented {
		if !referencedInNonTest[v] {
			operatorPhantom = append(operatorPhantom, v)
		}
	}
	sort.Strings(operatorPhantom)
	if len(operatorPhantom) > 0 {
		t.Errorf("%d operator-facing variable(s) are referenced by _test.go files ONLY, not by "+
			"any non-test code (M-56, audit-2026-09-10.md) — an operator setting this knob has no "+
			"effect. Move to the Diagnostics section (if it's a deliberate test-only knob) or "+
			"delete the row (if it's simply dead):\n  %s",
			len(operatorPhantom), strings.Join(operatorPhantom, "\n  "))
	}
}

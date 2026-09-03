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

	getenv := regexp.MustCompile(`os\.Getenv\("(GOINFER_[A-Z0-9_]+)"\)`)
	anyRef := regexp.MustCompile(`"(GOINFER_[A-Z0-9_]+)"`)
	readInProd := map[string]string{} // var -> first file that reads it
	referencedAnywhere := map[string]bool{}

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
}

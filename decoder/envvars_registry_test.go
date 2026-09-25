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
	// One ROW per variable. The same var documented twice drifts: GOINFER_CUDA_FLASH_DECODE had an
	// "(unset = off)" row and an "ON (S=16)" row at once after its default flipped, and
	// GOINFER_CUDA_GRAPHS_SYNC sat in both the operator and the diagnostics sections.
	rowSeen := map[string]int{}
	for i, line := range strings.Split(string(doc), "\n") {
		if m := regexp.MustCompile("^\\|\\s*`(GOINFER_[A-Z0-9_]+)`").FindStringSubmatch(line); m != nil {
			if prev, dup := rowSeen[m[1]]; dup {
				t.Errorf("docs/env-vars.md documents %s in two rows (lines %d and %d); keep one", m[1], prev, i+1)
			}
			rowSeen[m[1]] = i + 1
		}
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
	// envBoolDefault (decoder/cpu_gateup_fused.go) is the one indirect reader: a helper taking the name.
	getenv := regexp.MustCompile(`(?:os\.Getenv|os\.LookupEnv|\benv|\benvBoolDefault)\("(GOINFER_[A-Z0-9_]+)"`)
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
		// A file that only builds with goinfer_testhooks is test-seam code: no shipped binary reads its env
		// vars, so they are not production reads for the ratchet (phase 6, docs/tasks/task-env-config-2026-09.md).
		// Only the positive, &&-joined form exists in the tree; `!goinfer_testhooks` files are production.
		if testhooksOnly(b) {
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
	checkEnvReadRatchet(t, root, readInProd)

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

// checkEnvReadRatchet is phase 1 of docs/tasks/task-env-config-2026-09.md: testdata/env_reads.txt is
// the committed list of every GOINFER_* variable production code reads through the environment, and
// it may only SHRINK. A production read of a variable not on the list fails (rule 1: a new operator
// choice is an Options field with a per-model accessor; a new diagnostic is a test hook); a listed
// variable nothing reads any more also fails, so the list tracks each migration instead of going
// stale. `go test ./decoder -run TestEnvVars -update` rewrites the list from the code — for removals;
// an addition made that way is exactly what the list exists to make deliberate, and shows in review.
func checkEnvReadRatchet(t *testing.T, root string, readInProd map[string]string) {
	t.Helper()
	path := filepath.Join(root, "testdata", "env_reads.txt")
	// Each line is NAME<TAB>annotation. The annotation is the task's done criterion made checkable
	// (docs/tasks/task-env-config-2026-09.md, amended 2026-09-24): every read that remains is either
	// "startup: <who> — <why>" (configuration read once at startup) or "diagnostic: <owning doc> — <why>".
	listed := map[string]string{}
	if b, err := os.ReadFile(path); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "#") {
				name, ann, _ := strings.Cut(line, "\t")
				listed[strings.TrimSpace(name)] = strings.TrimSpace(ann)
			}
		}
	} else if !*updateMatrix {
		t.Fatalf("read %s: %v (generate it with -update)", path, err)
	}
	if *updateMatrix {
		vars := make([]string, 0, len(readInProd))
		for v := range readInProd {
			vars = append(vars, v)
		}
		sort.Strings(vars)
		out := "# Generated by `go test ./decoder -run TestEnvVars -update` (annotations are kept). Every GOINFER_* variable\n" +
			"# production code reads from the process environment, as NAME<TAB>startup: ... | diagnostic: <owning doc> — <why>.\n" +
			"# This list may only shrink: see docs/tasks/task-env-config-2026-09.md.\n"
		for _, v := range vars {
			out += v + "\t" + listed[v] + "\n"
		}
		if err := os.WriteFile(path, []byte(out), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	var unannotated []string
	for v, ann := range listed {
		if !strings.HasPrefix(ann, "startup: ") && !strings.HasPrefix(ann, "diagnostic: ") {
			unannotated = append(unannotated, v)
		}
	}
	sort.Strings(unannotated)
	if len(unannotated) > 0 {
		t.Errorf("%d variable(s) in testdata/env_reads.txt lack an owner and reason — each line must read "+
			"NAME<TAB>\"startup: <who> — <why>\" or NAME<TAB>\"diagnostic: <owning doc> — <why>\" "+
			"(docs/tasks/task-env-config-2026-09.md, the done criterion):\n  %s", len(unannotated), strings.Join(unannotated, "\n  "))
	}
	var added, stale []string
	for v, file := range readInProd {
		if _, ok := listed[v]; !ok {
			added = append(added, v+"  ("+file+")")
		}
	}
	for v := range listed {
		if _, ok := readInProd[v]; !ok {
			stale = append(stale, v)
		}
	}
	sort.Strings(added)
	sort.Strings(stale)
	if len(added) > 0 {
		t.Errorf("%d NEW production read(s) of the process environment — configuration goes through Options now "+
			"(docs/tasks/task-env-config-2026-09.md, rule 1): an operator choice is a decoder.Options field with a "+
			"per-model accessor, a diagnostic is a test hook. If this really must be an env var, add it to "+
			"testdata/env_reads.txt deliberately:\n  %s", len(added), strings.Join(added, "\n  "))
	}
	if len(stale) > 0 {
		t.Errorf("%d variable(s) in testdata/env_reads.txt are no longer read by production code — a migration "+
			"landed; shrink the list (go test ./decoder -run TestEnvVars -update):\n  %s", len(stale), strings.Join(stale, "\n  "))
	}
}

// testhooksOnly reports whether a Go file's build constraint requires the goinfer_testhooks tag.
func testhooksOnly(src []byte) bool {
	for _, line := range strings.Split(string(src), "\n") {
		if c, ok := strings.CutPrefix(line, "//go:build "); ok {
			for _, term := range strings.Split(c, "&&") {
				if strings.TrimSpace(term) == "goinfer_testhooks" {
					return !strings.Contains(c, "||")
				}
			}
			return false
		}
		if line != "" && !strings.HasPrefix(line, "//") {
			return false
		}
	}
	return false
}

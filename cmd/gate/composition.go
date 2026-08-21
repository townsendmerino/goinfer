package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// The release gate's coverage COMPOSITION along its axes — family × quant × loader.
//
// THE RULE THIS IMPLEMENTS: a gate whose value depends on an axis must print its composition along
// that axis. The sweep reported pass/fail per gate with nothing saying what the set SPANNED, and
// that is the shape that let the forward goldens report "19 passed" through nine deps_hash refreshes
// while every one of the 19 was f32 — an accurate count that could not distinguish "the axis is
// covered" from "the axis collapsed to one value".
//
// DERIVED, NOT DECLARED. Quant comes from grepping each gate's own test source for `Quant: "..."`,
// and loader from the test name. Hand-maintained axis metadata beside the gate list would be a
// second copy to drift, which is the defect this repo keeps finding. A gate whose test source
// cannot be located is reported as UNKNOWN rather than defaulted to f32 — defaulting would inflate
// the f32 count with gates nobody checked, which is the opposite of the point.
//
// MIGRATED FROM scripts/sweep_composition.py (E8). It had to move in the same commit as the sweep
// itself, not "later as a config": it PARSED the `GATES=(…)` array out of parity_sweep.sh with a
// regexp, so deleting that shell script would have broken it outright. The gate list is now the Go
// slice both sides read, which removes the parse rather than reimplementing it.

var (
	quantRe  = regexp.MustCompile(`Quant:\s*"([a-z0-9]+)"`)
	funcRe   = regexp.MustCompile(`func (Test[A-Za-z0-9_]+)\(`)
	goldenRe = regexp.MustCompile(`GOLDEN_RE='([^']+)'`)
	buildRe  = regexp.MustCompile(`(?m)^//go:build`)
)

// compSearchDirs mirrors the directories the Python looked in for a gate's test source.
var compSearchDirs = []string{"decoder", "cuda", "metal", "tokenizer", "internal"}

type compRow struct{ Family, Test, Quant, Loader string }

// loaderOf derives the loader axis from the test name, the same way the Python did.
func loaderOf(test string) string {
	if strings.Contains(test, "GGUF") || strings.Contains(strings.ToLower(test), "gguf") {
		return "gguf"
	}
	return "safetensors"
}

// testSource finds the file declaring `func <name>(`. Walks in sorted order so the answer does not
// depend on directory iteration order — the Python took grep's first hit, which for a test declared
// once is the same file, but only accidentally deterministic.
func testSource(name string) string {
	want := []byte("func " + name + "(")
	var hit string
	for _, d := range compSearchDirs {
		_ = filepath.WalkDir(d, func(path string, e os.DirEntry, err error) error {
			if err != nil || hit != "" || e.IsDir() || !strings.HasSuffix(path, ".go") {
				return nil
			}
			b, err := os.ReadFile(path)
			if err == nil && strings.Contains(string(b), string(want)) {
				hit = path
			}
			return nil
		})
		if hit != "" {
			return hit
		}
	}
	return ""
}

func quantsOf(src string) []string {
	b, err := os.ReadFile(src)
	if err != nil {
		return nil
	}
	set := map[string]bool{}
	for _, m := range quantRe.FindAllStringSubmatch(string(b), -1) {
		set[m[1]] = true
	}
	return sortedSet(set)
}

// goldenAxes reports the quantizations AND loaders the forward goldens actually drive.
//
// The loader axis was the one nobody had checked: the cross-gate check compared quant only, so
// "both gates span the same axes" was an answer about ONE axis presented as an answer about the
// gate. The selector comes from refresh_parity_hashes.sh's own GOLDEN_RE rather than a second list
// that could drift from it.
func goldenAxes() (quants, loaders map[string]bool, ok bool) {
	b, err := os.ReadFile(filepath.Join("scripts", "refresh_parity_hashes.sh"))
	if err != nil {
		return nil, nil, false
	}
	m := goldenRe.FindStringSubmatch(string(b))
	if m == nil {
		return nil, nil, false
	}
	sel, err := regexp.Compile(m[1])
	if err != nil {
		return nil, nil, false
	}
	files, _ := filepath.Glob(filepath.Join("decoder", "*_test.go"))
	sort.Strings(files)
	quants, loaders = map[string]bool{}, map[string]bool{}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		txt := string(raw)
		var names []string
		for _, mm := range funcRe.FindAllStringSubmatch(txt, -1) {
			if sel.MatchString(mm[1]) {
				names = append(names, mm[1])
			}
		}
		if len(names) == 0 {
			continue
		}
		// Behind a build tag the refresh does not pass — invisible to it, so it must not be counted
		// as coverage the refresh has.
		if buildRe.MatchString(txt) && strings.Contains(strings.SplitN(txt, "\n", 2)[0], "realckpt") {
			continue
		}
		qs := quantRe.FindAllStringSubmatch(txt, -1)
		if len(qs) == 0 {
			quants["f32"] = true
		}
		for _, q := range qs {
			quants[q[1]] = true
		}
		for _, n := range names {
			loaders[loaderOf(n)] = true
		}
	}
	return quants, loaders, true
}

// atoms splits composite labels ("int4/int8") before comparing.
//
// Without this the cross-gate check reports a difference that is purely NOTATIONAL — a gate whose
// test file drives two quantizations gets a composite label, and comparing it against the atomic
// labels the other side produces is a permanent false positive in the check built to make real
// differences visible.
func atoms(xs map[string]bool) map[string]bool {
	out := map[string]bool{}
	for x := range xs {
		for _, a := range strings.Split(x, "/") {
			out[a] = true
		}
	}
	return out
}

func setDiff(a, b map[string]bool) []string {
	var out []string
	for k := range a {
		if !b[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func counted(xs []string) string {
	c := map[string]int{}
	for _, x := range xs {
		c[x]++
	}
	keys := make([]string, 0, len(c))
	for k := range c {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, c[k]))
	}
	return strings.Join(parts, "  ")
}

// composition prints the sweep's coverage composition. Returns the process exit code.
func composition(w io.Writer, verbose bool) int {
	if len(parityGates) == 0 {
		fmt.Fprintf(os.Stderr, "composition: the gate list is EMPTY — refusing to report a "+
			"composition derived from nothing.\n")
		return 1
	}
	rows := make([]compRow, 0, len(parityGates))
	unknown := 0
	for _, g := range parityGates {
		quant := "UNKNOWN"
		if src := testSource(g.Test); src != "" {
			if qs := quantsOf(src); len(qs) > 0 {
				quant = strings.Join(qs, "/")
			} else {
				quant = "f32"
			}
		} else {
			unknown++
		}
		rows = append(rows, compRow{g.Family, g.Test, quant, loaderOf(g.Test)})
	}

	if verbose {
		fmt.Fprintf(w, "  %-22s%-14s%-14s%s\n", "family", "quant", "loader", "test")
		for _, r := range rows {
			fmt.Fprintf(w, "  %-22s%-14s%-14s%s\n", r.Family, r.Quant, r.Loader, r.Test)
		}
		fmt.Fprintln(w)
	}

	var qs, ls []string
	fams := map[string]bool{}
	qset, lset := map[string]bool{}, map[string]bool{}
	for _, r := range rows {
		qs = append(qs, r.Quant)
		ls = append(ls, r.Loader)
		fams[r.Family] = true
		qset[r.Quant] = true
		lset[r.Loader] = true
	}
	fmt.Fprintf(w, "  COMPOSITION of the release gate — %d gates over %d family labels\n", len(rows), len(fams))
	fmt.Fprintf(w, "    quant :  %s\n", counted(qs))
	fmt.Fprintf(w, "    loader:  %s\n", counted(ls))
	if unknown > 0 {
		fmt.Fprintf(w, "    NOTE: %d gate(s) UNKNOWN — test source not located, NOT counted as f32\n", unknown)
	}

	// CROSS-GATE. The sweep and the goldens refresh protect overlapping properties over the same
	// axis, and until both printed their composition the difference between them was invisible: the
	// sweep covered int4 all along while the refresh was f32-only, and nobody saw the gap because
	// neither said what it spanned.
	if goldQ, goldL, ok := goldenAxes(); ok {
		sweepQ, gq := atoms(qset), atoms(goldQ)
		sweepL, gl := atoms(lset), atoms(goldL)
		fmt.Fprintln(w)
		fmt.Fprintf(w, "  CROSS-GATE quant coverage (release gate vs the freeze-exception goldens):\n")
		fmt.Fprintf(w, "    gate parity       : %s\n", orNone(sortedSet(sweepQ)))
		fmt.Fprintf(w, "    forward goldens   : %s\n", orNone(sortedSet(gq)))
		onlySweep, onlyGold := setDiff(sweepQ, gq), setDiff(gq, sweepQ)
		if len(onlySweep) > 0 {
			fmt.Fprintf(w, "    ONLY in the sweep : %s\n", strings.Join(onlySweep, " "))
			fmt.Fprintf(w, "      -> a core edit can pass the goldens refresh and still be unproven on these,\n")
			fmt.Fprintf(w, "         because the refresh is the ONLY numeric proof a frozen-core edit gets.\n")
		}
		if len(onlyGold) > 0 {
			fmt.Fprintf(w, "    ONLY in the goldens: %s\n", strings.Join(onlyGold, " "))
		}
		if len(onlySweep) == 0 && len(onlyGold) == 0 {
			fmt.Fprintf(w, "    -> the two gates span the same quantizations.\n")
		}
		fmt.Fprintln(w)
		fmt.Fprintf(w, "  CROSS-GATE loader coverage:\n")
		fmt.Fprintf(w, "    gate parity       : %s\n", strings.Join(sortedSet(sweepL), " "))
		fmt.Fprintf(w, "    forward goldens   : %s\n", strings.Join(sortedSet(gl), " "))
		lsOnly, lgOnly := setDiff(sweepL, gl), setDiff(gl, sweepL)
		if len(lsOnly) > 0 {
			fmt.Fprintf(w, "    ONLY in the sweep : %s\n", strings.Join(lsOnly, " "))
		}
		if len(lgOnly) > 0 {
			fmt.Fprintf(w, "    ONLY in the goldens: %s\n", strings.Join(lgOnly, " "))
		}
		if len(lsOnly) == 0 && len(lgOnly) == 0 {
			fmt.Fprintf(w, "    -> the two gates span the same loaders.\n")
		}
	}

	// The collapse warning is the whole reason the composition is printed at all.
	if len(qset) == 1 {
		fmt.Fprintf(w, "    WARNING: the quant axis has collapsed to a single value (%s). "+
			"This gate no longer varies over the axis it is supposed to protect.\n", sortedSet(qset)[0])
	}
	return 0
}

func orNone(xs []string) string {
	if len(xs) == 0 {
		return "(none)"
	}
	return strings.Join(xs, " ")
}

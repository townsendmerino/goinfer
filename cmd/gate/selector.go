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

// Tests that EXIST versus tests any selector actually RUNS.
//
// Three coverage gaps this campaign found were PLUMBING, not authorship — every one a test that
// existed, passed when invoked, and was simply never selected:
//
//	the three int8int8 goldens      — skipped on GOINFER_HEAVY_TESTS being unset
//	gpt_oss's int8 golden           — behind //go:build realckpt, AND a missing checkpoint
//	eleven GGUF quant-format gates  — outside the goldens selector's regexp
//
// Each was found by someone asking a different question and noticing in passing. This asks it
// directly: enumerate what exists, enumerate what the selectors reach, print the difference.
//
// DESIGN, from what the other censuses learned:
//
//   - DERIVE BOTH SIDES. The selectors come from refresh_parity_hashes.sh's GOLDEN_RE and from the
//     sweep's own gate list, never restated here. (The Python read the gate list by regexping
//     `GATES=(…)` out of parity_sweep.sh — which is why this had to migrate in the same commit as
//     the sweep rather than "later as a config": deleting that file would have broken it outright.
//     It now reads the same Go slice the sweep checks, so the second copy is gone rather than
//     re-implemented.)
//   - SEPARATE THE REASONS. never-selected, build-tag-excluded, env-gated and asset-blocked have
//     different remedies and different costs; collapsing them into one "uncovered" number is what
//     made the int8int8 rows look as expensive as authoring new fixtures when they cost one env var.
//   - ERR TOWARD FLAGGING. The env detection matches any GOINFER_* read in the file, so it flags
//     TestInt4_forwardParity for GOINFER_INT4_GOLDEN_UPDATE — which gates REGENERATION, not the
//     test. That false positive is deliberate: a census that UNDER-reports is the failure mode that
//     produced all three gaps above, and a reader dismisses a flagged line in seconds where a
//     missing one costs weeks.
//   - PRINT THE DIFFERENCE, NOT A VERDICT. A test can be selected and still vacuous, so a green
//     here means "nothing became unreachable since a person last looked", not "coverage is
//     adequate".

var (
	testFnRe    = regexp.MustCompile(`(?m)^func (Test[A-Za-z0-9_]+)\(`)
	buildTagRe  = regexp.MustCompile(`(?m)^//go:build (.+)$`)
	envRe       = regexp.MustCompile(`Getenv\("(GOINFER_HEAVY_TESTS|GOINFER_[A-Z0-9_]*)"\)`)
	assetSkipRe = regexp.MustCompile(`(?i)t\.Skipf?\([^)]*\b(no |not found|missing|regenerate|checkpoint|gguf)`)
)

// selectorScanDirs is HAND-MAINTAINED, and that is why the report states it: a package outside this
// list is invisible to the census, and its absence looks identical to full coverage. cuda/ and
// metal/ are device-gated suites with their own runners.
var selectorScanDirs = []string{"decoder"}

type selRow struct {
	Test, File, Tag string
	Envs            []string
	Asset, Selected bool
}

// bucket is an insertion-ordered multimap, so ties in the descending-count sort keep the order the
// reasons were first seen — matching Python's stable sort over an insertion-ordered dict.
type bucket struct {
	keys []string
	m    map[string][]string
}

func newBucket() *bucket { return &bucket{m: map[string][]string{}} }

func (b *bucket) add(k, v string) {
	if _, ok := b.m[k]; !ok {
		b.keys = append(b.keys, k)
	}
	b.m[k] = append(b.m[k], v)
}

func (b *bucket) byCountDesc() []string {
	ks := append([]string{}, b.keys...)
	sort.SliceStable(ks, func(i, j int) bool { return len(b.m[ks[i]]) > len(b.m[ks[j]]) })
	return ks
}

func (b *bucket) sortedKeys() []string {
	ks := append([]string{}, b.keys...)
	sort.Strings(ks)
	return ks
}

func (b *bucket) writeGroups(w io.Writer, keys []string) {
	for _, k := range keys {
		ts := append([]string{}, b.m[k]...)
		sort.Strings(ts)
		fmt.Fprintf(w, "    %s  (%d)\n", k, len(ts))
		for i, t := range ts {
			if i >= 6 {
				fmt.Fprintf(w, "      … and %d more\n", len(ts)-6)
				break
			}
			fmt.Fprintf(w, "      %s\n", t)
		}
	}
}

func gateName(r selRow) string { return r.Test }

// selectorCoverage prints the census. Returns the process exit code.
func selectorCoverage(w io.Writer) int {
	b, err := os.ReadFile(filepath.Join("scripts", "refresh_parity_hashes.sh"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "selector: cannot read refresh_parity_hashes.sh: %v\n", err)
		return 1
	}
	m := goldenRe.FindStringSubmatch(string(b))
	if m == nil {
		fmt.Fprintf(os.Stderr, "selector: GOLDEN_RE not found — cannot derive the goldens selector\n")
		return 1
	}
	selG, err := regexp.Compile(m[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "selector: GOLDEN_RE does not compile: %v\n", err)
		return 1
	}
	selS := map[string]bool{}
	for _, g := range parityGates {
		selS[g.Test] = true
	}
	if len(selS) == 0 {
		fmt.Fprintf(os.Stderr, "selector: the sweep gate list is EMPTY — cannot derive the sweep selector\n")
		return 1
	}

	var rows []selRow
	for _, d := range selectorScanDirs {
		files, _ := filepath.Glob(filepath.Join(d, "*_test.go"))
		sort.Strings(files)
		for _, f := range files {
			raw, err := os.ReadFile(f)
			if err != nil {
				continue
			}
			txt := string(raw)
			tag := ""
			if t := buildTagRe.FindStringSubmatch(txt); t != nil {
				tag = strings.TrimSpace(t[1])
			}
			envSet := map[string]bool{}
			for _, e := range envRe.FindAllStringSubmatch(txt, -1) {
				envSet[e[1]] = true
			}
			envs := sortedSet(envSet)
			asset := assetSkipRe.MatchString(txt)
			for _, fn := range testFnRe.FindAllStringSubmatch(txt, -1) {
				name := fn[1]
				rows = append(rows, selRow{
					Test: name, File: d + "/" + filepath.Base(f), Tag: tag,
					Envs: envs, Asset: asset,
					Selected: selG.MatchString(name) || selS[name],
				})
			}
		}
	}
	if len(rows) == 0 {
		fmt.Fprintf(os.Stderr, "selector: found ZERO tests — the scanner is broken, not the tree empty\n")
		return 1
	}

	var selected, unselected []selRow
	for _, r := range rows {
		if r.Selected {
			selected = append(selected, r)
		} else {
			unselected = append(unselected, r)
		}
	}

	// DENOMINATOR, stated before the numbers.
	fmt.Fprintf(w, "  EXAMINED: %d test func(s) across %d scanned dir(s) — %s. Packages outside that list are NOT counted here.\n",
		len(rows), len(selectorScanDirs), strings.Join(selectorScanDirs, ", "))
	fmt.Fprintf(w, "  SELECTOR COVERAGE — %d tests in %s/\n", len(rows), strings.Join(selectorScanDirs, "/"))
	fmt.Fprintf(w, "    reached by a selector : %d\n", len(selected))
	fmt.Fprintf(w, "    reached by NONE       : %d\n", len(unselected))
	fmt.Fprintln(w)

	// Of the SELECTED ones, which cannot actually run as invoked — the int8int8 and gpt_oss shapes.
	blocked := newBucket()
	for _, r := range selected {
		switch {
		case r.Tag != "":
			blocked.add("build tag: "+r.Tag, gateName(r))
		case len(r.Envs) > 0:
			blocked.add("env-gated: "+strings.Join(r.Envs, ","), gateName(r))
		case r.Asset:
			blocked.add("asset-gated (skips without a checkpoint)", gateName(r))
		}
	}
	if len(blocked.keys) > 0 {
		fmt.Fprintf(w, "  SELECTED but conditionally unreachable — these are the plumbing gaps:\n")
		blocked.writeGroups(w, blocked.sortedKeys())
		fmt.Fprintln(w)
	}

	// THE UNSELECTED BUCKET, BROKEN DOWN. Added after this census failed to surface a real gap:
	// GOINFER_PREQUANT_GGUF gated two tests that nothing set, so both had been skipping silently —
	// and neither they nor that variable appeared ANYWHERE in the report, because gating was only
	// analysed for the SELECTED tests while these sat in the "reached by NONE" bucket, printed as a
	// bare count and never attributed. An env-gated test that no selector reaches was invisible AS
	// env-gated. Reporting the count without the breakdown is the denominator problem one level in:
	// the number was right and told you nothing.
	unsel := newBucket()
	for _, r := range unselected {
		switch {
		case r.Tag != "":
			unsel.add("build tag: "+r.Tag, gateName(r))
		case len(r.Envs) > 0:
			unsel.add("env-gated: "+strings.Join(r.Envs, ","), gateName(r))
		case r.Asset:
			unsel.add("asset-gated (skips without a checkpoint)", gateName(r))
		default:
			unsel.add("no gate — runs whenever selected, but nothing selects it", gateName(r))
		}
	}
	if len(unsel.keys) > 0 {
		fmt.Fprintf(w, "  REACHED BY NONE (%d), broken down by what ALSO gates them:\n", len(unselected))
		fmt.Fprintf(w, "    A test here runs only if someone invokes it directly. One that is additionally\n")
		fmt.Fprintf(w, "    env- or asset-gated will then SKIP unless that is set too — two silent layers.\n")
		unsel.writeGroups(w, unsel.byCountDesc())
		fmt.Fprintln(w)
		envSet := map[string]bool{}
		for _, r := range unselected {
			for _, e := range r.Envs {
				envSet[e] = true
			}
		}
		if envs := sortedSet(envSet); len(envs) > 0 {
			fmt.Fprintf(w, "    ENV VARS gating otherwise-unreached tests (%d) — set these, or\n", len(envs))
			fmt.Fprintf(w, "    accept that the tests behind them have never run:\n")
			fmt.Fprintf(w, "      %s\n", strings.Join(envs, "  "))
			fmt.Fprintln(w)
		}
	}

	fmt.Fprintf(w, "  A green here means nothing became UNREACHABLE since a person last looked.\n")
	fmt.Fprintf(w, "  It does NOT mean coverage is adequate — a test can be selected and still vacuous.\n")
	return 0
}

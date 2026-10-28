package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// ANSI, matching what the shell gates printed. A gate's output is read by a human deciding
// whether to tag a release; the colours were load-bearing enough that dropping them would be a
// regression in the only interface this program has.
const (
	bold  = "\033[1m"
	red   = "\033[31m"
	green = "\033[32m"
	amber = "\033[33m"
	off   = "\033[0m"
)

type skipRow struct{ Pkg, Test, Reason, Bucket string }

// collectSkips walks the results in stream order and extracts one row per skipped test.
func collectSkips(res *results, topOnly bool) []skipRow {
	var rows []skipRow
	for _, k := range res.order {
		if res.final[k] != "skip" || (topOnly && isSubtest(k)) {
			continue
		}
		reason := skipReason(res.out[k])
		rows = append(rows, skipRow{k.Pkg, k.Test, reason, classifySkip(reason)})
	}
	return rows
}

// reportTally is the shell-gate report shape (heavy_gate, and later parity_sweep/gpu_gate):
// provenance, a per-cell tally, the skip list WITH REASONS, and a verdict. It returns the exit
// code rather than exiting, so the decision is testable.
func reportTally(w io.Writer, cfg *gateConfig, res *results, cells []cellResult, prov provenance) int {
	fmt.Fprintf(w, "%s== %s provenance ==%s\n", bold, cfg.Name, off)
	prov.write(w)
	for _, c := range cells {
		fmt.Fprintf(w, "\n%s== %s ==%s\n", bold, c.Cell.Name, off)
		fmt.Fprintf(w, "  %sPASS %d%s  %sSKIP %d%s  %sFAIL %d%s  (go rc: %d, %s)\n",
			green, c.Pass, off, amber, c.Skip, off, red, c.Fail, off, c.RC, c.Dur.Round(1e9))
		if c.Err != nil {
			fmt.Fprintf(w, "    cell did not start: %v\n", c.Err)
		}
		if c.Hidden {
			// rc-aware honesty: a non-zero exit with no counted failure is a panic, a fatal error,
			// a timeout or a zero-match. Counting only per-test failures would report GREEN on a
			// package that crashed — the "green that means nothing" this gate exists to prevent.
			fmt.Fprintf(w, "    %sgo test exited %d with 0 counted failures — a HIDDEN failure "+
				"(panic / fatal / timeout / 0-match)%s\n", red, c.RC, off)
		}
		if c.LogPath != "" {
			fmt.Fprintf(w, "    raw log: %s\n", c.LogPath)
		}
	}

	var pass, skip, fail int
	for _, c := range cells {
		pass, skip, fail = pass+c.Pass, skip+c.Skip, fail+c.Fail
	}
	writeFailures(w, res, cfg.TopLevelOnly)

	fmt.Fprintf(w, "\n%s== verdict ==%s\n", bold, off)
	fmt.Fprintf(w, "  PASSED %d   SKIPPED %d   FAILED %d\n", pass, skip, fail)
	rows := collectSkips(res, cfg.TopLevelOnly)
	if len(rows) > 0 {
		// A missing checkpoint must never masquerade as coverage — so every skip is listed with
		// its reason, not merely counted.
		fmt.Fprintf(w, "  skipped (missing checkpoint or unmet gate — NOT coverage):\n")
		for _, r := range rows {
			fmt.Fprintf(w, "    %s  %s — %s\n", r.Pkg, r.Test, trunc(r.Reason, 100))
		}
	}
	if pkgFails := writePkgFails(w, res); pkgFails > 0 && cfg.PkgFailIsFailure {
		fail += pkgFails
	}
	if fail > 0 {
		fmt.Fprintf(w, "  %sRED — %d failed%s\n", red, fail, off)
		return 1
	}
	if cfg.ZeroPolicy == "no-pass" && pass == 0 {
		fmt.Fprintf(w, "  %sRED — nothing actually ran (all skipped / no match); a green with 0 tests is not green%s\n", red, off)
		return 1
	}
	fmt.Fprintf(w, "  %sGREEN — %d tests ran and passed (%d skipped)%s\n", green, pass, skip, off)
	return 0
}

// reportCensus is skip_census.py's report shape.
func reportCensus(w io.Writer, cfg *gateConfig, res *results, requireFixtures bool) int {
	var npass, nfail, nskip int
	for k, a := range res.final {
		if cfg.TopLevelOnly && isSubtest(k) {
			continue
		}
		switch a {
		case "pass":
			npass++
		case "fail":
			nfail++
		case "skip":
			nskip++
		}
	}
	total := npass + nfail + nskip

	rows := collectSkips(res, cfg.TopLevelOnly)
	byBucket := map[string][]skipRow{}
	for _, r := range rows {
		byBucket[r.Bucket] = append(byBucket[r.Bucket], r)
	}

	line := strings.Repeat("=", 72)
	fmt.Fprintln(w, line)
	fmt.Fprintln(w, " goinfer test census")
	fmt.Fprintln(w, line)
	fmt.Fprintf(w, "  PASS  %d\n", npass)
	fmt.Fprintf(w, "  FAIL  %d\n", nfail)
	fmt.Fprintf(w, "  SKIP  %d\n", nskip)
	fmt.Fprintf(w, "  total %d\n", total)
	fmt.Fprintf(w, "\n  skip buckets:\n")
	for _, b := range bucketOrder {
		fmt.Fprintf(w, "    %-16s %d\n", b, len(byBucket[b]))
	}

	if nfail > 0 {
		fmt.Fprintf(w, "\n  FAILURES:\n")
		var fails []string
		for k, a := range res.final {
			if a == "fail" && !(cfg.TopLevelOnly && isSubtest(k)) {
				fails = append(fails, fmt.Sprintf("    %s  %s", k.Pkg, k.Test))
			}
		}
		sort.Strings(fails)
		for _, f := range fails {
			fmt.Fprintln(w, f)
		}
	}

	pkgFails := writePkgFails(w, res)

	if other := byBucket["other"]; len(other) > 0 {
		fmt.Fprintf(w, "\n  unclassified skips (inspect — bucket rules may need a new pattern):\n")
		for i, r := range other {
			if i >= 40 {
				break
			}
			fmt.Fprintf(w, "    %s  %s\n      %s\n", r.Pkg, r.Test, trunc(r.Reason, 100))
		}
	}

	rc := 0
	if nfail > 0 {
		rc = 1
	}
	if pkgFails > 0 {
		if cfg.PkgFailIsFailure {
			rc = 1
		} else {
			// Say out loud what is being let through. skip_census.py exits 0 on a package-level
			// fail, and E8 does not change what a gate decides — but a suppressed failure that
			// nobody is told about is indistinguishable from no failure, which is the defect this
			// whole census exists to prevent. So it is preserved AND announced.
			fmt.Fprintf(w, "\n  %sNOTE: %d package-level failure(s) above are NOT counted in this verdict%s\n",
				amber, pkgFails, off)
			fmt.Fprintf(w, "        (parity with scripts/skip_census.py, which exits 0 on them). Read them.\n")
		}
	}

	// A census of ZERO tests is not a clean census — it is the absence of one. An empty stream
	// (bad -tags, a package path matching nothing, a truncated capture, a build error that emitted
	// no test events) lands as PASS 0 / FAIL 0 / SKIP 0, and exiting 0 on that reads as "nothing
	// wrong" when it means "nothing looked". Wherever a zero can mean either, it has to say which.
	if total == 0 && cfg.ZeroPolicy == "no-tests" {
		fmt.Fprintf(w, "\n  ✗ NO TESTS OBSERVED — the -json stream contained no test events.\n")
		fmt.Fprintf(w, "    This is NOT a pass. Usual causes: a build/tag error that produced no test\n")
		fmt.Fprintf(w, "    events, a package path matching nothing, or a truncated recorded stream.\n")
		fmt.Fprintf(w, "    Re-run and check `go test` itself succeeds before reading this census.\n")
		rc = 1
	}
	if requireFixtures && len(byBucket["missing-fixture"]) > 0 {
		fmt.Fprintf(w, "\n  ✗ GOINFER_REQUIRE_FIXTURES=1 and missing-fixture skips present:\n")
		for _, r := range byBucket["missing-fixture"] {
			fmt.Fprintf(w, "      %s  %s  — %s\n", r.Pkg, r.Test, trunc(r.Reason, 80))
		}
		fmt.Fprintf(w, "    A committed-fixture family must run. Regenerate the fixture (scripts/pin_*.py) or fix the box; do not tag on a skipped parity gate.\n")
		rc = 1
	}
	return rc
}

// writePkgFails prints package-level failures — build errors and native crashes, which produce no
// per-test failure at all — and returns how many there were.
func writePkgFails(w io.Writer, res *results) int {
	if len(res.pkgFail) == 0 {
		return 0
	}
	fmt.Fprintf(w, "\n  PACKAGE-LEVEL FAILS (not counted in the per-test tally):\n")
	for _, pkg := range res.pkgFail {
		tag := ""
		if looksLikeCrash(strings.Join(res.pkgOut[pkg], "")) {
			tag = "  ⚠ NATIVE CRASH (known flaky fault 0x10 on metal — shard the run; tests pass in isolation)"
		}
		fmt.Fprintf(w, "    %s%s\n", pkg, tag)
	}
	return len(res.pkgFail)
}

// writeFailures prints each failing test with the source-located lines from its own output. The
// shell gate grepped the raw log for `\.go:[0-9]+:` and took the first 30 lines, which mixed
// several tests' output together; holding structured results means each excerpt is attributable.
func writeFailures(w io.Writer, res *results, topOnly bool) {
	var any bool
	for _, k := range res.order {
		if res.final[k] != "fail" || (topOnly && isSubtest(k)) {
			continue
		}
		if !any {
			fmt.Fprintf(w, "\n%s== failures ==%s\n", bold, off)
			any = true
		}
		fmt.Fprintf(w, "  %s%s  %s%s\n", red, k.Pkg, k.Test, off)
		n := 0
		for _, ln := range res.out[k] {
			ls := strings.TrimRight(ln, "\n")
			if !strings.Contains(ls, ".go:") {
				continue
			}
			fmt.Fprintf(w, "    %s\n", strings.TrimSpace(ls))
			if n++; n >= 10 {
				fmt.Fprintf(w, "    … (truncated; see the raw log)\n")
				break
			}
		}
	}
}

// trunc cuts to n RUNES, not n bytes. Python's `r[:80]` counts characters, and these reasons are
// full of em dashes (3 bytes each) — byte-slicing produced a string two characters shorter than the
// script it replaced, which is a diff in the acceptance-(a) comparison and, worse, could cut a
// multi-byte rune in half and emit mojibake into a release verdict.
func trunc(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// isSubtest reports whether the key names a subtest (`TestFoo/case`) rather than a top-level test. The
// distinction is load-bearing: tallying top-level tests only and tallying every subtest as its own result
// are not the same number, so which applies is a per-config choice (topLevelOnly), not a house style.
func isSubtest(k testKey) bool { return strings.Contains(k.Test, "/") }

// text reconstructs the `go test -v` output this stream carried.
func (r *results) text() string { return strings.Join(r.outAll, "") }

// topLevel returns the top-level tests with the given terminal action, in stream order.
func (r *results) topLevel(action string) []testKey {
	var out []testKey
	for _, k := range r.order {
		if !isSubtest(k) && r.final[k] == action {
			out = append(out, k)
		}
	}
	return out
}

// provenance is the header every gate prints before its verdict. It exists because the same
// invocation error produced a false "15 BLOCKER(S)" three separate times while the tree was fine:
// a gate that reports its FINDING without reporting WHAT IT EXAMINED cannot be debugged from its
// own output.
type provenance struct {
	Commit string      `json:"commit"`
	Dirty  bool        `json:"dirty"`
	Date   string      `json:"date"`
	Host   string      `json:"host"`
	Fields [][2]string `json:"fields"` // gate-specific rows (models dir, run filter, packages, …)
}

func (p provenance) write(w io.Writer) {
	d := ""
	if p.Dirty {
		d = " +dirty"
	}
	fmt.Fprintf(w, "  repo        %s%s\n", p.Commit, d)
	fmt.Fprintf(w, "  date (UTC)  %s\n", p.Date)
	fmt.Fprintf(w, "  host        %s\n", p.Host)
	for _, f := range p.Fields {
		fmt.Fprintf(w, "  %-11s %s\n", f[0], f[1])
	}
	if p.Dirty {
		fmt.Fprintf(w, "  NOTE: working tree DIRTY — this verdict does not describe a committed state.\n")
	}
}

// tally counts one cell's results out of the shared set. Cell-stamped keys make this exact: diffing a
// before/after snapshot of the map loses any test whose result did not change between cells (a skip that
// stayed a skip counted once for two runs).
func (r *results) tally(cellName string, topOnly bool) cellResult {
	var cr cellResult
	for k, act := range r.final {
		if k.Cell != cellName || (topOnly && isSubtest(k)) {
			continue
		}
		switch act {
		case "pass":
			cr.Pass++
		case "skip":
			cr.Skip++
		case "fail":
			cr.Fail++
		}
	}
	return cr
}

// gradeCell is a finished cell's result: its counts from the stream, its exit code, and a non-zero exit with no counted
// failure turned into one hidden failure when the gate says an exit code is a failure. runCell and a replay both call it,
// so a replayed record grades its cells as the run did.
func gradeCell(c cell, cfg *gateConfig, res *results, rc int, logPath string, dur time.Duration) cellResult {
	cr := res.tally(c.Name, cfg.TopLevelOnly)
	cr.Cell, cr.RC, cr.LogPath, cr.Dur = c, rc, logPath, dur
	if rc != 0 && cr.Fail == 0 && cfg.RCIsFailure {
		cr.Fail++
		cr.Hidden = true
	}
	return cr
}

// gradeTallyRun is a tally gate's verdict on a finished run, live or replayed: the census report for a census gate,
// the tally report otherwise.
func gradeTallyRun(w io.Writer, cfg *gateConfig, res *results, cells []cellResult, prov provenance, requireFixtures bool) int {
	if cfg.Decision == "census" {
		prov.write(w)
		return reportCensus(w, cfg, res, requireFixtures)
	}
	return reportTally(w, cfg, res, cells, prov)
}

// lookupTop returns the LAST terminal action recorded for a TOP-LEVEL test of this exact name, across
// every cell and package, and whether it was seen at all. "Last" because the sweep runs some gates twice
// (the plain cell and the realckpt cell); "exact" so TestFoo does not match TestFooBar. Not-seen is a
// FOURTH outcome, not a flavour of skip: a required gate that never ran is the one case where the sweep
// learned nothing at all.
func (r *results) lookupTop(name string) (string, bool) {
	act, found := "", false
	for _, k := range r.order {
		if k.Test != name || isSubtest(k) {
			continue
		}
		if a, ok := r.final[k]; ok {
			act, found = a, true
		}
	}
	return act, found
}

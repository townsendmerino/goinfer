package gatecore

import (
	"cmp"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"time"
)

// ANSI escape codes used across gate reports.
const (
	Bold  = "\033[1m"
	Red   = "\033[31m"
	Green = "\033[32m"
	Amber = "\033[33m"
	Off   = "\033[0m"
)

// SkipRow summarizes one skipped test and its classified skip bucket.
type SkipRow struct{ Pkg, Test, Reason, Bucket string }

// CollectSkips extracts one SkipRow per skipped test in stream order.
func CollectSkips(res *Results, topOnly bool) []SkipRow {
	var rows []SkipRow
	for _, k := range res.Order {
		if res.Final[k] != "skip" || (topOnly && IsSubtest(k)) {
			continue
		}
		reason := SkipReason(res.Out[k])
		rows = append(rows, SkipRow{k.Pkg, k.Test, reason, ClassifySkip(reason)})
	}
	return rows
}

// ReportTally formats the per-cell tally, failures, skip reasons, and verdict, returning the exit code.
func ReportTally(w io.Writer, cfg *GateConfig, res *Results, cells []CellResult, prov Provenance) int {
	fmt.Fprintf(w, "%s== %s provenance ==%s\n", Bold, cfg.Name, Off)
	prov.Write(w)
	for _, c := range cells {
		fmt.Fprintf(w, "\n%s== %s ==%s\n", Bold, c.Cell.Name, Off)
		fmt.Fprintf(w, "  %sPASS %d%s  %sSKIP %d%s  %sFAIL %d%s  (go rc: %d, %s)\n",
			Green, c.Pass, Off, Amber, c.Skip, Off, Red, c.Fail, Off, c.RC, c.Dur.Round(1e9))
		if c.Err != nil {
			fmt.Fprintf(w, "    cell did not start: %v\n", c.Err)
		}
		if c.Hidden {
			fmt.Fprintf(w, "    %sgo test exited %d with 0 counted failures — a HIDDEN failure "+
				"(panic / fatal / timeout / 0-match)%s\n", Red, c.RC, Off)
		}
		if c.LogPath != "" {
			fmt.Fprintf(w, "    raw log: %s\n", c.LogPath)
		}
	}

	var pass, skip, fail int
	for _, c := range cells {
		pass, skip, fail = pass+c.Pass, skip+c.Skip, fail+c.Fail
	}
	WriteFailures(w, res, cfg.TopLevelOnly)

	fmt.Fprintf(w, "\n%s== verdict ==%s\n", Bold, Off)
	fmt.Fprintf(w, "  PASSED %d   SKIPPED %d   FAILED %d\n", pass, skip, fail)
	rows := CollectSkips(res, cfg.TopLevelOnly)
	if len(rows) > 0 {
		fmt.Fprintf(w, "  skipped (missing checkpoint or unmet gate — NOT coverage):\n")
		for _, r := range rows {
			fmt.Fprintf(w, "    %s  %s — %s\n", r.Pkg, r.Test, Trunc(r.Reason, 100))
		}
	}
	if pkgFails := WritePkgFails(w, res); pkgFails > 0 && cfg.PkgFailIsFailure {
		fail += pkgFails
	}
	if fail > 0 {
		fmt.Fprintf(w, "  %sRED — %d failed%s\n", Red, fail, Off)
		return 1
	}
	if cfg.ZeroPolicy == "no-pass" && pass == 0 {
		fmt.Fprintf(w, "  %sRED — nothing actually ran (all skipped / no match); a green with 0 tests is not green%s\n", Red, Off)
		return 1
	}
	fmt.Fprintf(w, "  %sGREEN — %d tests ran and passed (%d skipped)%s\n", Green, pass, skip, Off)
	return 0
}

// ReportCensus formats the whole-tree PASS/SKIP/FAIL census report and returns the exit code.
func ReportCensus(w io.Writer, cfg *GateConfig, res *Results, requireFixtures bool) int {
	var npass, nfail, nskip int
	for k, a := range res.Final {
		if cfg.TopLevelOnly && IsSubtest(k) {
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

	rows := CollectSkips(res, cfg.TopLevelOnly)
	byBucket := map[string][]SkipRow{}
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
	for _, b := range BucketOrder {
		fmt.Fprintf(w, "    %-16s %d\n", b, len(byBucket[b]))
	}

	if nfail > 0 {
		fmt.Fprintf(w, "\n  FAILURES:\n")
		var fails []string
		for k, a := range res.Final {
			if a == "fail" && !(cfg.TopLevelOnly && IsSubtest(k)) {
				fails = append(fails, fmt.Sprintf("    %s  %s", k.Pkg, k.Test))
			}
		}
		slices.Sort(fails)
		for _, f := range fails {
			fmt.Fprintln(w, f)
		}
	}

	pkgFails := WritePkgFails(w, res)

	if other := byBucket["other"]; len(other) > 0 {
		fmt.Fprintf(w, "\n  unclassified skips (inspect — bucket rules may need a new pattern):\n")
		for i, r := range other {
			if i >= 40 {
				break
			}
			fmt.Fprintf(w, "    %s  %s\n      %s\n", r.Pkg, r.Test, Trunc(r.Reason, 100))
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
			fmt.Fprintf(w, "\n  %sNOTE: %d package-level failure(s) above are NOT counted in this verdict%s\n",
				Amber, pkgFails, Off)
			fmt.Fprintf(w, "        (parity with scripts/skip_census.py, which exits 0 on them). Read them.\n")
		}
	}

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
			fmt.Fprintf(w, "      %s  %s  — %s\n", r.Pkg, r.Test, Trunc(r.Reason, 80))
		}
		fmt.Fprintf(w, "    A committed-fixture family must run. Regenerate the fixture (scripts/pin_*.py) or fix the box; do not tag on a skipped parity gate.\n")
		rc = 1
	}
	return rc
}

// WritePkgFails prints package-level failures and returns how many occurred.
func WritePkgFails(w io.Writer, res *Results) int {
	if len(res.PkgFail) == 0 {
		return 0
	}
	fmt.Fprintf(w, "\n  PACKAGE-LEVEL FAILS (not counted in the per-test tally):\n")
	for _, pkg := range res.PkgFail {
		tag := ""
		if LooksLikeCrash(strings.Join(res.PkgOut[pkg], "")) {
			tag = "  ⚠ NATIVE CRASH (known flaky fault 0x10 on metal — shard the run; tests pass in isolation)"
		}
		fmt.Fprintf(w, "    %s%s\n", pkg, tag)
	}
	return len(res.PkgFail)
}

// WriteFailures prints each failing test with up to 10 source-located lines from its own output.
func WriteFailures(w io.Writer, res *Results, topOnly bool) {
	var any bool
	for _, k := range res.Order {
		if res.Final[k] != "fail" || (topOnly && IsSubtest(k)) {
			continue
		}
		if !any {
			fmt.Fprintf(w, "\n%s== failures ==%s\n", Bold, Off)
			any = true
		}
		fmt.Fprintf(w, "  %s%s  %s%s\n", Red, k.Pkg, k.Test, Off)
		n := 0
		for _, ln := range res.Out[k] {
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

// Trunc cuts s to at most n runes without splitting multi-byte UTF-8 sequences.
func Trunc(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// IsSubtest reports whether k names a subtest (`TestFoo/case`).
func IsSubtest(k TestKey) bool { return strings.Contains(k.Test, "/") }

// Text reconstructs the raw output stream carried by r.
func (r *Results) Text() string { return strings.Join(r.OutAll, "") }

// TopLevel returns top-level tests with the given terminal action in stream order.
func (r *Results) TopLevel(action string) []TestKey {
	var out []TestKey
	for _, k := range r.Order {
		if !IsSubtest(k) && r.Final[k] == action {
			out = append(out, k)
		}
	}
	return out
}

// Provenance records repository commit, dirty flag, UTC timestamp, host, and gate-specific header rows.
type Provenance struct {
	Commit string      `json:"commit"`
	Dirty  bool        `json:"dirty"`
	Date   string      `json:"date"`
	Host   string      `json:"host"`
	Fields [][2]string `json:"fields"`
}

// Write prints p's provenance header lines to w.
func (p Provenance) Write(w io.Writer) {
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

// Tally counts cellName's results from r.
func (r *Results) Tally(cellName string, topOnly bool) CellResult {
	var cr CellResult
	for k, act := range r.Final {
		if k.Cell != cellName || (topOnly && IsSubtest(k)) {
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

// GradeCell summarizes a finished cell's counts and turns a non-zero exit with zero counted failures into a hidden failure when configured.
func GradeCell(c Cell, cfg *GateConfig, res *Results, rc int, logPath string, dur time.Duration) CellResult {
	cr := res.Tally(c.Name, cfg.TopLevelOnly)
	cr.Cell, cr.RC, cr.LogPath, cr.Dur = c, rc, logPath, dur
	if rc != 0 && cr.Fail == 0 && cfg.RCIsFailure {
		cr.Fail++
		cr.Hidden = true
	}
	return cr
}

// GradeTallyRun grades a finished census or tally run.
func GradeTallyRun(w io.Writer, cfg *GateConfig, res *Results, cells []CellResult, prov Provenance, requireFixtures bool) int {
	if cfg.Decision == "census" {
		prov.Write(w)
		return ReportCensus(w, cfg, res, requireFixtures)
	}
	return ReportTally(w, cfg, res, cells, prov)
}

// LookupTop returns the last terminal action recorded for a top-level test of exact name across all cells.
func (r *Results) LookupTop(name string) (string, bool) {
	act, found := "", false
	for _, k := range r.Order {
		if k.Test != name || IsSubtest(k) {
			continue
		}
		if a, ok := r.Final[k]; ok {
			act, found = a, true
		}
	}
	return act, found
}

// TopOutput returns the output lines of the last terminal top-level run of name.
func (r *Results) TopOutput(name string) []string {
	var out []string
	for _, k := range r.Order {
		if k.Test != name || IsSubtest(k) {
			continue
		}
		if _, ok := r.Final[k]; ok {
			out = r.Out[k]
		}
	}
	return out
}

// GradeCellError builds a hidden-failure CellResult for a cell that failed before producing a stream.
func GradeCellError(c Cell, rc int, err error, logPath string, dur time.Duration) CellResult {
	return CellResult{Cell: c, RC: rc, Err: err, LogPath: logPath, Dur: dur, Fail: 1, Hidden: true}
}

// SortedSet returns the sorted keys of m whose value is true.
func SortedSet(m map[string]bool) []string {
	var out []string
	for k, v := range m {
		if v {
			out = append(out, k)
		}
	}
	slices.Sort(out)
	return out
}

// OrDash returns "-" when s is empty, else s.
func OrDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// WrapWords wraps words to width columns, prefixing continuation lines with indent.
func WrapWords(words []string, width int, indent string) string {
	var b strings.Builder
	col := 0
	for i, wd := range words {
		if i > 0 {
			if col+1+len(wd) > width {
				b.WriteString("\n" + indent)
				col = 0
			} else {
				b.WriteByte(' ')
				col++
			}
		}
		b.WriteString(wd)
		col += len(wd)
	}
	return b.String()
}

// TopTests returns up to n test names from p.ByTest ordered by largest sampled VmRSS first.
func (p ProcPeak) TopTests(n int) []string {
	names := slices.Collect(maps.Keys(p.ByTest))
	slices.SortFunc(names, func(a, b string) int {
		return cmp.Or(cmp.Compare(p.ByTest[b], p.ByTest[a]), cmp.Compare(a, b))
	})
	if len(names) > n {
		names = names[:n]
	}
	return names
}

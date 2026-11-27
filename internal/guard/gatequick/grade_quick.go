package gatequick

import (
	"cmp"
	"fmt"
	"io"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
	"github.com/townsendmerino/goinfer/internal/guard/gatehygiene"
)

func (jb *quickJob) dur() time.Duration { return jb.end.Sub(jb.start) }

// ok is the job's own verdict.
func (jb *quickJob) ok() bool {
	if jb.Kind == "lint" {
		if jb.Canary != "" {
			return strings.Contains(jb.Out, jb.Canary)
		}
		return jb.RC == 0 && jb.Err == nil && (!jb.QuietOK || strings.TrimSpace(jb.Out) == "")
	}
	return jb.RC == 0 && jb.Err == nil && jb.Fail == 0 && !jb.Hidden
}

func statusWord(jb *quickJob) string {
	switch {
	case !jb.ok():
		return "FAIL"
	case jb.Kind == "test" && jb.Cached:
		return "CACHED"
	case jb.Kind == "test" && jb.NoTest:
		return "NOTESTS"
	case jb.Kind == "test" && jb.Pass == 0 && jb.Skip > 0:
		return "ALLSKIP"
	}
	return "PASS"
}

// GradeQuick is quick's verdict on finished jobs, live or replayed: the lint and test tables, each failing cell's
// lines or recorded log tail, and the verdict block, ending at end. goldens is the number of goldens the selection
// triggered. It fills sum for the quick-full record.
func GradeQuick(w io.Writer, jobs []*quickJob, notRun, deferred []*quickCell, goldens int, t0, end time.Time, verbose bool, sum *QuickSummary) int {
	var lint, tests []*quickJob
	for _, jb := range jobs {
		if jb.Kind == "lint" {
			lint = append(lint, jb)
		} else {
			tests = append(tests, jb)
		}
	}
	failures := 0
	if len(lint) > 0 {
		fmt.Fprintf(w, "\n%s== lint ==%s\n", gatecore.Bold, gatecore.Off)
		for _, jb := range lint {
			st, col := "PASS", gatecore.Green
			if !jb.ok() {
				st, col = "FAIL", gatecore.Red
				failures++
			}
			fmt.Fprintf(w, "  %s%-4s%s  %-52s %-9s %6s   %s\n", col, st, gatecore.Off, jb.Name, jb.Detail, jb.dur().Round(100*time.Millisecond), jb.Mirrors)
			if !jb.ok() {
				msg := jb.Out
				if jb.Err != nil {
					msg = jb.Err.Error() + "\n" + msg
				}
				if jb.QuietOK {
					msg = "unformatted:\n" + msg
				}
				if jb.Canary != "" {
					msg = "the canary's U1000 did NOT appear — this staticcheck cannot be trusted to find anything:\n" + msg
				}
				for i, ln := range strings.Split(strings.TrimRight(msg, "\n"), "\n") {
					if i >= 15 {
						fmt.Fprintf(w, "        … (see %s)\n", jb.LogPath)
						break
					}
					fmt.Fprintf(w, "        %s\n", ln)
				}
			}
		}
	}

	fmt.Fprintf(w, "\n%s== tests ==%s\n", gatecore.Bold, gatecore.Off)
	var pass, skip, fail, ran, cached, allSkip int
	var goldenJobs []*quickJob
	var vacuousAffected []string
	skipBuckets := map[string]int{}
	var skipRows []gatecore.SkipRow
	sorted := slices.Clone(tests)
	slices.SortStableFunc(sorted, func(a, b *quickJob) int { return cmp.Compare(a.Name, b.Name) })
	for _, jb := range sorted {
		pass, skip, fail = pass+jb.Pass, skip+jb.Skip, fail+jb.Fail
		if jb.Hidden || jb.Err != nil {
			fail++
		}
		if jb.Cached {
			cached++
		} else {
			ran++
		}
		st := statusWord(jb)
		col := gatecore.Green
		switch st {
		case "FAIL":
			col = gatecore.Red
			failures++
		case "ALLSKIP", "NOTESTS":
			col = gatecore.Amber
			allSkip++
			if jb.Cell.Tier == "affected" {
				vacuousAffected = append(vacuousAffected, jb.Name)
			}
		}
		if jb.Cell.Golden {
			goldenJobs = append(goldenJobs, jb)
		}
		fmt.Fprintf(w, "  %s%-7s%s %-34s %-13s %7s  %4d pass %4d skip %3d fail\n", col, st, gatecore.Off, jb.Name, jb.Cell.Tier,
			jb.dur().Round(100*time.Millisecond), jb.Pass, jb.Skip, jb.Fail)
		if jb.res != nil {
			for _, r := range gatecore.CollectSkips(jb.res, true) {
				skipBuckets[r.Bucket]++
				skipRows = append(skipRows, r)
			}
		}
	}
	for _, c := range notRun {
		fmt.Fprintf(w, "  %s%-7s%s %-34s %s\n", gatecore.Amber, "NOT RUN", gatecore.Off, c.Name, c.NotRun)
	}
	for _, c := range deferred {
		fmt.Fprintf(w, "  %s%-8s%s %-33s %s\n", gatecore.Amber, "DEFERRED", gatecore.Off, c.Name, c.Deferred)
	}

	// Each failing cell: its source-located lines, or — for a failure with no per-test FAIL (a
	// build error, a panic, a timeout) — the tail of its log, which is the only place it shows.
	for _, jb := range sorted {
		if jb.ok() {
			continue
		}
		fmt.Fprintf(w, "\n  %s%s%s — log: %s\n", gatecore.Red, jb.Name, gatecore.Off, jb.LogPath)
		if jb.Err != nil {
			fmt.Fprintf(w, "    did not run: %v\n", jb.Err)
		}
		if jb.res != nil && jb.Fail > 0 {
			gatecore.WriteFailures(w, jb.res, true)
		}
		if jb.Hidden {
			fmt.Fprintf(w, "    go test exited %d with 0 counted failures (build error / panic / timeout); log tail:\n", jb.RC)
			for _, ln := range jb.LogTail {
				fmt.Fprintf(w, "      %s\n", ln)
			}
		}
	}

	fmt.Fprintf(w, "\n%s== verdict ==%s\n", gatecore.Bold, gatecore.Off)
	fmt.Fprintf(w, "  tests    PASSED %d   SKIPPED %d   FAILED %d   (top-level tests; a skip is not a pass)\n", pass, skip, fail)
	fmt.Fprintf(w, "  cells    %d: %d ran, %d replayed from the test cache, %d not runnable on this host, %d deferred to the pre-tag tier\n",
		len(tests)+len(notRun)+len(deferred), ran, cached, len(notRun), len(deferred))
	if sum != nil {
		*sum = QuickSummary{Pass: pass, Skip: skip, Fail: fail, Cells: len(tests) + len(notRun) + len(deferred), Ran: ran, Cached: cached}
		for _, c := range notRun {
			sum.NotRun = append(sum.NotRun, c.Name)
		}
	}
	if len(skipBuckets) > 0 {
		var parts []string
		for _, b := range gatecore.BucketOrder {
			if n := skipBuckets[b]; n > 0 {
				parts = append(parts, fmt.Sprintf("%s %d", b, n))
			}
		}
		fmt.Fprintf(w, "  skips    %s\n", strings.Join(parts, ", "))
		if verbose {
			for _, r := range skipRows {
				fmt.Fprintf(w, "    %s  %s — %s\n", r.Pkg, r.Test, gatecore.Trunc(r.Reason, 100))
			}
		}
	}
	lintFail := 0
	for _, jb := range lint {
		if !jb.ok() {
			lintFail++
		}
	}
	if len(lint) > 0 {
		fmt.Fprintf(w, "  lint     %d step(s): %d pass, %d fail\n", len(lint), len(lint)-lintFail, lintFail)
	} else {
		fmt.Fprintf(w, "  lint     not run\n")
	}
	if goldens > 0 {
		gp, gs, gf := 0, 0, 0
		for _, jb := range goldenJobs {
			gp, gs, gf = gp+jb.Pass, gs+jb.Skip, gf+jb.Fail
		}
		fmt.Fprintf(w, "  goldens  %d passed, %d skipped, %d failed\n", gp, gs, gf)
		// ZERO GOLDENS RAN is not a green: a shared-set change with no numeric proof is exactly
		// what scripts/refresh_parity_hashes.sh refuses on, for the same reason.
		if gp == 0 && gf == 0 && len(goldenJobs) > 0 {
			fmt.Fprintf(w, "  %sgoldens VACUOUS — a shared set changed and no forward golden ran (fixtures absent?)%s\n", gatecore.Red, gatecore.Off)
			failures++
		}
	} else {
		fmt.Fprintf(w, "  goldens  not triggered\n")
	}
	fmt.Fprintf(w, "  wall     %s (%s → %s)\n", end.Sub(t0).Round(time.Second), t0.Format("15:04:05"), end.Format("15:04:05"))
	if len(notRun) > 0 {
		fmt.Fprintf(w, "  %sNOT COVERED on this host%s: %d cell(s) of module(s) that do not build here — run quick on a box that builds them.\n",
			gatecore.Amber, gatecore.Off, len(notRun))
	}
	if len(deferred) > 0 {
		n := 0
		for _, c := range deferred {
			n += strings.Count(c.Run, "|") + 1
		}
		fmt.Fprintf(w, "  %sDEFERRED to the pre-tag tier%s: %d cell(s), %d test(s) not run here — `go run ./cmd/gate quick -full` runs them.\n",
			gatecore.Amber, gatecore.Off, len(deferred), n)
	}
	if len(vacuousAffected) > 0 {
		fmt.Fprintf(w, "  %sUNVERIFIED%s: affected cell(s) in which every test skipped: %s\n", gatecore.Amber, gatecore.Off, strings.Join(vacuousAffected, ", "))
	}
	if failures > 0 {
		fmt.Fprintf(w, "  %sRED — %d failing step(s)/cell(s)%s\n", gatecore.Red, failures, gatecore.Off)
		return 1
	}
	fmt.Fprintf(w, "  %sGREEN — %d tests passed (%d skipped)%s\n", gatecore.Green, pass, skip, gatecore.Off)
	return 0
}

// GradeQuickTest derives a test job's outcome from its parsed stream and exit code: the top-level counts, whether go
// test replayed it from the cache, a non-zero exit with no counted failure (hidden), and a clean exit with no test.
func GradeQuickTest(jb *quickJob) {
	for k, act := range jb.res.Final {
		if gatecore.IsSubtest(k) {
			continue
		}
		switch act {
		case "pass":
			jb.Pass++
		case "skip":
			jb.Skip++
		case "fail":
			jb.Fail++
		}
	}
	for _, lines := range jb.res.PkgOut {
		for _, ln := range lines {
			if strings.HasPrefix(ln, "ok ") && strings.Contains(ln, "(cached)") {
				jb.Cached = true
			}
		}
	}
	// rc-aware, like the heavy tier: a build error, a panic in a goroutine, a timeout or a fatal
	// error exits non-zero WITHOUT a per-test FAIL line, and counting only FAIL lines would call a
	// package that never ran green.
	jb.Hidden = jb.RC != 0 && jb.Fail == 0
	jb.NoTest = jb.RC == 0 && jb.Pass+jb.Skip+jb.Fail == 0
}

// Valid says whether the run is one a release can lean on: every cell run, green, on a clean tree.
func (r QuickRecord) Valid() (bool, string) {
	switch {
	case !r.Full:
		return false, "the run deferred cells (not -full)"
	case r.Verdict != "PASS":
		return false, "verdict " + r.Verdict
	case r.Dirty:
		return false, "the run was made on a dirty tree"
	case r.Pass == 0:
		return false, "no test passed"
	}
	return true, ""
}

// QuickCoverageR is, for one operating system, whether the records cover head. The newest record made there at the
// same code decides, whatever its verdict, so a later FAIL at unchanged code is not hidden by an earlier PASS.
// changed lists the paths that differ between two revisions.
func QuickCoverageR(recs []QuickRecord, goos, head string, changed func(from, to string) ([]string, error)) (cover, failed *QuickRecord, why string) {
	var newestWhy string
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		if r.OS != goos {
			continue
		}
		paths, err := changed(r.Commit, head)
		if err != nil {
			return nil, nil, fmt.Sprintf("could not compare %s with HEAD: %v", r.Commit[:8], err)
		}
		first := ""
		for _, p := range paths {
			if QuickRecordVoidedBy(p) {
				first = p
				break
			}
		}
		if first == "" {
			if ok, why := r.Valid(); !ok {
				return nil, r, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, r.Commit[:8], why)
			}
			return r, nil, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, r.Commit[:8], first)
		}
	}
	if newestWhy == "" {
		return nil, nil, "no record"
	}
	return nil, nil, newestWhy
}

// QuickCoverage is QuickCoverageR without the record that failed to count.
func QuickCoverage(recs []QuickRecord, goos, head string, changed func(from, to string) ([]string, error)) (*QuickRecord, string) {
	cover, _, why := QuickCoverageR(recs, goos, head, changed)
	return cover, why
}

// QuickRecordVoidedBy reports whether a change to path, since a run, voids the run. Everything a test can compile or
// read does; the documentation tree, the website, the CI definitions, the CHANGELOG's entries (changelog.d/) and the
// top-level prose files do not, so a release commit that writes the CHANGELOG and commits the records keeps its run.
func QuickRecordVoidedBy(path string) bool {
	for _, p := range []string{"docs/", "apps/site/", ".github/", gatehygiene.ChangelogFragmentDir + "/"} {
		if strings.HasPrefix(path, p) {
			return false
		}
	}
	return strings.Contains(path, "/") || !strings.HasSuffix(path, ".md")
}

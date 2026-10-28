package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
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

// gradeQuick is quick's verdict on finished jobs, live or replayed: the lint and test tables, each failing cell's
// lines or recorded log tail, and the verdict block, ending at end. goldens is the number of goldens the selection
// triggered. It fills sum for the quick-full record.
func gradeQuick(w io.Writer, jobs []*quickJob, notRun, deferred []*quickCell, goldens int, t0, end time.Time, verbose bool, sum *quickSummary) int {
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
		fmt.Fprintf(w, "\n%s== lint ==%s\n", bold, off)
		for _, jb := range lint {
			st, col := "PASS", green
			if !jb.ok() {
				st, col = "FAIL", red
				failures++
			}
			fmt.Fprintf(w, "  %s%-4s%s  %-52s %-9s %6s   %s\n", col, st, off, jb.Name, jb.Detail, jb.dur().Round(100*time.Millisecond), jb.Mirrors)
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

	fmt.Fprintf(w, "\n%s== tests ==%s\n", bold, off)
	var pass, skip, fail, ran, cached, allSkip int
	var goldenJobs []*quickJob
	var vacuousAffected []string
	skipBuckets := map[string]int{}
	var skipRows []skipRow
	sorted := append([]*quickJob(nil), tests...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
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
		col := green
		switch st {
		case "FAIL":
			col = red
			failures++
		case "ALLSKIP", "NOTESTS":
			col = amber
			allSkip++
			if jb.Cell.Tier == "affected" {
				vacuousAffected = append(vacuousAffected, jb.Name)
			}
		}
		if jb.Cell.Golden {
			goldenJobs = append(goldenJobs, jb)
		}
		fmt.Fprintf(w, "  %s%-7s%s %-34s %-13s %7s  %4d pass %4d skip %3d fail\n", col, st, off, jb.Name, jb.Cell.Tier,
			jb.dur().Round(100*time.Millisecond), jb.Pass, jb.Skip, jb.Fail)
		if jb.res != nil {
			for _, r := range collectSkips(jb.res, true) {
				skipBuckets[r.Bucket]++
				skipRows = append(skipRows, r)
			}
		}
	}
	for _, c := range notRun {
		fmt.Fprintf(w, "  %s%-7s%s %-34s %s\n", amber, "NOT RUN", off, c.Name, c.NotRun)
	}
	for _, c := range deferred {
		fmt.Fprintf(w, "  %s%-8s%s %-33s %s\n", amber, "DEFERRED", off, c.Name, c.Deferred)
	}

	// Each failing cell: its source-located lines, or — for a failure with no per-test FAIL (a
	// build error, a panic, a timeout) — the tail of its log, which is the only place it shows.
	for _, jb := range sorted {
		if jb.ok() {
			continue
		}
		fmt.Fprintf(w, "\n  %s%s%s — log: %s\n", red, jb.Name, off, jb.LogPath)
		if jb.Err != nil {
			fmt.Fprintf(w, "    did not run: %v\n", jb.Err)
		}
		if jb.res != nil && jb.Fail > 0 {
			writeFailures(w, jb.res, true)
		}
		if jb.Hidden {
			fmt.Fprintf(w, "    go test exited %d with 0 counted failures (build error / panic / timeout); log tail:\n", jb.RC)
			for _, ln := range jb.LogTail {
				fmt.Fprintf(w, "      %s\n", ln)
			}
		}
	}

	fmt.Fprintf(w, "\n%s== verdict ==%s\n", bold, off)
	fmt.Fprintf(w, "  tests    PASSED %d   SKIPPED %d   FAILED %d   (top-level tests; a skip is not a pass)\n", pass, skip, fail)
	fmt.Fprintf(w, "  cells    %d: %d ran, %d replayed from the test cache, %d not runnable on this host, %d deferred to the pre-tag tier\n",
		len(tests)+len(notRun)+len(deferred), ran, cached, len(notRun), len(deferred))
	if sum != nil {
		*sum = quickSummary{Pass: pass, Skip: skip, Fail: fail, Cells: len(tests) + len(notRun) + len(deferred), Ran: ran, Cached: cached}
		for _, c := range notRun {
			sum.NotRun = append(sum.NotRun, c.Name)
		}
	}
	if len(skipBuckets) > 0 {
		var parts []string
		for _, b := range bucketOrder {
			if n := skipBuckets[b]; n > 0 {
				parts = append(parts, fmt.Sprintf("%s %d", b, n))
			}
		}
		fmt.Fprintf(w, "  skips    %s\n", strings.Join(parts, ", "))
		if verbose {
			for _, r := range skipRows {
				fmt.Fprintf(w, "    %s  %s — %s\n", r.Pkg, r.Test, trunc(r.Reason, 100))
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
			fmt.Fprintf(w, "  %sgoldens VACUOUS — a shared set changed and no forward golden ran (fixtures absent?)%s\n", red, off)
			failures++
		}
	} else {
		fmt.Fprintf(w, "  goldens  not triggered\n")
	}
	fmt.Fprintf(w, "  wall     %s (%s → %s)\n", end.Sub(t0).Round(time.Second), t0.Format("15:04:05"), end.Format("15:04:05"))
	if len(notRun) > 0 {
		fmt.Fprintf(w, "  %sNOT COVERED on this host%s: %d cell(s) of module(s) that do not build here — run quick on a box that builds them.\n",
			amber, off, len(notRun))
	}
	if len(deferred) > 0 {
		n := 0
		for _, c := range deferred {
			n += strings.Count(c.Run, "|") + 1
		}
		fmt.Fprintf(w, "  %sDEFERRED to the pre-tag tier%s: %d cell(s), %d test(s) not run here — `go run ./cmd/gate quick -full` runs them.\n",
			amber, off, len(deferred), n)
	}
	if len(vacuousAffected) > 0 {
		fmt.Fprintf(w, "  %sUNVERIFIED%s: affected cell(s) in which every test skipped: %s\n", amber, off, strings.Join(vacuousAffected, ", "))
	}
	if failures > 0 {
		fmt.Fprintf(w, "  %sRED — %d failing step(s)/cell(s)%s\n", red, failures, off)
		return 1
	}
	fmt.Fprintf(w, "  %sGREEN — %d tests passed (%d skipped)%s\n", green, pass, skip, off)
	return 0
}

// gradeQuickTest derives a test job's outcome from its parsed stream and exit code: the top-level counts, whether go
// test replayed it from the cache, a non-zero exit with no counted failure (hidden), and a clean exit with no test.
func gradeQuickTest(jb *quickJob) {
	for k, act := range jb.res.final {
		if isSubtest(k) {
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
	for _, lines := range jb.res.pkgOut {
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

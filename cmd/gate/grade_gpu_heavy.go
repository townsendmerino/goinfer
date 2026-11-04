package main

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// The gpu gate's CUDA heavy tier as graders (SB12, docs/tasks/task-subsystems-2026-10.md): the partition block, each main-group
// process's memory line, the two reconciliations and the drain, fresh-process and tier verdicts are functions of what the
// runner recorded (heavyInput) and of each process's graded cell, results and text. gpuGate.heavyTier prints them, live and
// in `gate gpu -replay`.

const heavyHeader = "2c. heavy tier (GOINFER_HEAVY_TESTS=1 — real models; main group one process per model family)"

// heavyInput is what the heavy tier read beside its cells. Skipped is GOINFER_GATE_SKIP_HEAVY. Drain and Iso are the tests
// the drainsDevice and needsFreshProcess markers name. Total is ./cuda/'s listed test count; ListErr, PartitionErr and
// KernelsErr are why the listing, the model-family partition or the kernels package's listing failed. Groups are the main
// group's processes, the kernels one last, and Problems the tests that fit no group. Peaks is each process's memory, in
// Groups order, and Secs the tier's wall time.
type heavyInput struct {
	Skipped      bool         `json:"skipped,omitempty"`
	Drain        []string     `json:"drain,omitempty"`
	Total        int          `json:"total,omitempty"`
	ListErr      string       `json:"list_err,omitempty"`
	PartitionErr string       `json:"partition_err,omitempty"`
	KernelsErr   string       `json:"kernels_err,omitempty"`
	Kernels      int          `json:"kernels,omitempty"`
	Listed       []string     `json:"listed,omitempty"` // every test the main group's processes must account for: ./cuda/'s and the kernels package's
	Groups       []heavyGroup `json:"groups,omitempty"`
	Problems     []string     `json:"problems,omitempty"`
	Iso          []string     `json:"iso,omitempty"`
	Run          []string     `json:"run,omitempty"` // -heavy-group: the groups this run is narrowed to; none runs every group
	Peaks        []procPeak   `json:"peaks,omitempty"`
	Secs         int          `json:"secs,omitempty"`
}

// gradeHeavyStart is the tier's verdicts before any process runs, and whether the processes run: a skipped tier, an empty
// drain derivation (a failure that does not stop the tier), and a listing or partition that failed (which does), then the
// partition block itself with a failure naming the tests that fit no group.
func gradeHeavyStart(in heavyInput) ([]groupVerdict, bool) {
	if in.Skipped {
		return []groupVerdict{{State: groupSkip, Line: "heavy tier (GOINFER_GATE_SKIP_HEAVY set) — the real-model gates did NOT run: 26B expert\n" +
			"        streaming, real-weight GEMV parity, resident spec-serve, the bandwidth benchmarks"}}, false
	}
	var out []groupVerdict
	if len(in.Drain) == 0 {
		out = append(out, groupVerdict{State: groupFail, Line: "heavy tier partition: the marker derivation found ZERO draining tests",
			Notes: []string{"drainsDevice() marker matched nothing — the derivation is broken, not the tree clean"}})
	}
	switch {
	case in.ListErr != "" || in.Total == 0:
		return append(out, groupVerdict{State: groupFail,
			Line:  fmt.Sprintf("heavy tier: `go test -list` found no tests in ./cuda/, so the main tier cannot be partitioned (%s)", errText(in.ListErr)),
			Notes: []string{"heavy tier did not run: the ./cuda/ test listing was empty or failed to build"}}), false
	case in.PartitionErr != "":
		return append(out, groupVerdict{State: groupFail, Line: "heavy tier partition: " + in.PartitionErr,
			Notes: []string{"heavy tier did not run: the model-family derivation could not read cuda/*_test.go"}}), false
	case in.KernelsErr != "" || in.Kernels == 0:
		return append(out, groupVerdict{State: groupFail,
			Line:  fmt.Sprintf("heavy tier: `go test -list` found no tests in %s (%s)", cudaKernelsPkg, errText(in.KernelsErr)),
			Notes: []string{"heavy tier did not run: the " + cudaKernelsPkg + " test listing was empty or failed to build"}}), false
	}
	lines := []string{
		"      partition (drain group from drainsDevice() in cuda/drain_marker_test.go; main group by model family, heavyFamilies):",
		fmt.Sprintf("        package total     %d test(s)   [go test -list '.*'], plus %d in %s", in.Total, in.Kernels, cudaKernelsPkg),
		fmt.Sprintf("        drain group        %d test(s)   %s", len(in.Drain), strings.Join(in.Drain, " ")),
		fmt.Sprintf("        main group         %d test(s)   [complement, by construction], %d process(es):", in.Total-len(in.Drain), len(in.Groups)),
	}
	for _, h := range in.Groups {
		lines = append(lines, fmt.Sprintf("          %-14s %d test(s)", h.Label, len(h.Tests)))
	}
	var narrowed []string
	if len(in.Run) > 0 {
		groups, _, drain, iso := heavyRunSet(in)
		lines = append(lines, fmt.Sprintf("        NARROWED by -heavy-group to %s: %d of %d process(es), with %d drain and %d fresh-process test(s) inside them",
			strings.Join(in.Run, ", "), len(groups), len(in.Groups), len(drain), len(iso)))
		narrowed = []string{"heavy tier NARROWED by -heavy-group to " + strings.Join(in.Run, ", ") + ": this run is not a release verdict"}
	}
	if len(in.Problems) == 0 {
		return append(out, groupVerdict{State: groupNone, Lines: lines, Notes: narrowed}), true
	}
	var probs []string
	for _, p := range in.Problems {
		probs = append(probs, "        "+p)
	}
	return append(out, groupVerdict{State: groupFail, Pre: lines, Line: fmt.Sprintf("heavy tier partition: %d test(s) fit no group", len(in.Problems)),
		Lines: probs, Notes: append(narrowed, fmt.Sprintf("heavy tier partition: %d test(s) fit no group; add a heavyFamilies row (cmd/gate/heavy_groups.go)", len(in.Problems)))}), true
}

// heavyRunSet is what a run runs: the main-group processes, the tests they must account for, and the drain and fresh-process
// tests. A run narrowed by -heavy-group keeps the named processes and only the listed, drain and fresh-process tests inside
// them, so its reconciliations are exact over what it ran.
func heavyRunSet(in heavyInput) (groups []heavyGroup, listed, drain, iso []string) {
	if len(in.Run) == 0 {
		return in.Groups, in.Listed, in.Drain, in.Iso
	}
	keep := map[string]bool{}
	for _, l := range in.Run {
		keep[l] = true
	}
	inRun := map[string]bool{}
	for _, h := range in.Groups {
		if keep[h.Label] {
			groups = append(groups, h)
			for _, t := range h.Tests {
				inRun[t] = true
			}
		}
	}
	pick := func(ts []string) []string {
		var out []string
		for _, t := range ts {
			if inRun[t] {
				out = append(out, t)
			}
		}
		return out
	}
	return groups, pick(in.Listed), pick(in.Drain), pick(in.Iso)
}

// errText is a recorded error's text as fmt's %v printed a nil error.
func errText(s string) string {
	if s == "" {
		return "<nil>"
	}
	return s
}

// heavyProcLines is what the tier prints after one main-group process: its memory line, a note when every test in it
// skipped, and the path of its full output when it failed. It also returns the process's row for the gate record.
func heavyProcLines(h heavyGroup, cr cellResult, p procPeak) ([]string, []string, heavyProcRecord) {
	rec := heavyProcRecord{Group: h.Label, Tests: len(h.Tests), WallS: int(cr.Dur.Seconds()), Samples: p.Samples}
	var lines, notes []string
	if p.Samples == 0 {
		lines = append(lines, fmt.Sprintf("      %s memory: NOT SAMPLED (no test binary found under /proc)", h.Label))
	} else {
		rec.VmHWMMB, rec.VmSwapPkMB = p.HWMKB/1024, p.SwapPeakKB/1024
		var top []string
		for _, n := range p.topTests(3) {
			top = append(top, fmt.Sprintf("%s %d MB", n, p.ByTest[n]/1024))
		}
		lines = append(lines, fmt.Sprintf("      %s memory: VmHWM %d MB, peak VmSwap %d MB, wall %ds (pid %d, %d samples); largest RSS in flight: %s",
			h.Label, rec.VmHWMMB, rec.VmSwapPkMB, rec.WallS, p.PID, p.Samples, strings.Join(top, ", ")))
	}
	if cr.Pass == 0 && cr.Fail == 0 && cr.Skip > 0 {
		lines = append(lines, fmt.Sprintf("      %s: all %d test(s) SKIPPED — its checkpoints are likely absent (census below)", h.Label, cr.Skip))
		notes = append(notes, fmt.Sprintf("heavy group %s: all %d test(s) skipped", h.Label, cr.Skip))
	}
	if cr.RC != 0 {
		lines = append(lines, fmt.Sprintf("      %s full output: %s", h.Label, cr.LogPath))
	}
	return lines, notes, rec
}

// gradeHeavySplit reconciles the main group's processes against the listing: every listed test must have a result in exactly
// one process, since a partition that drops or doubles a test is the failure mode.
func gradeHeavySplit(listed int, missing, doubled []string) groupVerdict {
	line := fmt.Sprintf("      split reconciliation: listed=%d  missing=%d  in-two=%d", listed, len(missing), len(doubled))
	if len(missing) == 0 && len(doubled) == 0 {
		return groupVerdict{State: groupNone, Lines: []string{line}}
	}
	return groupVerdict{State: groupFail, Pre: []string{line}, Line: "heavy tier split does not reconcile — a test is in no process or in two",
		Notes: []string{fmt.Sprintf("main-tier split mismatch: missing %v, in two processes %v", missing, doubled)}}
}

// heavyNoDrain says whether a run narrowed by -heavy-group has no drain test inside its groups, and then what it prints in
// place of the drain process: a filter of no tests would match nothing, which is not a pass. An unnarrowed run always runs the
// drain process; an empty drain derivation there is gradeHeavyStart's failure.
func heavyNoDrain(run, drain []string) (groupVerdict, bool) {
	if len(run) == 0 || len(drain) > 0 {
		return groupVerdict{}, false
	}
	return groupVerdict{State: groupNone, Lines: []string{"      drain group: no drain test inside the narrowed groups, so no drain process runs"}}, true
}

// gradeHeavyDrain is the drain group's verdicts after its own process: its reconciliation, counted from the marker's own tokens
// both ways (every marked test SKIPPED in the main group and RAN in the drain group, so a derivation miss fails here rather than
// being absent from both halves), then the drain process's own pass or fail.
func gradeHeavyDrain(marked, mainSkipped, drainRan int, cr cellResult, out string) []groupVerdict {
	var vs []groupVerdict
	rec := groupVerdict{State: groupNone, Lines: []string{fmt.Sprintf("      reconciliation: marked=%d  skipped-in-main=%d  ran-in-drain=%d", marked, mainSkipped, drainRan)}}
	if mainSkipped != marked || drainRan != marked {
		rec = groupVerdict{State: groupFail, Pre: rec.Lines, Line: "heavy tier partition does not reconcile — a test is in neither half or in both",
			Notes: []string{fmt.Sprintf("partition mismatch: %d marked, %d skipped in main, %d ran in drain", marked, mainSkipped, drainRan)}}
	}
	vs = append(vs, rec)
	if cr.RC == 0 {
		return append(vs, groupVerdict{State: groupPass, Line: fmt.Sprintf("drain group (%d test(s), separate process)", marked)})
	}
	return append(vs, groupVerdict{State: groupFail, Line: fmt.Sprintf("drain group (%d test(s), separate process)", marked),
		Lines: append(detailLines(out, failTestRe), "      full output: "+cr.LogPath)})
}

// gradeHeavyIsoTest fails one fresh-process test whose process failed; it says nothing of one that passed.
func gradeHeavyIsoTest(name string, cr cellResult, out string) (groupVerdict, bool) {
	if cr.RC == 0 {
		return groupVerdict{}, false
	}
	return groupVerdict{State: groupFail, Line: name + " (fresh process)", Lines: append(detailLines(out, failTestRe), "      full output: "+cr.LogPath)}, true
}

// gradeHeavyIso reconciles the fresh-process tests the same way as the drain group (each marked test ISOLATED-SKIP in the main
// group and ISOLATED-RUN alone), and passes the set when it reconciles and none failed.
func gradeHeavyIso(marked, mainSkipped, ran, failed int) groupVerdict {
	line := fmt.Sprintf("      reconciliation: marked=%d  skipped-in-main=%d  ran-alone=%d", marked, mainSkipped, ran)
	switch {
	case mainSkipped != marked || ran != marked:
		return groupVerdict{State: groupFail, Pre: []string{line}, Line: "fresh-process partition does not reconcile — a test is in neither half or in both",
			Notes: []string{fmt.Sprintf("fresh-process mismatch: %d marked, %d skipped in main, %d ran alone", marked, mainSkipped, ran)}}
	case failed == 0 && marked > 0:
		return groupVerdict{State: groupPass, Pre: []string{line}, Line: fmt.Sprintf("fresh-process tests (%d, one process each)", marked)}
	}
	return groupVerdict{State: groupNone, Lines: []string{line}}
}

// gradeHeavyTier is the tier's verdict over its main-group processes, then its skip census by name: a tier whose value is
// "it runs the real models" is worth nothing if the real-model tests inside it skipped. A failure names the failing tests
// first, then their assertion lines, never a bare "file.go:N:" match, which under -v is every log line in the run.
func gradeHeavyTier(mainRC, secs int, mainOut string, runLines int, skipped [][]string) groupVerdict {
	v := groupVerdict{State: groupPass, Line: fmt.Sprintf("heavy tier (real models) — %ds", secs), Lines: evidenceLines(mainOut, okLineRe)}
	if mainRC != 0 {
		v = groupVerdict{State: groupFail, Line: fmt.Sprintf("heavy tier (real models) — %ds", secs), Lines: detailLines(mainOut, failTestRe)}
	}
	n := 0
	for _, s := range skipped {
		n += len(s)
	}
	v.Lines = append(v.Lines, fmt.Sprintf("      ran %d tests, skipped %d", runLines, n))
	if n > 0 {
		for _, s := range skipped {
			v.Lines = append(v.Lines, skipCensusLines(s, "        ")...)
		}
		v.Notes = append(v.Notes, fmt.Sprintf("heavy tier skipped %d test(s) — see the 2c census for which", n))
	}
	return v
}

// failTestRe picks a failing test's header and a panic, for a heavy process's detail.
var failTestRe = regexp.MustCompile(`^(---|    ---) FAIL|^panic:`)

// topTests returns up to n tests of a peak's ByTest, largest first.
func (p procPeak) topTests(n int) []string {
	names := make([]string, 0, len(p.ByTest))
	for k := range p.ByTest {
		names = append(names, k)
	}
	sort.Slice(names, func(i, j int) bool {
		if p.ByTest[names[i]] != p.ByTest[names[j]] {
			return p.ByTest[names[i]] > p.ByTest[names[j]]
		}
		return names[i] < names[j]
	})
	if len(names) > n {
		names = names[:n]
	}
	return names
}

package main

import (
	"fmt"
	"regexp"
	"strings"
)

// The gpu gate's Metal cell groups as a grader (SB12 step 8, docs/tasks/task-subsystems-2026-10.md): suite, parity,
// lifecycle and prefill are one `go test` cell each, and each group's state, line and detail or evidence lines are a
// function of its spec, whether the checkpoint it needs was present, and its cell's graded result and text. The live gate
// (gpuGate.metalCells) and `gate gpu -replay` (replayGPU) both print what gradeMetalCell returns.

// metalCellGroup is one Metal group that runs one cell.
type metalCellGroup struct {
	Group, Header string
	Cell          cell
	NeedsModel    bool // the cell needs the Metal checkpoint; when it is absent the group skips and runs no cell
	VacuousFails  bool // a cell whose tests all skipped fails the group
	OK, Bad       string
	Missing       string         // the skip line when the checkpoint is absent; %s is its path
	Detail        *regexp.Regexp // the lines detailLines picks on a failure
	Evidence      *regexp.Regexp // the lines evidenceLines prints on a pass; nil prints none
}

// metalCellGroups is the Metal cell groups in the order the gate runs them, for the checkpoint at model.
func metalCellGroups(model string) []metalCellGroup {
	return []metalCellGroup{
		{
			Group: "suite", Header: "2. Metal suite",
			Cell: cell{Name: "metal-suite", Pkgs: []string{"./metal/"}, Serial: true, Extra: []string{"-short"}},
			OK:   "full metal suite", Bad: "metal suite", Detail: failLineRe, Evidence: okLineRe,
		},
		{
			// Without goinfer_testhooks no resident-parity gate is compiled, so the suite above cannot see a forward move.
			// metalParityRun (parity.go) is named so this cell and TestMetalGateIsListedOrExplicitlyNotRequired read the same
			// string; a hand-copied pattern here is how gate-shaped tests go unmatched. The cell is filtered, and declares its
			// timeout, so a timeout is not mistaken for a crash.
			Group: "parity", Header: "2c. resident PARITY gates (-tags goinfer_testhooks — the forward is asserted here)",
			Cell: cell{Name: "metal-parity", Pkgs: []string{"./metal/"}, Tags: []string{"goinfer_testhooks"}, Run: metalParityRun,
				Serial: true, Timeout: "20m", Env: map[string]string{"GOINFER_HEAVY_TESTS": "1"}},
			VacuousFails: true,
			OK:           "resident parity gates (dense, gemma3, gemma4 MoE, qwen3.5, mellum, gpt-oss, paging bit-exactness)",
			Bad:          "resident parity gates — a Metal forward moved. This is the group the suite cannot see.",
			Detail:       failLineRe,
		},
		{
			// The cell sets GOINFER_HEAVY_TESTS itself: its tests call requireHeavyModel and the gate unsets the ambient value
			// (runGPU), so without it every test skips and the group would pass having run nothing.
			Group: "lifecycle", Header: "4. lifecycle",
			Cell: cell{Name: "metal-lifecycle", Pkgs: []string{"./metal/"},
				Run: "TestMetal_CloseFreesMemory|TestMetal_CloseWithSecondModelAlive",
				Env: map[string]string{"GOINFER_HEAVY_TESTS": "1"}},
			NeedsModel: true, VacuousFails: true,
			OK:       "Close() frees — sawtooth not staircase, and frees with a second model resident",
			Bad:      "Close() leaks memory",
			Missing:  "Close() lifecycle gate needs %s",
			Detail:   regexp.MustCompile(`^--- FAIL|LEAK|did NOT free|USE-AFTER-FREE|\.go:[0-9]+:`),
			Evidence: regexp.MustCompile(`trajectory|A\+B alive`),
		},
		{
			Group: "prefill", Header: "4b. prefill (f16-MMA TTFT — a shipped path, and it shipped NaN)",
			Cell: cell{Name: "metal-prefill", Pkgs: []string{"./metal/"}, Run: "TestPrefillParity|TestPrefillNoNaN",
				Env: map[string]string{"GOINFER_METAL_MODEL": model, "GOINFER_HEAVY_TESTS": "1"}},
			NeedsModel: true, VacuousFails: true,
			OK: "prefill matches sequential decode and emits finite logits (no NaN)",
			Bad: "prefill parity/NaN gate — the f16-MMA TTFT path is wrong on a shipped model, " +
				"or every test in it skipped and verified nothing",
			Missing:  "prefill gate needs %s",
			Detail:   regexp.MustCompile(`^--- FAIL|parity FAIL|contain NaN|\.go:[0-9]+:`),
			Evidence: regexp.MustCompile(`argmax matches|faster TTFT`),
		},
	}
}

// groupState is a group's outcome.
type groupState int

const (
	groupPass groupState = iota
	groupFail
	groupSkip
)

// groupVerdict is what one group prints: its state and line, then Lines verbatim (detail or evidence, indented). Notes are
// added to the gate's notes after it prints. Ran counts a group that did not pass toward the gate's ran count (a pass always
// counts).
type groupVerdict struct {
	State groupState
	Line  string
	Lines []string
	Notes []string
	Ran   bool
}

// gradeMetalCell grades a Metal cell group. A group that needs the checkpoint skips when it was absent (and then ran no
// cell, so cr and out are not read). Otherwise it fails on a non-zero exit, or on a cell whose tests all skipped when
// VacuousFails (`go test` exits 0 on an all-skip package), with detailLines; and passes with evidenceLines.
func gradeMetalCell(g metalCellGroup, model string, modelPresent bool, cr cellResult, out string) groupVerdict {
	if g.NeedsModel && !modelPresent {
		return groupVerdict{State: groupSkip, Line: fmt.Sprintf(g.Missing, model)}
	}
	if cr.RC != 0 || (g.VacuousFails && cr.Pass == 0 && cr.Fail == 0 && cr.Skip > 0) {
		return groupVerdict{State: groupFail, Line: g.Bad, Lines: detailLines(out, g.Detail)}
	}
	v := groupVerdict{State: groupPass, Line: g.OK}
	if g.Evidence != nil {
		v.Lines = evidenceLines(out, g.Evidence)
	}
	return v
}

// detailLines is the matching assertion lines, or the RAW TAIL if nothing matched, so a failing group
// never explains nothing: `go test` killed by a signal, an OOM or a timeout emits neither "--- FAIL" nor a
// "file.go:N:" line, only "FAIL <pkg> <secs>".
func detailLines(out string, re *regexp.Regexp) []string {
	// A process death is reported FIRST and on its own terms. It has no
	// `--- FAIL` to match, and its stack head names the test that died — which a
	// tail-based excerpt would bury under a register dump.
	if ex := crashExcerpt(out); len(ex) > 0 {
		lines := []string{"      PROCESS DIED — no test-level failure was reported. Crash head:"}
		for _, ln := range ex {
			lines = append(lines, "      | "+strings.TrimRight(ln, "\r"))
		}
		return lines
	}
	var hits []string
	if re.String() == failLineRe.String() {
		hits = failureLines(out)
	} else {
		for ln := range strings.SplitSeq(out, "\n") {
			if re.MatchString(ln) {
				hits = append(hits, ln)
			}
		}
	}
	if len(hits) > 0 {
		var lines []string
		for i, h := range hits {
			if i >= 12 {
				break
			}
			lines = append(lines, "      "+strings.TrimRight(h, "\r"))
		}
		return lines
	}
	lines := []string{"      NO ASSERTION LINE MATCHED — this run failed without a test-level failure.",
		"      That is a signal, an OOM kill, or a timeout. Raw tail (last 15 lines):"}
	tail := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(tail) > 15 {
		tail = tail[len(tail)-15:]
	}
	for _, ln := range tail {
		lines = append(lines, "      | "+ln)
	}
	return lines
}

// evidenceLines is the lines a passing group offered as its own proof: the `ok <pkg> <secs>` line, a
// lifecycle trajectory, a prefill's argmax comparison. They are part of the scope line, not decoration:
// "PASS metal suite" says a verdict, `ok github.com/…/metal 31.2s` says what ran and for how long.
func evidenceLines(out string, re *regexp.Regexp) []string {
	var lines []string
	for ln := range strings.SplitSeq(out, "\n") {
		if re.MatchString(ln) {
			lines = append(lines, "      "+strings.TrimRight(ln, "\r"))
		}
	}
	return lines
}

var okLineRe = regexp.MustCompile(`^ok\s`)

// failLineRe matches a test-level failure header. It deliberately does NOT match bare `file.go:N:` lines:
// every passing t.Log emits one, and that breadth defeats detail()'s crash fallback, which fires only when
// nothing matches. Assertion lines are still shown, but only the ones below a failing test (failureLines).
var failLineRe = regexp.MustCompile(`^(---|    ---) FAIL`)

// assertionLineRe is a test's own `file.go:N: …` output. Shown only while inside
// a failing test.
var assertionLineRe = regexp.MustCompile(`^\s*[\w./-]+\.go:[0-9]+:`)

// testBoundaryRe ends a failing test's block: anything that starts a new test or
// closes the package.
var testBoundaryRe = regexp.MustCompile(`^(=== RUN|=== CONT|=== PAUSE|(---|    ---) (PASS|SKIP)|ok\s|PASS$|FAIL\s)`)

// crashHeadRe is a process death: a signal, a panic, or a runtime fatal. These
// carry no `--- FAIL`, which is exactly why detail() must recognize them.
var crashHeadRe = regexp.MustCompile(`^(SIGSEGV|SIGBUS|SIGABRT|SIGILL|panic:|fatal error:)|signal arrived during`)

// ourFrameRe matches a stack frame in code this repo owns or vendors — the frames
// that identify WHICH test or function died, as opposed to the runtime and FFI
// scaffolding above them.
var ourFrameRe = regexp.MustCompile(`townsendmerino/(goinfer|aikit)|_test\.go:[0-9]+`)

// registerDumpRe is the tail of a Go crash report — `r14  0x8000…`, `pc 0x…`.
// Machine state, useless for identifying WHICH test died, and long enough to push
// the part that matters off the end of a fixed-size excerpt.
var registerDumpRe = regexp.MustCompile(`^(r[0-9]+|x[0-9]+|sp|pc|lr|fp|fault)\s+0x`)

// failureLines picks the lines that explain a failure: every test-level FAIL
// header, the assertion lines belonging to it, and nothing from tests that
// passed.
func failureLines(out string) []string {
	var hits []string
	inFail := false
	for ln := range strings.SplitSeq(out, "\n") {
		switch {
		case failLineRe.MatchString(ln):
			inFail = true
			hits = append(hits, ln)
		case testBoundaryRe.MatchString(ln):
			inFail = false
		case inFail && assertionLineRe.MatchString(ln):
			hits = append(hits, ln)
		}
	}
	return hits
}

// crashExcerpt returns the head of a process-death report: the signal line and
// the frames just below it, which name the test that died. It stops at the
// register dump — for the Metal `fault 0x10` tail the last 15 lines are
// registers, so a tail-based excerpt shows machine state and hides the answer.
func crashExcerpt(out string) []string {
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	start := -1
	for i, ln := range lines {
		if crashHeadRe.MatchString(ln) {
			start = i
			break
		}
	}
	if start < 0 {
		return nil
	}
	// The head (signal, address, goroutine) plus the first frames belonging to OUR code: a fixed prefix does
	// not reach them (runtime/purego/reflect frames sit above the goinfer frame), and the test name is the one
	// fact worth printing.
	var head, ours []string
	for _, ln := range lines[start:] {
		if registerDumpRe.MatchString(ln) {
			break
		}
		if len(head) < 6 {
			head = append(head, ln)
			continue
		}
		if ourFrameRe.MatchString(ln) && len(ours) < 8 {
			ours = append(ours, strings.TrimRight(ln, "\r"))
		}
	}
	if len(ours) == 0 {
		return head
	}
	return append(append(head, "      ...(frames from this repo)..."), ours...)
}

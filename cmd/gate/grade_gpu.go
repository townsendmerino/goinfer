package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

// gpuTally is what the gpu gate's check groups leave for its verdict: the groups declared and the ones that reported,
// the verdict counts within them, the filtered cells whose tests all skipped or matched nothing, and the skip notes.
type gpuTally struct {
	Expect                []string
	Emitted               map[string]bool
	Pass, Skip, Fail, Ran int
	Vacuous, Empty, Notes []string
}

// gpuVerdictInputs is what the verdict prints beyond the tally; the runner gathers each before grading.
type gpuVerdictInputs struct {
	Backend    string
	Commit     string
	Host       string   // uname -s
	Dirty      bool     // the working tree had uncommitted changes
	DirtyLines []string // the first lines of git status --porcelain, when Dirty
	Now        string   // the UTC time a PASS is stamped with
}

// gradeGPU is the gpu gate's verdict: it reconciles the reported groups against the declared ones (a group missing or
// undeclared is a failure, added to t.Fail), prints the verdict block, and returns the exit code and the record's label
// (NO_GATE, VACUOUS, EMPTY, FAIL, INCONCLUSIVE or PASS, decided in that order).
func gradeGPU(w io.Writer, t *gpuTally, in gpuVerdictInputs) (int, string) {
	declared := map[string]bool{}
	for _, e := range t.Expect {
		declared[e] = true
	}
	var missing, unexpected []string
	for _, e := range t.Expect {
		if !t.Emitted[e] {
			missing = append(missing, e)
		}
	}
	for _, e := range sortedSet(t.Emitted) {
		if !declared[e] {
			unexpected = append(unexpected, e)
		}
	}
	bad := func(format string, a ...any) {
		fmt.Fprintf(w, "  %sFAIL%s  %s\n", red, off, fmt.Sprintf(format, a...))
		t.Fail++
	}
	if len(missing) > 0 {
		bad("check group(s) declared but emitted NO verdict: %s — the gate tested less than it reports (audit G-01)",
			strings.Join(missing, " "))
	}
	if len(unexpected) > 0 {
		bad("check group(s) emitted but not declared: %s — update the declared set so the tally stays meaningful",
			strings.Join(unexpected, " "))
	}

	fmt.Fprintf(w, "\n%s== %s ==%s\n", bold, "verdict", off)
	// ONE UNIT: check groups, so a reader deciding whether to ship sees at a glance whether one is missing.
	fmt.Fprintf(w, "  check groups: %d declared -> %d reported   |   verdicts within them: %d pass, %d skip, %d fail\n",
		len(t.Expect), len(t.Emitted), t.Pass, t.Skip, t.Fail)
	// The release record turns on this distinction: "the suite passed" is NOT "the forward is
	// gated". Say which of the two actually happened, by name, so neither can be read as the other.
	if in.Backend == "cuda" {
		s2a, s2b := "not run", "not run"
		if t.Emitted["suite"] {
			s2a = "reported"
		}
		if t.Emitted["parity"] {
			s2b = "reported"
		}
		fmt.Fprintf(w, "  of which: kernel-level suite = %s   |   resident PARITY gates (forward asserted) = %s\n", s2a, s2b)
	}
	if len(t.Expect) != len(t.Emitted) {
		fmt.Fprintf(w, "  (declared != reported: %d group(s) produced no verdict — see the FAIL above)\n",
			len(t.Expect)-len(t.Emitted))
	}
	if t.Skip > 0 {
		fmt.Fprintf(w, "\n  %sSkipped — a skip is not a pass; this gate does NOT cover:%s\n", amber, off)
		for _, n := range t.Notes {
			fmt.Fprintf(w, "    - %s\n", n)
		}
	}
	if t.Ran == 0 {
		fmt.Fprintf(w, "\n  %sNO GATE%s — nothing actually ran. Do not read this as a pass.\n", red, off)
		return 1, "NO_GATE"
	}
	// A FILTERED CELL WHOSE TESTS ALL SKIPPED IS THE SAME CLASS AS AN EMPTY ONE: coverage the verdict vouches
	// for did not run. It is separate from Empty because the tests exist and were selected, they all
	// opted out, and `go test` exits 0 on an all-skip package, so nothing upstream notices.
	if len(t.Vacuous) > 0 {
		fmt.Fprintf(w, "\n  %sVACUOUS CELL(S)%s — every test skipped, so the PASS above vouches for nothing:\n", red, off)
		for _, c := range t.Vacuous {
			fmt.Fprintf(w, "    - %s\n", c)
		}
		fmt.Fprintf(w, "    A skip is not a pass. Usually a missing asset, or an opt-in env var\n"+
			"    the cell must set for itself because the gate deliberately unsets ambient ones.\n")
		return 1, "VACUOUS"
	}
	if len(t.Empty) > 0 {
		fmt.Fprintf(w, "\n  %sEMPTY CELL(S)%s — a -run pattern matched no test, so that coverage is gone:\n", red, off)
		for _, c := range t.Empty {
			fmt.Fprintf(w, "    - %s\n", c)
		}
		fmt.Fprintf(w, "    Nothing FAILED; the tests were not there to run. Fix the pattern or the name.\n")
		return 1, "EMPTY"
	}
	if t.Fail > 0 {
		d := ""
		if in.Dirty {
			d = " +dirty"
		}
		fmt.Fprintf(w, "\n  %sFAIL%s — %s on %s @ %s%s. Do not tag.\n", red, off, in.Backend, in.Host, in.Commit, d)
		return 1, "FAIL"
	}
	// THREE STATES, NOT TWO. Every check is green here, but a dirty tree is a failure of PROVENANCE, not of
	// the checks: the verdict names a commit, and an uncommitted edit means it does not describe what that
	// commit contains. Collapsing the two loses the distinction a reader needs (is the CODE broken, or the
	// EVIDENCE?). Verdicts get pasted into tag messages, which is what this gate is for.
	if in.Dirty {
		fmt.Fprintf(w, "\n  %sINCONCLUSIVE%s — %d/%d groups green, but the working tree is DIRTY.\n",
			amber, off, len(t.Expect), len(t.Expect))
		fmt.Fprintf(w, "  The verdict names %s and the tree is not %s. Commit, then re-run before tagging.\n", in.Commit, in.Commit)
		for _, ln := range in.DirtyLines {
			fmt.Fprintf(w, "    %s\n", ln)
		}
		return 1, "INCONCLUSIVE"
	}
	fmt.Fprintf(w, "\n  %sPASS%s — %s on %s @ %s (%s)\n", green, off, in.Backend, in.Host, in.Commit, in.Now)
	fmt.Fprintf(w, "  Paste this block for the tag. The OTHER box must pass its own run: no machine has both GPUs.\n")
	return 0, "PASS"
}
func sortedSet(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

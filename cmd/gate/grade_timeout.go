package main

import (
	"regexp"
	"strings"
)

// A cell that hits go test's -timeout exits non-zero with zero --- FAIL lines, which the sweep reports as "crash,
// timeout or build failure". The v0.21.0 sweep's realckpt cell was exactly that (120m, TestQwen3MoeReal_oracle in
// flight, four gates never reached), and the verdict read like a failure to investigate rather than a budget to raise.
// The panic text names the budget and the tests that were running, so the verdict can say so.
var (
	timeoutPanicRE = regexp.MustCompile(`panic: test timed out after (\S+)`)
	runningTestRE  = regexp.MustCompile(`^\s+(Test\S+) \(`)
)

// timeoutPanic reads go test output for the -timeout panic. after is the budget it fired at ("2h0m0s") and running
// the top-level tests the panic listed as in flight; after is "" when the output holds no such panic.
func timeoutPanic(blob string) (after string, running []string) {
	m := timeoutPanicRE.FindStringSubmatch(blob)
	if m == nil {
		return "", nil
	}
	after = m[1]
	inList := false
	for l := range strings.SplitSeq(blob[strings.Index(blob, m[0]):], "\n") {
		switch {
		case strings.Contains(l, "running tests:"):
			inList = true
		case inList && runningTestRE.MatchString(l):
			running = append(running, runningTestRE.FindStringSubmatch(l)[1])
		case inList && strings.TrimSpace(l) == "":
			return after, running
		}
	}
	return after, running
}

// timeoutPanic looks for that panic in everything the cells printed: package-level output, and a test's own output
// (go test -json attributes the panic to either, depending on where it was printed).
func (r *results) timeoutPanic() (after string, running []string) {
	for _, lines := range r.pkgOut {
		if a, run := timeoutPanic(strings.Join(lines, "")); a != "" {
			return a, run
		}
	}
	for _, lines := range r.out {
		if a, run := timeoutPanic(strings.Join(lines, "")); a != "" {
			return a, run
		}
	}
	return "", nil
}

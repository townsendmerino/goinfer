package gatecore

import (
	"regexp"
	"strings"
)

var (
	timeoutPanicRE = regexp.MustCompile(`panic: test timed out after (\S+)`)
	runningTestRE  = regexp.MustCompile(`^\s+(Test\S+) \(`)
)

// TimeoutPanic extracts the timeout duration and running test names from `go test` timeout panic output.
func TimeoutPanic(blob string) (after string, running []string) {
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

// TimeoutPanic scans package-level and per-test output in r for a `go test` timeout panic.
func (r *Results) TimeoutPanic() (after string, running []string) {
	for _, lines := range r.PkgOut {
		if a, run := TimeoutPanic(strings.Join(lines, "")); a != "" {
			return a, run
		}
	}
	for _, lines := range r.Out {
		if a, run := TimeoutPanic(strings.Join(lines, "")); a != "" {
			return a, run
		}
	}
	return "", nil
}

package decoder

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// Startup self-tests (docs/tasks/task-hardware-coverage-2026-10.md, H2). goinfer is built and measured on two machines, and a wrong kernel on hardware nobody here owns does not crash: it
// returns slightly wrong numbers. A self-test runs the kernels this process is about to use against a reference on a small fixed input, judged by the agreement the parity suite already
// certified for them, and on a mismatch DECLINES (names the kernel and the numbers, falls back to the next path) instead of continuing silently. This file is the common record every backend
// reports into, and the CPU test: aikit's linalg.SelfCheck, run once per process before the first model load.

// Self-test outcomes.
const (
	SelfTestPass     = "pass"     // every check agreed with its reference
	SelfTestRepaired = "repaired" // a check disagreed and the process stepped down to the next kernel tier, which agrees (CPU)
	SelfTestDeclined = "declined" // a check disagreed and the backend is not used (GPU backends)
	SelfTestSkipped  = "skipped"  // not run: disabled by the operator, or not applicable
)

// SelfTestResult is one backend's startup self-test.
type SelfTestResult struct {
	Backend    string
	Status     string
	Mismatches []string      // each disagreement, as kernel: observed against allowed (detail)
	Disabled   []string      // the kernel tiers stepped off, in order (CPU)
	Elapsed    time.Duration // the check's own cost
	Note       string        // why nothing was checked, for a skipped result
}

// OK reports a result that left the process on kernels that agree with their references.
func (r SelfTestResult) OK() bool {
	return r.Status == SelfTestPass || r.Status == SelfTestRepaired || r.Status == SelfTestSkipped
}

// Summary is the one line a log or a report shows.
func (r SelfTestResult) Summary() string {
	s := fmt.Sprintf("%s %s (%.1f ms)", r.Backend, r.Status, float64(r.Elapsed.Microseconds())/1000)
	if len(r.Disabled) > 0 {
		s += ", stepped down: " + strings.Join(r.Disabled, ", ")
	}
	if r.Note != "" {
		s += ", " + r.Note
	}
	for _, m := range r.Mismatches {
		s += "; " + m
	}
	return s
}

var (
	selfTestMu      sync.Mutex
	selfTestResults = map[string]SelfTestResult{}
	selfTestSkipped atomic.Bool
	cpuSelfTestOnce sync.Once
	// selfTestWarn is where a failing self-test announces itself; a variable so a test can capture it.
	selfTestWarn = func(msg string) { fmt.Fprintln(os.Stderr, msg) }
)

// RecordSelfTest stores a backend's result and, if it is not a clean pass, prints one WARN line naming what mismatched. Backends call it from their own startup self-test.
func RecordSelfTest(r SelfTestResult) {
	selfTestMu.Lock()
	selfTestResults[r.Backend] = r
	selfTestMu.Unlock()
	if r.Status == SelfTestRepaired || r.Status == SelfTestDeclined {
		selfTestWarn("WARN: self-test: " + r.Summary())
	}
}

// SelfTestResults returns every recorded result, sorted by backend name.
func SelfTestResults() []SelfTestResult {
	selfTestMu.Lock()
	defer selfTestMu.Unlock()
	out := make([]SelfTestResult, 0, len(selfTestResults))
	for _, r := range selfTestResults {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Backend < out[j].Backend })
	return out
}

// SkipSelfTests turns the startup self-tests off for this process (serve's and chat's -no-selftest). Call it before the first Load. A skipped test is recorded as skipped, so a report
// says so.
func SkipSelfTests() { selfTestSkipped.Store(true) }

// SelfTestsSkipped reports whether SkipSelfTests was called, for a backend deciding whether to run its own.
func SelfTestsSkipped() bool { return selfTestSkipped.Load() }

// ensureCPUSelfTest runs aikit's linalg.SelfCheck once per process, with repair: a CPU kernel tier whose result disagrees with its reference is turned off and the next tier checked, so a bad
// ISA path degrades instead of producing wrong numbers. It must run before concurrent use of the kernels (the dispatch flags are plain variables), which is why it is the first thing Load does:
// the process's first Load has no other model running.
func ensureCPUSelfTest() { cpuSelfTestOnce.Do(runCPUSelfTest) }

// runCPUSelfTest is the once-guarded body, separate so a test can run it again without copying the sync.Once.
func runCPUSelfTest() {
	if selfTestSkipped.Load() {
		RecordSelfTest(SelfTestResult{Backend: "cpu", Status: SelfTestSkipped})
		return
	}
	RecordSelfTest(cpuResult(linalg.SelfCheck(true)))
}

// cpuResult maps a linalg.SelfCheckReport to the record. A clean first pass is pass. A first-pass mismatch that a stepped-down tier cured is repaired. Mismatches that remain after the last
// tier is off are declined: the scalar kernels themselves disagree with their references, so there is nothing left to fall back to.
func cpuResult(rep linalg.SelfCheckReport) SelfTestResult {
	res := SelfTestResult{Backend: "cpu", Status: SelfTestPass, Disabled: rep.Disabled, Elapsed: rep.Elapsed}
	list := rep.Mismatches
	switch {
	case len(rep.Remaining) > 0:
		res.Status = SelfTestDeclined
		list = rep.Remaining
	case !rep.OK():
		res.Status = SelfTestRepaired
	}
	for _, m := range list {
		res.Mismatches = append(res.Mismatches, m.String())
	}
	return res
}

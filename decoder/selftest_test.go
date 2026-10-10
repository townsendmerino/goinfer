package decoder

import (
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// The CPU self-test runs from Load, once per process, and records a result a report can show. On a healthy host it passes, and it says how long it took.
func TestEnsureCPUSelfTest_recordsAPass(t *testing.T) {
	ensureCPUSelfTest()
	var got *SelfTestResult
	for _, r := range SelfTestResults() {
		if r.Backend == "cpu" {
			r := r
			got = &r
		}
	}
	if got == nil {
		t.Fatal("no cpu self-test recorded after ensureCPUSelfTest")
	}
	if got.Status != SelfTestPass && got.Status != SelfTestSkipped {
		t.Fatalf("a healthy host's cpu self-test is %s: %s", got.Status, got.Summary())
	}
	// Windows' monotonic clock ticks every ~0.5-15 ms and the CPU self-test takes about 1 ms, so a pass there can honestly
	// record 0.
	if got.Status == SelfTestPass && got.Elapsed <= 0 && runtime.GOOS != "windows" {
		t.Errorf("a pass records its cost: elapsed %v", got.Elapsed)
	}
}

// The mapping from aikit's report to the record: pass, repaired, declined. A clean first pass is pass, a cured mismatch is repaired, and anything still wrong at the end is declined.
func TestCPUResult_statuses(t *testing.T) {
	mm := func(k string) linalg.Mismatch {
		return linalg.Mismatch{Kernel: k, Observed: 1, Allowed: 0, Detail: "d"}
	}
	for _, c := range []struct {
		name string
		rep  linalg.SelfCheckReport
		want string
		list int
	}{
		{"clean", linalg.SelfCheckReport{Elapsed: time.Millisecond}, SelfTestPass, 0},
		{"cured by stepping down", linalg.SelfCheckReport{Mismatches: []linalg.Mismatch{mm("dotI8")}, Disabled: []string{"avx2"}}, SelfTestRepaired, 1},
		{"nothing left to step down to", linalg.SelfCheckReport{Mismatches: []linalg.Mismatch{mm("dotI8")}, Remaining: []linalg.Mismatch{mm("dotI8"), mm("quantizeRowInt8")}, Disabled: []string{"avx512vnni", "avx2"}}, SelfTestDeclined, 2},
	} {
		got := cpuResult(c.rep)
		if got.Status != c.want || len(got.Mismatches) != c.list {
			t.Errorf("%s: status %q with %d mismatches, want %q with %d", c.name, got.Status, len(got.Mismatches), c.want, c.list)
		}
		if got.OK() != (c.want != SelfTestDeclined) {
			t.Errorf("%s: OK() = %v for %s", c.name, got.OK(), got.Status)
		}
	}
}

// A failing result announces itself on the warning channel once, naming what mismatched and what was stepped off; a pass is silent.
func TestRecordSelfTest_warnsOnFailureOnly(t *testing.T) {
	var msgs []string
	var mu sync.Mutex
	prev := selfTestWarn
	selfTestWarn = func(m string) { mu.Lock(); msgs = append(msgs, m); mu.Unlock() }
	t.Cleanup(func() { selfTestWarn = prev })
	RecordSelfTest(SelfTestResult{Backend: "testbackend-pass", Status: SelfTestPass})
	if len(msgs) != 0 {
		t.Fatalf("a pass warned: %v", msgs)
	}
	RecordSelfTest(SelfTestResult{Backend: "testbackend-bad", Status: SelfTestRepaired, Disabled: []string{"avx2"}, Mismatches: []string{"dotI8: observed 1, allowed 0 (n=65)"}})
	if len(msgs) != 1 || !strings.Contains(msgs[0], "WARN: self-test") || !strings.Contains(msgs[0], "dotI8") || !strings.Contains(msgs[0], "avx2") {
		t.Fatalf("the warning must name the kernel and the tier stepped off: %v", msgs)
	}
}

// -no-selftest: a skipped test is recorded as skipped (a report says so) and does not run the check.
func TestEnsureCPUSelfTest_skipIsRecorded(t *testing.T) {
	t.Cleanup(func() { selfTestSkipped.Store(false); runCPUSelfTest() }) // leave the process with the real result recorded
	SkipSelfTests()
	runCPUSelfTest()
	for _, r := range SelfTestResults() {
		if r.Backend == "cpu" && r.Status == SelfTestSkipped {
			return
		}
	}
	t.Fatalf("a skipped self-test must be recorded as skipped: %v", SelfTestResults())
}

func TestRegisteredBackendsAndHardwareInfo(t *testing.T) {
	if names := RegisteredBackends(); len(names) == 0 || names[0] == "" {
		t.Fatalf("cpu is always registered: %v", names)
	}
	RegisterHardwareInfo("testbackend-info", func() []string { return []string{"device: a", "driver: b"} })
	t.Cleanup(func() { hwInfoMu.Lock(); delete(hwInfo, "testbackend-info"); hwInfoMu.Unlock() })
	if got := HardwareInfo()["testbackend-info"]; len(got) != 2 || got[0] != "device: a" {
		t.Fatalf("registered info not returned: %v", got)
	}
}

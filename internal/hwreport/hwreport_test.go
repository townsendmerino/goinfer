package hwreport

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func report(t *testing.T) string {
	t.Helper()
	var b bytes.Buffer
	Write(&b, Options{})
	return b.String()
}

// The block carries every section a hardware bug report needs.
func TestWrite_hasEverySection(t *testing.T) {
	decoder.RegisterHardwareInfo("cpu", func() []string { return []string{"test device line"} })
	out := report(t)
	for _, want := range []string{"goinfer hardware report", "\ngoinfer: ", "\nos: ", "\ncpu: ", "\ncpu kernels (", "detected ", "in use ", "forced by build tags", "\nmemory: ",
		"backends linked into this binary: ", "  cpu: test device line", "self-tests", "  cpu "} {
		if !strings.Contains(out, want) {
			t.Errorf("the report lacks %q:\n%s", want, out)
		}
	}
	// The cpu self-test is run for the report and its outcome is stated, never omitted.
	if !strings.Contains(out, "cpu pass") && !strings.Contains(out, "cpu skipped") && !strings.Contains(out, "cpu repaired") && !strings.Contains(out, "cpu declined") {
		t.Errorf("the report does not state the cpu self-test's outcome:\n%s", out)
	}
}

// A backend's own self-test is run for the report and shown, including a failure.
func TestWrite_showsBackendSelfTests(t *testing.T) {
	decoder.RegisterSelfTest("reporttest-gpu", func() decoder.SelfTestResult {
		return decoder.SelfTestResult{Backend: "reporttest-gpu", Status: decoder.SelfTestDeclined, Mismatches: []string{"attention: observed 0.37, allowed 0.001 (past 256 keys)"}}
	})
	out := report(t)
	if !strings.Contains(out, "reporttest-gpu declined") || !strings.Contains(out, "attention: observed 0.37") {
		t.Errorf("a declined backend self-test must appear with its mismatch:\n%s", out)
	}
}

// Privacy: the block names no person and no machine: no hostname, no user name, no home directory, and a model by file name only.
func TestWrite_leaksNothingIdentifying(t *testing.T) {
	out := report(t)
	host, _ := os.Hostname()
	home, _ := os.UserHomeDir()
	for name, v := range map[string]string{"hostname": host, "user": os.Getenv("USER"), "home directory": home} {
		if len(v) >= 3 && strings.Contains(out, v) {
			t.Errorf("the report contains the %s %q:\n%s", name, v, out)
		}
	}
}

// `check` without --hardware is a usage error, not a silent success.
func TestRun_requiresHardware(t *testing.T) {
	if code := Run(nil, "goinfer-chat"); code != 2 {
		t.Errorf("check with no --hardware exited %d, want 2", code)
	}
}

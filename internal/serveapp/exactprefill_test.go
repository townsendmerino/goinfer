package serveapp

import (
	"github.com/townsendmerino/goinfer/internal/loadflags"
	"os"
	"testing"
)

// unsetenvT unsets name for the duration of the test, restoring whatever value (or absence) it
// had beforehand once the test finishes.
func unsetenvT(t *testing.T, name string) {
	t.Helper()
	if v, ok := os.LookupEnv(name); ok {
		t.Cleanup(func() { os.Setenv(name, v) })
	} else {
		t.Cleanup(func() { os.Unsetenv(name) })
	}
	os.Unsetenv(name)
}

// The prompt-ingestion flags reach each model through its decoder.Options, never the process environment
// (phase 5, docs/tasks/task-env-config-2026-09.md — they used to be applied by applyExactPrefillEnv's
// os.Setenv, process-wide). M-48 still holds: --exact-prefill covers all three backends, now because
// Options.ExactPrefill is consulted by each (CPU, CUDA resident build, Metal resident) per model.
func TestPrefillFlags_reachTheDecoderThroughOptions(t *testing.T) {
	for _, v := range []string{"GOINFER_METAL_FAST_PREFILL", "GOINFER_CUDA_FAST_PREFILL", "GOINFER_CPU_FAST_ATTENTION"} {
		unsetenvT(t, v)
	}
	for _, c := range []struct {
		name    string
		cfg     config
		exact   bool
		cpuFast string
	}{
		{"default", config{}, false, "1"},
		{"--exact-prefill", config{load: loadflags.Flags{ExactPrefill: true}}, true, "0"},
		{"--cpu-exact-prefill", config{load: loadflags.Flags{CPUExactPrefill: true}}, false, "0"},
	} {
		opts := (modelSpec{}).options(c.cfg)
		if opts.ExactPrefill != c.exact {
			t.Errorf("%s: Options.ExactPrefill = %v, want %v", c.name, opts.ExactPrefill, c.exact)
		}
		if opts.Knobs == nil || (*opts.Knobs)["GOINFER_CPU_FAST_ATTENTION"] != c.cpuFast {
			t.Errorf("%s: Options.Knobs GOINFER_CPU_FAST_ATTENTION = %v, want %q — set explicitly either way, so the flags beat an inherited env var",
				c.name, opts.Knobs, c.cpuFast)
		}
	}
	for _, v := range []string{"GOINFER_METAL_FAST_PREFILL", "GOINFER_CUDA_FAST_PREFILL", "GOINFER_CPU_FAST_ATTENTION"} {
		if got, set := os.LookupEnv(v); set {
			t.Errorf("building Options wrote %s=%q to the process environment", v, got)
		}
	}
}

// TestMoEPagerFlag_reachesTheDecoderThroughOptions: --moe-pager is carried to decoder.Load in
// Options.MoEPager (it used to be applied by setting GOINFER_MOE_PREAD_CPU, process-wide).
func TestMoEPagerFlag_reachesTheDecoderThroughOptions(t *testing.T) {
	unsetenvT(t, "GOINFER_MOE_PREAD_CPU")
	for _, mode := range []string{"pool", "mmap"} {
		if got := (modelSpec{}).options(config{load: loadflags.Flags{MoEPager: mode}}).MoEPager; got != mode {
			t.Errorf("--moe-pager=%s: Options.MoEPager = %q", mode, got)
		}
	}
	if _, set := os.LookupEnv("GOINFER_MOE_PREAD_CPU"); set {
		t.Error("building Options wrote GOINFER_MOE_PREAD_CPU")
	}
}

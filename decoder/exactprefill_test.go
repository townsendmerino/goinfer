package decoder

import (
	"os"
	"testing"
)

// unsetenvT unsets name for the duration of the test, restoring whatever value (or absence) it
// had beforehand once the test finishes — same helper internal/serveapp/exactprefill_test.go
// uses for the sibling env vars this Options field now also sets.
func unsetenvT(t *testing.T, name string) {
	t.Helper()
	if v, ok := os.LookupEnv(name); ok {
		t.Cleanup(func() { os.Setenv(name, v) })
	} else {
		t.Cleanup(func() { os.Unsetenv(name) })
	}
	os.Unsetenv(name)
}

// TestLoad_exactPrefillOptionSetsAllThreeBackendEnvVars is M-26's chokepoint gate
// (docs/audit-2026-09-10.md): decoder.Options.ExactPrefill is the library-level field
// docs/completed/task-prefill-gap.md already documented as existing — this proves Load actually
// sets the three fast-prefill env vars from it, the same three internal/serveapp's own
// applyExactPrefillEnv sets from --exact-prefill (that CLI-level helper is untouched by this fix;
// this is the new chokepoint chatapp/gemmaapp's own --exact-prefill flags now reach instead of
// duplicating serve's CLI-only logic).
func TestLoad_exactPrefillOptionSetsAllThreeBackendEnvVars(t *testing.T) {
	unsetenvT(t, "GOINFER_METAL_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CUDA_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CPU_FAST_ATTENTION")

	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32", ExactPrefill: true})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	for _, want := range []struct{ name, value string }{
		{"GOINFER_METAL_FAST_PREFILL", "0"},
		{"GOINFER_CUDA_FAST_PREFILL", "0"},
		{"GOINFER_CPU_FAST_ATTENTION", "0"},
	} {
		if got := os.Getenv(want.name); got != want.value {
			t.Errorf("Options.ExactPrefill=true: %s = %q, want %q", want.name, got, want.value)
		}
	}
}

// TestLoad_exactPrefillFalseLeavesEnvUntouched confirms the negative: Options.ExactPrefill's
// zero value must not touch these env vars at all (not even to reset them to "1"/unset), unlike
// serve's own applyExactPrefillEnv which explicitly reasserts GOINFER_CPU_FAST_ATTENTION either
// way — Load is called by every library consumer, including ones managing these same env vars
// themselves (serve calls applyExactPrefillEnv once at startup, independently of any particular
// Load call), so a false value must be a true no-op, not "reset to default."
func TestLoad_exactPrefillFalseLeavesEnvUntouched(t *testing.T) {
	unsetenvT(t, "GOINFER_METAL_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CUDA_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CPU_FAST_ATTENTION")
	// Pre-set a sentinel value a real caller (e.g. serve's own granular flags) might have chosen —
	// Load with ExactPrefill=false must leave it exactly alone.
	os.Setenv("GOINFER_CPU_FAST_ATTENTION", "1")

	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	if v, ok := os.LookupEnv("GOINFER_METAL_FAST_PREFILL"); ok {
		t.Errorf("GOINFER_METAL_FAST_PREFILL = %q, want unset", v)
	}
	if v, ok := os.LookupEnv("GOINFER_CUDA_FAST_PREFILL"); ok {
		t.Errorf("GOINFER_CUDA_FAST_PREFILL = %q, want unset", v)
	}
	if got := os.Getenv("GOINFER_CPU_FAST_ATTENTION"); got != "1" {
		t.Errorf("GOINFER_CPU_FAST_ATTENTION = %q, want unchanged %q — Load with ExactPrefill=false "+
			"clobbered a caller-managed env var (M-26)", got, "1")
	}
}

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

// TestLoad_exactPrefillOptionIsAModelProperty is M-26's chokepoint gate (docs/audit-2026-09-10.md),
// revised 2026-09-24. decoder.Options.ExactPrefill used to be applied by Load setting the three
// fast-prefill env vars — process-global and never undone, so every model loaded later in the same
// process inherited it. It is now recorded on the Model and consulted by each backend's switch next
// to its env var (CPU: Model.cpuFastAttention; CUDA: at resident build; Metal: on its resident). This
// proves Load reports it on the model, applies it to this model's CPU prefill, and writes nothing to
// the environment. The two-model inheritance case is TestExactPrefill_isPerModel.
func TestLoad_exactPrefillOptionIsAModelProperty(t *testing.T) {
	unsetenvT(t, "GOINFER_METAL_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CUDA_FAST_PREFILL")
	unsetenvT(t, "GOINFER_CPU_FAST_ATTENTION")

	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32", ExactPrefill: true})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	if !m.ExactPrefill() || m.cpuFastAttention() {
		t.Errorf("Options.ExactPrefill=true: ExactPrefill()=%v cpuFastAttention()=%v, want true, false", m.ExactPrefill(), m.cpuFastAttention())
	}
	for _, name := range []string{"GOINFER_METAL_FAST_PREFILL", "GOINFER_CUDA_FAST_PREFILL", "GOINFER_CPU_FAST_ATTENTION"} {
		if got, set := os.LookupEnv(name); set {
			t.Errorf("Load wrote the process environment: %s=%q", name, got)
		}
	}
}

// TestLoad_exactPrefillFalseLeavesEnvUntouched confirms the negative: Options.ExactPrefill's
// zero value must not touch these env vars at all (not even to reset them to "1"/unset) — Load is
// called by every library consumer, including ones managing these same env vars themselves, so a
// false value must be a true no-op, not "reset to default."
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

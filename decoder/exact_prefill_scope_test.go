package decoder

import (
	"os"
	"testing"
)

// Options.ExactPrefill applies to the model that asked for it and to nothing else. Load used to
// implement it with os.Setenv, which is process-global and never undone: a second model loaded
// into the same process (serve's multi-model, any library user) inherited exact prefill.
func TestExactPrefill_isPerModel(t *testing.T) {
	for _, v := range []string{"GOINFER_CPU_FAST_ATTENTION", "GOINFER_CUDA_FAST_PREFILL", "GOINFER_METAL_FAST_PREFILL"} {
		t.Setenv(v, "") // start from the default for every backend's knob
		os.Unsetenv(v)
	}
	exact, err := Load(tinyFixture(t), Options{Backend: "cpu", ExactPrefill: true})
	if err != nil {
		t.Fatal(err)
	}
	defer exact.Close()
	plain, err := Load(tinyFixture(t), Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()

	if !exact.ExactPrefill() || exact.cpuFastAttention() {
		t.Errorf("the ExactPrefill model: ExactPrefill()=%v cpuFastAttention()=%v — want true, false", exact.ExactPrefill(), exact.cpuFastAttention())
	}
	if plain.ExactPrefill() || !plain.cpuFastAttention() {
		t.Errorf("a model loaded AFTER it without ExactPrefill: ExactPrefill()=%v cpuFastAttention()=%v — want false, true (it inherited the first model's setting)",
			plain.ExactPrefill(), plain.cpuFastAttention())
	}
	for _, v := range []string{"GOINFER_CPU_FAST_ATTENTION", "GOINFER_CUDA_FAST_PREFILL", "GOINFER_METAL_FAST_PREFILL"} {
		if got, set := os.LookupEnv(v); set {
			t.Errorf("Load wrote the process environment: %s=%q", v, got)
		}
	}
}

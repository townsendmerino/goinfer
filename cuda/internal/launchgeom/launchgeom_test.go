package launchgeom

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// TestGemvRN_matchesCuDefines holds GemvRNRows and GemvRNMT to the #define defaults in gemv_w4a8_rn.cu, so a change to one
// side alone fails here instead of launching the kernel with a grid that skips or repeats rows.
func TestGemvRN_matchesCuDefines(t *testing.T) {
	b, err := os.ReadFile("../../gemv_w4a8_rn.cu")
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]int{"RN": GemvRNRows, "MT": GemvRNMT} {
		m := regexp.MustCompile(`(?m)^#define ` + name + ` (\d+)$`).FindSubmatch(b)
		if m == nil {
			t.Fatalf("gemv_w4a8_rn.cu has no #define %s", name)
		}
		if got, _ := strconv.Atoi(string(m[1])); got != want {
			t.Errorf("gemv_w4a8_rn.cu defines %s %d; launchgeom has %d", name, got, want)
		}
	}
}

// TestKQuantGemv_matchesCuDefine holds KQuantGemvRows to #define KQ_ROWS in actgroup.cu, so a change to one side alone fails
// here instead of launching gemv_q5k_g32 / gemv_q6k_g32 with a grid that skips or repeats rows.
func TestKQuantGemv_matchesCuDefine(t *testing.T) {
	b, err := os.ReadFile("../../actgroup.cu")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?m)^#define KQ_ROWS (\d+)$`).FindSubmatch(b)
	if m == nil {
		t.Fatal("actgroup.cu has no #define KQ_ROWS")
	}
	if got, _ := strconv.Atoi(string(m[1])); got != KQuantGemvRows {
		t.Errorf("actgroup.cu defines KQ_ROWS %d; launchgeom has %d", got, KQuantGemvRows)
	}
}

// TestGemmMMA_matchesCuDefines holds the GemmMMA* tile constants to gemm_w4a8_mma.cu's defines (GemmMMAThreads to GWARPS*32),
// so a change to one side alone fails here instead of launching the kernel with a grid or shared-memory size it does not use.
func TestGemmMMA_matchesCuDefines(t *testing.T) {
	b, err := os.ReadFile("../../gemm_w4a8_mma.cu")
	if err != nil {
		t.Fatal(err)
	}
	// Each define times its unit is the launchgeom value: GWARPS counts warps, so it is compared as GWARPS*32 and a thread
	// count that is not a multiple of 32 cannot match by truncation.
	for name, want := range map[string]struct{ value, unit int }{
		"GBM": {GemmMMABM, 1}, "GBN": {GemmMMABN, 1}, "GWARPS": {GemmMMAThreads, 32}, "GKSTEP": {GemmMMAKStep, 1}, "GAPAD": {GemmMMAAPad, 1},
	} {
		m := regexp.MustCompile(`(?m)^#define ` + name + ` (\d+)\b`).FindSubmatch(b)
		if m == nil {
			t.Fatalf("gemm_w4a8_mma.cu has no #define %s", name)
		}
		if got, _ := strconv.Atoi(string(m[1])); got*want.unit != want.value {
			t.Errorf("gemm_w4a8_mma.cu defines %s %d (x%d = %d); launchgeom has %d", name, got, want.unit, got*want.unit, want.value)
		}
	}
}

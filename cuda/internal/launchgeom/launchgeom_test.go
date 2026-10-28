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

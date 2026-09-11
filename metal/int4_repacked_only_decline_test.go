//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestBuildResident_declinesWithLoudReasonForRepackedOnlyInt4 pins the exact regression
// TestMetalSnapshotGolden hit and the fix for it (aikit v1.41.0's repacked-only int4 policy,
// decoder/weightmat.go's wantsCanonicalInt4): a model loaded with Options.Backend:"cpu" — the
// explicit promise that unlocks repacked-only int4 — has no canonical bytes left for Q/K/V,
// so Metal's own buildResident must DECLINE (not panic uncaught, not silently build wrong) with
// a specific, actionable reason naming the real condition, not the old "weight kind %q not int8
// or int4" message that both misdescribed the check and gave no next step.
func TestBuildResident_declinesWithLoudReasonForRepackedOnlyInt4(t *testing.T) {
	const fixture = "../testdata/gemma4-dense-scaled"
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("no fixture at %s", fixture)
	}
	m, err := decoder.Load(fixture, decoder.Options{Quant: "int4", Backend: "cpu"})
	if err != nil {
		t.Fatalf("load %s: %v", fixture, err)
	}
	defer m.Close()

	_, err = buildResident(m)
	if err == nil {
		t.Fatal("buildResident succeeded on a Backend:\"cpu\"-loaded (repacked-only int4) model " +
			"— want a clean decline naming the reason")
	}
	got := err.Error()
	for _, want := range []string{
		"no canonical bytes", // names the real condition, not "weight kind ... not int8 or int4"
		"row4-only",          // names which layout, not just "int4"
		"Options.Backend",    // the actionable fix
		"\"metal\"",          // the specific value to set it to
	} {
		if !strings.Contains(got, want) {
			t.Errorf("decline error %q does not mention %q", got, want)
		}
	}
	t.Logf("decline error: %s", got)
}

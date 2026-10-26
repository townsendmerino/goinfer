package decoder

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/paritykey"
)

// TestParityKey_matchesFreshDepsHash is VK5's P1 (docs/tasks/task-subsystems-2026-10.md §5.8): internal/paritykey's
// today key equals freshDepsHash for every family, so the shadow job's "today" column is the gate's own hash.
func TestParityKey_matchesFreshDepsHash(t *testing.T) {
	raw, err := os.ReadFile(parityManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	var m parityManifest
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	src := paritykey.Dir("..")
	pm, err := paritykey.ReadManifest(src)
	if err != nil {
		t.Fatal(err)
	}
	pin, err := paritykey.AikitPin(src)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Families) == 0 || len(pm.Families) != len(m.Families) {
		t.Fatalf("%d families in the manifest, %d read by paritykey", len(m.Families), len(pm.Families))
	}
	for name, fam := range m.Families {
		want, err := freshDepsHash(&m, fam)
		if err != nil {
			t.Fatal(err)
		}
		got, err := paritykey.Key(src, pm, pm.Families[name], paritykey.TodayTail(pin))
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("%s: paritykey %s, freshDepsHash %s", name, got, want)
		}
	}
}

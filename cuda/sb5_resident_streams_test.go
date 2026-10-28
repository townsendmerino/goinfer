//go:build cuda && goinfer_testhooks

package cuda

import (
	"flag"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

var (
	sb5Out  = flag.String("sb5.out", "", "write the resident decode streams to this file")
	sb5Want = flag.String("sb5.want", "", "hold the resident decode streams to the record in this file")
)

// TestSB5ResidentStreams records generateInto's decode paths on llama-tiny resident on cuda, alone and under MC3, and holds
// them to a record taken before a change (-sb5.out writes one, -sb5.want compares): the before/after identity for moving
// the plain decode loop onto Forward (docs/tasks/task-subsystems-2026-10.md §8 SB5). Skips without a device.
func TestSB5ResidentStreams(t *testing.T) {
	got := map[string]decoder.ResidentStreamCase{}
	for _, mc3 := range []bool{false, true} {
		opts := decoder.Options{Backend: "cuda", Quant: "int4"}
		prefix := "solo/"
		if mc3 {
			opts.ResidentKVSlots = 4
			prefix = "mc3/"
		}
		m, err := decoder.Load("../testdata/llama-tiny", opts)
		if err != nil {
			t.Skipf("no cuda load: %v", err)
		}
		if !m.ResidentActive() {
			m.Close()
			t.Skip("llama-tiny did not go resident on this device")
		}
		if mc3 {
			if n := m.EnableResidentConcurrency(4); n < 2 {
				t.Logf("MC3 not available on this resident (n=%d): the mc3/ cases run unbatched", n)
				prefix = "mc3-off/"
			}
		}
		cs, err := decoder.ResidentStreamsForTest(m, prefix)
		m.Close()
		if err != nil {
			t.Fatal(err)
		}
		for k, v := range cs {
			got[k] = v
		}
	}
	diffs, err := decoder.CompareResidentStreamsForTest(got, *sb5Out, *sb5Want)
	if err != nil {
		t.Fatal(err)
	}
	if len(diffs) > 0 {
		t.Fatalf("%d of %d resident decode streams differ from the record:\n%s", len(diffs), len(got), strings.Join(diffs, "\n"))
	}
	t.Logf("%d resident decode streams", len(got))
}

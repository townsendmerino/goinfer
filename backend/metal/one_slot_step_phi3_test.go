//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestOneSlotStep_phi3PrefillAB is P29's tier 1 measurement (docs/queue-performance.md, P29): Phi-3 mini 4k at --quant q4k
// on one KV slot, one 512-token prompt prefilled from position 0 by production's sequential decode and by the step route,
// 5 pairs in one process with the arm order alternating. It reports each pair's sequential ÷ step wall time and their
// median; the registration's reading is that median. It loads the real file, so it runs only with GOINFER_P29_AB=1.
//
//	GOINFER_P29_AB=1 go test -count=1 -tags goinfer_testhooks -timeout 20m -run '^TestOneSlotStep_phi3PrefillAB$' -v ./metal/
func TestOneSlotStep_phi3PrefillAB(t *testing.T) {
	if os.Getenv("GOINFER_P29_AB") != "1" {
		t.Skip("set GOINFER_P29_AB=1 (a timed run on the real Phi-3 file)")
	}
	oneSlotPrefillAB(t, decoder.AssetPathForTest(t, "GOINFER_PHI3_GGUF"), 2048) // the asset registry (testdata/assets.json) resolves it
}

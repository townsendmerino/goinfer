package decoder

import (
	"os"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
)

// deltaNetTiming prints per-phase DeltaNet timing when GOINFER_DELTANET_TIMING is set.
var deltaNetTiming = os.Getenv("GOINFER_DELTANET_TIMING") != ""

// convWinSliceForTest leaves the conv window as slices of the pass slab for the copy-vs-slice regression test.
var convWinSliceForTest bool

// PrintDeltaNetTiming prints the accumulated DeltaNet phase timings when enabled.
func PrintDeltaNetTiming() {
	syncCPUHooks()
	cpuengine.PrintDeltaNetTiming()
}

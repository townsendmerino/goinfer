//go:build cuda && goinfer_testhooks

package cuda

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// releasedDrafter is a drafter whose host weights were released; any other method it is asked for panics.
type releasedDrafter struct{ decoder.BlockDrafterWeights }

func (releasedDrafter) HostWeightsReleased() bool { return true }

// TestAttachDrafter_refusesReleasedWeights: a drafter whose host weights were released (DFlashDrafter.ReleaseHostWeights) is
// refused before anything is read or uploaded, rather than uploading zero matrices that draft noise.
func TestAttachDrafter_refusesReleasedWeights(t *testing.T) {
	_, err := (&cudaResident{}).AttachDrafter(releasedDrafter{})
	if err == nil || !strings.Contains(err.Error(), "released") {
		t.Fatalf("AttachDrafter on released weights: %v, want a refusal naming the release", err)
	}
}

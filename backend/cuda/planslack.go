//go:build cuda

package cuda

import (
	"errors"

	"github.com/townsendmerino/goinfer/decoder"
)

// The CUDA build's allocation rounding over a loaded model's weight buffers, for decoder.Model.Plan: the driver rounds every
// buffer of 2 MiB or more up to 2 MiB (allocRoundSlack), and the plan prices that beside the weights so `fit`, the night
// queue's check and the build use one figure. It is the build's own planning (buildResident in plan-only mode), so there is
// one copy of the buffer layout. ok is false when the build would decline the model.
func planSlack(m *decoder.Model) (int64, bool) {
	_, _, err := (&cudaBackend{}).buildResident(m, true)
	if s, ok := errors.AsType[planOnlySlack](err); ok {
		return int64(s), true
	}
	return 0, false
}

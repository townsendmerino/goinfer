//go:build cuda

package kernels

import (
	"errors"
	"testing"

	"github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// A caller that can degrade recovers gpudev's allocation panic, and mem.DeviceAllocFailed tells it from a bug by its text. gpudev may
// import nothing of goinfer's, so the two are held together here, in a package that imports gpudev and not decoder: reword either and
// every out-of-memory decline becomes a crash.
func TestAllocFailedPanic_isWhatMemReads(t *testing.T) {
	if p := gpudev.AllocFailedPanic("kv", 64, errors.New("x")); !mem.DeviceAllocFailed(p) {
		t.Errorf("mem.DeviceAllocFailed does not take gpudev's allocation panic %q", p)
	}
}

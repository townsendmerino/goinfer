//go:build cuda

package kernels

import (
	"errors"
	"testing"

	"github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// A caller that can degrade recovers gpudev's allocation panic, and mem.DeviceAllocFailed tells it from a bug by its type's marker
// method. gpudev may import nothing of goinfer's, so the two are held together here, in a package that imports gpudev and not decoder:
// rename the method on either side and every out-of-memory decline becomes a crash.
func TestAllocFailedPanic_isWhatMemReads(t *testing.T) {
	p := gpudev.AllocFailedPanic("kv", 64, errors.New("x"))
	if !mem.DeviceAllocFailed(p) {
		t.Errorf("mem.DeviceAllocFailed does not take gpudev's allocation panic %v", p)
	}
	if _, typed := any(p).(mem.AllocFailure); !typed {
		t.Errorf("gpudev's allocation panic %T is not a mem.AllocFailure: it passes on its text alone", p)
	}
}

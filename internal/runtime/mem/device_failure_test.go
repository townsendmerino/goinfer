package mem

import (
	"errors"
	"fmt"
	"testing"
)

func TestDeviceAllocFailed_readsErrorsAndRecoveredPanics(t *testing.T) {
	const panicText = "cuda: device allocation failed (kv, 64 bytes): cuMemAlloc_v2: CUDA_ERROR_OUT_OF_MEMORY"
	for _, c := range []struct {
		name string
		v    any
		want bool
	}{
		{"a recovered string", panicText, true},
		{"an error", errors.New(panicText), true},
		{"wrapped", fmt.Errorf("cuda: executor job panicked: %w", errors.New(panicText)), true},
		{"a programming bug", "runtime error: index out of range [3] with length 2", false},
		{"a launch out of memory", errors.New("cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY"), false},
		{"nil", nil, false},
	} {
		if got := DeviceAllocFailed(c.v); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

func TestDeviceOutOfMemory_isAnAllocationOrADriverOutOfMemory(t *testing.T) {
	for _, c := range []struct {
		name string
		err  error
		want bool
	}{
		{"a failed allocation", errors.New("gpu: newDecodeRunner: device allocation failed (VRAM exhausted?): x"), true},
		{"a launch out of memory", errors.New("cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY"), true},
		{"some other failure", errors.New("cuLaunchKernel: CUDA_ERROR_ILLEGAL_ADDRESS"), false},
		{"no error", nil, false},
	} {
		if got := DeviceOutOfMemory(c.err); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

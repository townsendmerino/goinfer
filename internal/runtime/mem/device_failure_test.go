package mem

import (
	"errors"
	"fmt"
	"runtime"
	"testing"
)

// typedFailure stands in a backend's allocation-failure type: it carries the marker method and none of the text.
type typedFailure struct{}

func (typedFailure) Error() string       { return "out of room" }
func (typedFailure) DeviceAllocFailure() {}

// aBug is a real runtime.Error, recovered from an index out of range.
func aBug() (bug runtime.Error) {
	defer func() { bug = recover().(runtime.Error) }()
	var xs []int
	_ = xs[len(xs)]
	return nil
}

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
		{"a typed failure", typedFailure{}, true},
		{"a typed failure, wrapped", fmt.Errorf("cuda: executor job panicked: %w", typedFailure{}), true},
		{"DeviceAllocError, wrapped", fmt.Errorf("gpu: newDecodeRunner: %w (VRAM exhausted?): x", DeviceAllocError{}), true},
		{"a programming bug's text", "runtime error: index out of range [3] with length 2", false},
		{"a runtime.Error", aBug(), false},
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

// RepanicBug raises a runtime.Error again and returns on anything else: an allocation failure, an error, and the string panics a
// backend uses as documented declines.
func TestRepanicBug_raisesOnlyRuntimeErrors(t *testing.T) {
	raises := func(v any) (raised bool) {
		defer func() { raised = recover() != nil }()
		RepanicBug(v)
		return false
	}
	if !raises(aBug()) {
		t.Error("RepanicBug returned on a runtime.Error")
	}
	for _, v := range []any{typedFailure{}, errors.New("cuda: launch failed"), "metal: expert weights are not int4", 7} {
		if raises(v) {
			t.Errorf("RepanicBug raised %T %v, which is not a programming bug", v, v)
		}
	}
}

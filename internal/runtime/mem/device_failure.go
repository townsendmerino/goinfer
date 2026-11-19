package mem

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
)

// AllocFailure is implemented by the error a backend raises, by panic or return, when the device has no memory for an allocation
// (backend/cuda's gpudev.AllocFailure, ErrDeviceAlloc). The marker method lets a backend package that imports nothing of goinfer's
// produce one: it declares the method and never names this interface.
type AllocFailure interface {
	error
	DeviceAllocFailure()
}

// ErrDeviceAlloc is a failed device allocation for a backend with no failure type of its own: wrap it with %w.
var ErrDeviceAlloc error = allocFailedErr{}

type allocFailedErr struct{}

func (allocFailedErr) Error() string       { return deviceAllocFailedText }
func (allocFailedErr) DeviceAllocFailure() {}

const (
	deviceAllocFailedText = "device allocation failed"
	deviceOOMText         = "OUT_OF_MEMORY"
)

// DeviceAllocFailed reports whether v, an error or a value recovered from a panic, is a failed device allocation. Only this answer lets
// a recover absorb a panic: any other is a bug, and must be raised again. A runtime.Error is never one.
//
// The AllocFailure type decides. The text match is the fallback for a failure that reached the caller as text: a recovered panic or a
// wrapped error formatted with %v on the way (an executor job's error, a vision tower's), which loses the type.
func DeviceAllocFailed(v any) bool {
	if v == nil {
		return false
	}
	if _, bug := v.(runtime.Error); bug {
		return false
	}
	if err, ok := v.(error); ok {
		var f AllocFailure
		if errors.As(err, &f) {
			return true
		}
		return strings.Contains(err.Error(), deviceAllocFailedText)
	}
	return strings.Contains(fmt.Sprint(v), deviceAllocFailedText)
}

// RepanicBug raises v again when it is a runtime.Error (a nil dereference, an index out of range, a bad conversion): a programming bug,
// which a recover that turns a panic into an error or a decline must not absorb, or the bug reads as "the device declined" and the
// model runs on a slower path. Call it first in such a recover; any other value returns.
func RepanicBug(v any) {
	if e, ok := v.(runtime.Error); ok {
		panic(e)
	}
}

// DeviceOutOfMemory reports whether err is the device running out of memory: a failed allocation, or the driver's OUT_OF_MEMORY from a
// launch.
func DeviceOutOfMemory(err error) bool {
	return err != nil && (DeviceAllocFailed(err) || strings.Contains(err.Error(), deviceOOMText))
}

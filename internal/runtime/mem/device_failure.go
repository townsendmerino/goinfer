package mem

import (
	"fmt"
	"strings"
)

// The device failures are told apart by their text. A failed cuMemAlloc reaches a caller as the panic gpudev's MustBufOn raises, a
// string, and a failed launch as the driver's error name. gpudev's TestAllocFailedPanic_saysWhatMemReads and this package's own test hold
// the same words, so change them together. Typing the panic, so this can be errors.As, is the follow-up in
// docs/measurements/error-hygiene-audit-2026-11-04/README.md.
const (
	deviceAllocFailedText = "device allocation failed"
	deviceOOMText         = "OUT_OF_MEMORY"
)

// DeviceAllocFailed reports whether v, an error or a value recovered from a panic, is a failed device allocation. Only this answer lets
// a recover absorb a panic: any other is a bug, and must be raised again.
func DeviceAllocFailed(v any) bool {
	return strings.Contains(fmt.Sprint(v), deviceAllocFailedText)
}

// DeviceOutOfMemory reports whether err is the device running out of memory: a failed allocation, or the driver's OUT_OF_MEMORY from a
// launch.
func DeviceOutOfMemory(err error) bool {
	return err != nil && (DeviceAllocFailed(err) || strings.Contains(err.Error(), deviceOOMText))
}

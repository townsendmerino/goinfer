package serveapp

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/townsendmerino/goinfer/multimodal"
)

// A device vision tower that runs out of VRAM on a request falls back to the CPU tower for that request and every later one, and gives its VRAM back (S4,
// docs/tasks/task-multimodal-support-2026-10.md). The towers load or grow their scratch after the resident decoder has taken the card, so the reserve in tower_reserve.go
// is an estimate and a large image can still miss it; failing the request (or, before recoverDeviceTower, the process) is worse than a slower tower. Only a device
// MEMORY failure triggers it: any other error is the caller's.

func isDeviceMemoryError(err error) bool {
	if err == nil {
		return false
	}
	m := err.Error()
	return strings.Contains(m, "OUT_OF_MEMORY") || strings.Contains(m, "device allocation failed")
}

// deviceFallback guards one grid tower's accelerator. Calls through the accelerator are serialized here (the accelerators serialize themselves anyway); the CPU path is not.
type deviceFallback struct {
	mu sync.Mutex
}

// run calls fn with the accelerator *accp, and if that fails for want of device memory closes it, clears *accp (so the tower is CPU from now on) and calls fn(nil).
func (f *deviceFallback) run(family string, accp *multimodal.GridTowerAccelerator, fn func(acc multimodal.GridTowerAccelerator) ([]float32, error)) ([]float32, error) {
	f.mu.Lock()
	acc := *accp
	if acc == nil {
		f.mu.Unlock()
		return fn(nil)
	}
	out, err := fn(acc)
	if err == nil || !isDeviceMemoryError(err) {
		f.mu.Unlock()
		return out, err
	}
	fmt.Fprintf(os.Stderr, "vision: the %s tower ran out of device memory (%v); it runs on the CPU from now on\n", family, err)
	_ = acc.Close()
	*accp = nil
	f.mu.Unlock()
	return fn(nil)
}

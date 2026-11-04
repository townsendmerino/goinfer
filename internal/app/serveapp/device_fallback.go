package serveapp

import (
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/townsendmerino/goinfer/multimodal"
)

// A device vision tower that runs out of VRAM on a request falls back to the CPU tower for that request and every
// later one, and gives its VRAM back. The towers load or grow their scratch after the resident decoder has taken the
// card, so the reserve in tower_reserve.go is an estimate and a large image can still miss it; failing the request
// (or the process) is worse than a slower tower. Only a device memory failure triggers it: any other error is the
// caller's. Under -require-backend there is no fallback: the request fails with the reason and the device tower stays
// (a smaller image may fit), because -require-backend means "never run off the requested backend", at load or later.

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

// refuseCPUFallback is the error a device tower's memory failure becomes under -require-backend.
func refuseCPUFallback(family string, err error) error {
	return fmt.Errorf("the %s tower ran out of device memory and -require-backend refuses the CPU fallback (lower -ctx or -kv-sessions, resize the image, or drop -require-backend): %w", family, err)
}

// run calls fn with the accelerator *accp; a panic on the device becomes an error (recoverDeviceTower). If the device call fails for want of device memory, under
// require (-require-backend) the request fails and the accelerator stays; otherwise it is closed, *accp cleared (so the tower is CPU from now on) and fn(nil) runs.
func (f *deviceFallback) run(family string, require bool, accp *multimodal.GridTowerAccelerator, fn func(acc multimodal.GridTowerAccelerator) ([]float32, error)) ([]float32, error) {
	f.mu.Lock()
	acc := *accp
	if acc == nil {
		f.mu.Unlock()
		return fn(nil)
	}
	out, err := recoverDeviceTower(family, func() ([]float32, error) { return fn(acc) })
	if err == nil || !isDeviceMemoryError(err) {
		f.mu.Unlock()
		return out, err
	}
	if require {
		f.mu.Unlock()
		return nil, refuseCPUFallback(family, err)
	}
	fmt.Fprintf(os.Stderr, "vision: the %s tower ran out of device memory (%v); it runs on the CPU from now on\n", family, err)
	_ = acc.Close()
	*accp = nil
	f.mu.Unlock()
	return fn(nil)
}

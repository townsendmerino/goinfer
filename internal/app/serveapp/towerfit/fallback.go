package towerfit

import (
	"fmt"
	"io"
	"os"
	"slices"
	"sync"

	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/multimodal"
)

// RecoverDeviceTower runs a tower forward pass and converts any panic into an
// actionable error so an allocation panic in a device tower fails only the request.
func RecoverDeviceTower(family string, fn func() ([]float32, error)) (out []float32, err error) {
	defer func() {
		if r := recover(); r != nil {
			hint := ""
			if rErr, ok := r.(error); ok {
				if mem.DeviceOutOfMemory(rErr) {
					hint = ": the input needs more device memory than the decoder left it; lower -ctx or -kv-sessions, resize the image, or use -vision-device cpu"
				}
				out, err = nil, fmt.Errorf("the %s tower failed (%w)%s", family, rErr, hint)
				return
			}
			if mem.DeviceOutOfMemory(fmt.Errorf("%v", r)) {
				hint = ": the input needs more device memory than the decoder left it; lower -ctx or -kv-sessions, resize the image, or use -vision-device cpu"
			}
			out, err = nil, fmt.Errorf("the %s tower failed (%v)%s", family, r, hint)
		}
	}()
	return fn()
}

// RefuseCPUFallback formats a device-memory error when -require-backend forbids CPU fallback.
func RefuseCPUFallback(family string, err error) error {
	return fmt.Errorf("the %s tower ran out of device memory and -require-backend refuses the CPU fallback (lower -ctx or -kv-sessions, resize the image, or drop -require-backend): %w", family, err)
}

// DeviceFallback serializes one grid tower's accelerator and moves it permanently
// to CPU (releasing device memory) on a device out-of-memory error unless require is set.
type DeviceFallback struct {
	mu sync.Mutex
}

// Run executes fn against *accp, falling back to fn(nil) on device OOM when !require.
func (f *DeviceFallback) Run(family string, require bool, accp *multimodal.GridTowerAccelerator, fn func(acc multimodal.GridTowerAccelerator) ([]float32, error)) ([]float32, error) {
	f.mu.Lock()
	acc := *accp
	if acc == nil {
		f.mu.Unlock()
		return fn(nil)
	}
	out, err := RecoverDeviceTower(family, func() ([]float32, error) { return fn(acc) })
	if !mem.DeviceOutOfMemory(err) {
		f.mu.Unlock()
		return out, err
	}
	if require {
		f.mu.Unlock()
		return nil, RefuseCPUFallback(family, err)
	}
	fmt.Fprintf(os.Stderr, "vision: the %s tower ran out of device memory (%v); it runs on the CPU from now on\n", family, err)
	_ = acc.Close()
	*accp = nil
	f.mu.Unlock()
	return fn(nil)
}

// EnableResidentTower attaches a device-resident vision tower on webgpu, cuda or
// metal, warning and leaving CPU intact on failure.
func EnableResidentTower(enc interface{ EnableResident() error }, backend string, warn io.Writer) bool {
	ok, _ := AttachResidentTower("", enc, backend, false, warn)
	return ok
}

// AttachResidentTower attaches a device-resident vision tower, refusing on error
// when require is true.
func AttachResidentTower(family string, enc interface{ EnableResident() error }, backend string, require bool, warn io.Writer) (bool, error) {
	if backend != "webgpu" && backend != "cuda" && backend != "metal" {
		return false, nil
	}
	if err := enc.EnableResident(); err != nil {
		if require {
			return false, fmt.Errorf("-require-backend: the %s tower could not start on %s: %w", family, backend, err)
		}
		fmt.Fprintf(warn, "warning: the resident GPU vision tower could not be enabled (%v); images will run through the CPU tower (slower)\n", err)
		return false, nil
	}
	return true, nil
}

// AttachWhisperDevice attaches a Whisper-style audio encoder to backend's device
// when registered, returning the device (or nil) and a human-readable placement label.
func AttachWhisperDevice(enc *audio.WhisperEncoder, backend string, warn io.Writer) (audio.WhisperDevice, string) {
	if !slices.Contains(multimodal.WhisperDevices(), backend) {
		return nil, "CPU"
	}
	d, err := multimodal.UseWhisperDevice(backend, enc)
	if err != nil {
		fmt.Fprintf(warn, "audio: the %s Whisper encoder declined (%v); the encoder runs on the CPU\n", backend, err)
		return nil, "CPU"
	}
	if p, ok := d.(interface{ Placement() string }); ok {
		return d, backend + " (" + p.Placement() + ")"
	}
	return d, backend
}

package embeddinggemma2

import (
	"fmt"
	"sort"

	"github.com/townsendmerino/aikit/vision"
)

// The vision tower on another device (Phase VM, docs/tasks/task-embeddinggemma2.md: Metal). A VisionAccelerator runs
// the tower's patch embed and every encoder layer from aikit's export (vision.Gemma4Encoder.Weights); the pool and the
// projection after them stay aikit's (FinishHidden), shared with the CPU tower.

// VisionAccelerator runs the Gemma 4 tower up to its pool on another device.
type VisionAccelerator interface {
	Name() string
	// Hidden is the last encoder layer's output [len(pos), hidden] for patches [len(pos), 3*16*16] in [0, 1] at
	// their (x, y) positions; patches is not modified.
	Hidden(patches []float32, pos [][2]int) ([]float32, error)
	Close() error
}

var visAccels = map[string]func(*vision.Gemma4Encoder) (VisionAccelerator, error){}

// RegisterVisionAccelerator makes a tower accelerator available by name, from a backend module's init (as
// RegisterAccelerator does for the text encoder). UseAccelerator with the same name moves the tower too.
func RegisterVisionAccelerator(name string, factory func(*vision.Gemma4Encoder) (VisionAccelerator, error)) {
	accelMu.Lock()
	defer accelMu.Unlock()
	visAccels[name] = factory
}

// VisionAccelerators lists the registered tower accelerator names, sorted.
func VisionAccelerators() []string {
	accelMu.Lock()
	defer accelMu.Unlock()
	var n []string
	for k := range visAccels {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// NewVisionAccelerator builds the named tower accelerator over a loaded tower (float32, LoadGemma4Encoder
// quant=false).
func NewVisionAccelerator(name string, enc *vision.Gemma4Encoder) (VisionAccelerator, error) {
	accelMu.Lock()
	f, ok := visAccels[name]
	accelMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("embeddinggemma2: no %q vision accelerator in this binary (registered: %v)", name, VisionAccelerators())
	}
	return f(enc)
}

// bindVisionAccel puts a loaded tower on the encoder's accelerator when one of the same name is registered for the
// tower; a decline leaves the CPU tower and is recorded in VisionDevice.
func (e *Encoder) bindVisionAccel() {
	v := e.vis
	if v == nil || e.accel == nil {
		return
	}
	if v.accel != nil {
		v.accel.Close()
		v.accel, v.device = nil, ""
	}
	name := e.accel.Name()
	accelMu.Lock()
	_, ok := visAccels[name]
	accelMu.Unlock()
	if !ok {
		v.device = "CPU (no " + name + " tower in this binary)"
		return
	}
	a, err := NewVisionAccelerator(name, v.enc)
	if err != nil {
		v.device = "CPU (" + name + " declined: " + err.Error() + ")"
		return
	}
	v.accel, v.device = a, name
}

// VisionDevice says where the tower runs: "CPU", an accelerator's name, or "CPU (<why not the accelerator>)". "" until
// EnableVision.
func (e *Encoder) VisionDevice() string {
	if e.vis == nil {
		return ""
	}
	if e.vis.device == "" {
		return "CPU"
	}
	return e.vis.device
}

// VisionAccelerator is the tower's accelerator, or nil when it runs on the CPU.
func (e *Encoder) VisionAccelerator() VisionAccelerator {
	if e.vis == nil {
		return nil
	}
	return e.vis.accel
}

// ImageFeaturesCPU is ImageFeaturesFrom on the CPU tower whatever the encoder's tower runs on (a test compares the
// two).
func (e *Encoder) ImageFeaturesCPU(patches []float32, pos [][2]int) ([]float32, int, error) {
	if e.vis == nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: vision is not enabled (EnableVision)")
	}
	return e.featuresOn(nil, patches, pos)
}

package embeddinggemma2

import (
	"fmt"
	"slices"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The vision tower on another device (Phase VM, docs/tasks/task-embeddinggemma2.md: Metal). A VisionAccelerator runs
// the tower's patch embed and every encoder layer from aikit's export (vision.Gemma4Encoder.Weights); the pool and the
// projection after them stay aikit's (FinishHidden), shared with the CPU tower.

// VisionAccelerator runs the Gemma 4 tower up to its pool on another device. The registry is multimodal's since
// 2026-10-06 (F2 of docs/multimodal.md), shared with Gemma 4's own image input; these names stay as aliases.
type VisionAccelerator = multimodal.Gemma4TowerAccelerator

// RegisterVisionAccelerator is multimodal.RegisterGemma4Tower. UseAccelerator with the same name moves the tower too.
func RegisterVisionAccelerator(name string, factory func(*vision.Gemma4Encoder) (VisionAccelerator, error)) {
	multimodal.RegisterGemma4Tower(name, factory)
}

// VisionAccelerators lists the registered tower accelerator names, sorted (multimodal.Gemma4Towers).
func VisionAccelerators() []string { return multimodal.Gemma4Towers() }

// NewVisionAccelerator builds the named tower accelerator over a loaded tower (multimodal.NewGemma4Tower).
func NewVisionAccelerator(name string, enc *vision.Gemma4Encoder) (VisionAccelerator, error) {
	return multimodal.NewGemma4Tower(name, enc)
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
	if !slices.Contains(multimodal.Gemma4Towers(), name) {
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

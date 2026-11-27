//go:build gpu

package towers

import (
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// init registers the WebGPU multimodal vision tower accelerators when this package is linked.
func init() {
	multimodal.RegisterGemma4Tower("webgpu", func(enc *vision.Gemma4Encoder) (multimodal.Gemma4TowerAccelerator, error) {
		return newG4VTower(enc)
	})
	multimodal.RegisterQwen3Tower("webgpu", func(enc *vision.Qwen3VisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newQwen3VTower(enc)
	})
	multimodal.RegisterGlmOcrTower("webgpu", func(enc *vision.GlmOcrVisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newGlmOcrVTower(enc)
	})
	vision.RegisterQwenResident(func(enc *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newQwen25VResident(enc)
	})
	multimodal.MarkResidentTower(multimodal.TowerQwen25VL, "webgpu")
	vision.RegisterResident(newSiglipResident)
	multimodal.MarkResidentTower(multimodal.TowerSigLIP, "webgpu")
}

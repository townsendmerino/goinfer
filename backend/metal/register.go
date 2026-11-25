//go:build darwin

package metal

import (
	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// init registers the Metal decoder backend, its memory/hardware/fit probes, its device ledger leaser,
// and its multimodal vision/audio/embedding accelerators when this package is blank-imported.
func init() {
	decoder.RegisterBackend("metal", func() (decoder.Backend, error) {
		return &metalBackend{}, nil
	})
	decoder.RegisterMemoryProbe("metal", func() (int64, bool) {
		ram, err := physicalRAM()
		if err != nil || ram == 0 {
			return 0, false
		}
		return metalMemoryCeiling(ram), true
	})
	decoder.RegisterBuildDemand("metal", buildDemand)
	decoder.RegisterHardwareInfo("metal", metalHardwareInfo)
	decoder.RegisterMetalWindowPool(windowPoolBlocks)

	gpu.NewLeaser = func() gpu.Leaser { return &deviceLedger{} }

	embeddinggemma2.RegisterAccelerator("metal", func(m *embeddinggemma2.Model) (embeddinggemma2.Accelerator, error) {
		return newEG2Accel(m)
	})
	embeddinggemma2.RegisterAudioAccelerator("metal", func(enc *audio.Gemma4AudioEncoder) (embeddinggemma2.AudioAccelerator, error) {
		return newG4AAccel(enc)
	})
	multimodal.RegisterGemma4Tower("metal", func(enc *vision.Gemma4Encoder) (multimodal.Gemma4TowerAccelerator, error) {
		return newG4VAccel(enc)
	})
	multimodal.RegisterQwen3Tower("metal", func(enc *vision.Qwen3VisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newQwen3VAccel(enc)
	})
	multimodal.RegisterGlmOcrTower("metal", func(enc *vision.GlmOcrVisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newGlmOcrVAccel(enc)
	})
	vision.RegisterResident(func(enc *vision.Encoder) (vision.ResidentEncoder, error) {
		return newSiglipVResident(enc)
	})
	vision.RegisterQwenResident(func(enc *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newQwen25VResident(enc)
	})
	multimodal.MarkResidentTower(multimodal.TowerSigLIP, "metal")
	multimodal.MarkResidentTower(multimodal.TowerQwen25VL, "metal")
}

//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// init registers the CUDA decoder backend, its memory/hardware/fit/KV-policy probes, its device
// ledger leaser, and its multimodal vision/audio accelerators when this package is blank-imported.
func init() {
	decoder.RegisterBackend("cuda", func() (decoder.Backend, error) {
		return &cudaBackend{}, nil
	})
	decoder.RegisterMemoryProbe("cuda", freeVRAM)
	decoder.RegisterBuildDemand("cuda", buildDemand)
	decoder.RegisterHardwareInfo("cuda", cudaHardwareInfo)
	decoder.RegisterPlanSlack("cuda", planSlack)
	decoder.RegisterKVF16Policy("cuda", decoder.KVF16Policy{
		Decline:     decoder.CUDAKVF16Decline,
		Auto:        func() bool { return kvF16Auto.Load() },
		AutoDecline: decoder.CUDAKVF16AutoDecline,
		DeclineI8:   decoder.CUDAKVI8Decline,
	})
	kvF16Auto.Store(true)

	gpu.NewLeaser = func() gpu.Leaser { return &deviceLedger{} }

	multimodal.RegisterWhisperDevice("cuda", func(enc *audio.WhisperEncoder) (audio.WhisperDevice, error) {
		t, err := newAudioTower(enc)
		if err == nil {
			return t, nil
		}
		// If full layers do not fit beside a resident decoder, stream one layer's weights at a time over the bus.
		t2, err2 := newAudioTowerMode(enc, true)
		if err2 != nil {
			return nil, fmt.Errorf("%w (and streamed: %v)", err, err2)
		}
		return t2, nil
	})
	multimodal.RegisterGemma4Tower("cuda", func(enc *vision.Gemma4Encoder) (multimodal.Gemma4TowerAccelerator, error) {
		return newG4Tower(enc)
	})
	multimodal.RegisterQwen3Tower("cuda", func(enc *vision.Qwen3VisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newQwen3Tower(enc)
	})
	multimodal.RegisterGlmOcrTower("cuda", func(enc *vision.GlmOcrVisionEncoder) (multimodal.GridTowerAccelerator, error) {
		return newGlmOcrTower(enc)
	})
	vision.RegisterQwenResident(func(enc *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newQwen25Tower(enc)
	})
	multimodal.MarkResidentTower(multimodal.TowerQwen25VL, "cuda")
	vision.RegisterResident(newSiglipResident)
	multimodal.MarkResidentTower(multimodal.TowerSigLIP, "cuda")
}

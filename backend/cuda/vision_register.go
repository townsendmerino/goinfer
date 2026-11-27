//go:build cuda

package cuda

import (
	"github.com/townsendmerino/goinfer/backend/cuda/internal/towers"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// VisionEncoder is the CUDA int8 SigLIP vision tower (backend/cuda/internal/towers.VisionEncoder).
type VisionEncoder = towers.VisionEncoder

type gridTower = towers.GridTower

func towerPTX() towers.PTX {
	return towers.PTX{
		TowerBase:      towerBasePTX,
		AttnFusedVit:   attnFusedVitPTX,
		AttnImgPrefill: attnImgPrefillPTX,
		AudioEncoder:   audioEncoderPTX,
		GeluQuant:      geluQuantPTX,
		GemvW8Batched:  gemvW8BatchedPTX,
		LayernormQuant: layernormQuantPTX,
		PrefillBatched: prefillBatchedPTX,
	}
}

// NewVisionEncoder uploads an int8 SigLIP vision tower to CUDA.
func NewVisionEncoder(w vision.GPUWeights) (*VisionEncoder, error) {
	return towers.NewVisionEncoder(w, towerPTX())
}

func newAudioTower(enc *audio.WhisperEncoder) (*towers.AudioTower, error) {
	return towers.NewAudioTower(enc, towerPTX())
}

func newAudioTowerMode(enc *audio.WhisperEncoder, streamed bool) (*towers.AudioTower, error) {
	return towers.NewAudioTowerMode(enc, streamed, towerPTX())
}

func newG4Tower(enc *vision.Gemma4Encoder) (*towers.G4Tower, error) {
	return towers.NewG4Tower(enc, towerPTX())
}

func newQwen3Tower(enc *vision.Qwen3VisionEncoder) (*towers.GridTower, error) {
	return towers.NewQwen3Tower(enc, towerPTX())
}

func newGlmOcrTower(enc *vision.GlmOcrVisionEncoder) (*towers.GridTower, error) {
	return towers.NewGlmOcrTower(enc, towerPTX())
}

func newQwen25Tower(enc *vision.QwenVisionEncoder) (*towers.Qwen25Tower, error) {
	return towers.NewQwen25Tower(enc, towerPTX())
}

func newSiglipTower(enc *vision.Encoder) (*towers.SiglipTower, error) {
	return towers.NewSiglipTower(enc, towerPTX())
}

// newSiglipResident builds the CUDA SigLIP vision tower for vision.RegisterResident (wired in register.go):
// float32 weights take the tower base (siglip_vision.go), and int8 weights take NewVisionEncoder.
func newSiglipResident(e *vision.Encoder) (vision.ResidentEncoder, error) {
	if _, err := e.Weights(); err == nil {
		return newSiglipTower(e)
	}
	w, err := e.GPUWeights() // requires int8 weights (LoadEncoder quant=true)
	if err != nil {
		return nil, err
	}
	return NewVisionEncoder(w)
}

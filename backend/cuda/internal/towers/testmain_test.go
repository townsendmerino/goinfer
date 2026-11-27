//go:build cuda

package towers

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
)

var (
	testPTXOnce sync.Once
	cachedPTX   PTX
)

func testPTX() PTX {
	testPTXOnce.Do(func() {
		must := func(name string) []byte {
			b, err := os.ReadFile(filepath.Join("..", "..", "testdata", name))
			if err != nil {
				panic("towers testPTX: " + err.Error())
			}
			return b
		}
		cachedPTX = PTX{
			TowerBase:      must("tower_base.ptx"),
			AttnFusedVit:   must("attn_fused_vit.ptx"),
			AttnImgPrefill: must("attn_img_prefill.ptx"),
			AudioEncoder:   must("audio_encoder.ptx"),
			GeluQuant:      must("gelu_quant.ptx"),
			GemvW8Batched:  must("gemv_w8_batched.ptx"),
			LayernormQuant: must("layernorm_quant.ptx"),
			PrefillBatched: must("prefill_batched.ptx"),
		}
	})
	return cachedPTX
}

func newAudioTower(enc *audio.WhisperEncoder) (*AudioTower, error) {
	return NewAudioTower(enc, testPTX())
}

func newAudioTowerMode(enc *audio.WhisperEncoder, streamed bool) (*AudioTower, error) {
	return NewAudioTowerMode(enc, streamed, testPTX())
}

func newG4Tower(enc *vision.Gemma4Encoder) (*G4Tower, error) {
	return NewG4Tower(enc, testPTX())
}

func newQwen3Tower(enc *vision.Qwen3VisionEncoder) (*GridTower, error) {
	return NewQwen3Tower(enc, testPTX())
}

func newGlmOcrTower(enc *vision.GlmOcrVisionEncoder) (*GridTower, error) {
	return NewGlmOcrTower(enc, testPTX())
}

func newQwen25Tower(enc *vision.QwenVisionEncoder) (*Qwen25Tower, error) {
	return NewQwen25Tower(enc, testPTX())
}

func newSiglipTower(enc *vision.Encoder) (*SiglipTower, error) {
	return NewSiglipTower(enc, testPTX())
}

func TestMain(m *testing.M) {
	vision.RegisterResident(func(e *vision.Encoder) (vision.ResidentEncoder, error) {
		if _, err := e.Weights(); err == nil {
			return newSiglipTower(e)
		}
		w, err := e.GPUWeights()
		if err != nil {
			return nil, err
		}
		return NewVisionEncoder(w, testPTX())
	})
	vision.RegisterQwenResident(func(e *vision.QwenVisionEncoder) (vision.QwenResidentEncoder, error) {
		return newQwen25Tower(e)
	})
	os.Exit(m.Run())
}

func freeVRAM() (int64, bool) {
	dev, err := gpu.CreateSystemDefaultDevice()
	if err != nil {
		return 0, false
	}
	defer dev.ReleaseObjects()
	free, _, err := dev.Context().MemInfo()
	if err != nil {
		return 0, false
	}
	return int64(free), true
}

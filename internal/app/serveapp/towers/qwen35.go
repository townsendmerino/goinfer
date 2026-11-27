package towers

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Qwen3Tower lazily loads a Qwen3.5+ or Qwen3-VL vision encoder on first image request.
type Qwen3Tower struct {
	Dir    string
	MMProj bool
	Quant  bool
	Plan   GridTowerPlan
	once   sync.Once
	Enc    *vision.Qwen3VisionEncoder
	Acc    multimodal.GridTowerAccelerator
	fb     towerfit.DeviceFallback
	Err    error
	Deep   int
}

// Encoder loads and returns the underlying Qwen3VisionEncoder, initializing its device accelerator once.
func (q *Qwen3Tower) Encoder() (*vision.Qwen3VisionEncoder, error) {
	q.once.Do(func() {
		if q.MMProj {
			q.Enc, q.Err = vision.LoadQwen3VisionEncoderMMProj(q.Dir, q.Quant)
		} else {
			q.Enc, q.Err = vision.LoadQwen3VisionEncoder(q.Dir, q.Quant)
		}
		if q.Err != nil {
			q.Err = fmt.Errorf("load qwen3.5 vision tower (%s): %w", q.Dir, q.Err)
			return
		}
		if q.Plan.Device != "" {
			acc, err := multimodal.NewQwen3Tower(q.Plan.Device, q.Enc)
			if err == nil && q.Deep > 0 {
				if _, ok := acc.(multimodal.GridTowerTapper); !ok {
					_ = acc.Close()
					acc, err = nil, fmt.Errorf("the %s tower cannot tap the DeepStack blocks", q.Plan.Device)
				}
			}
			q.Acc, q.Err = q.Plan.Started("Qwen3.5", acc, err)
			if q.Acc != nil {
				ReleaseTowerBlocks("Qwen3.5", q.Enc)
			}
		}
	})
	return q.Enc, q.Err
}

// Features runs the Qwen3.5+/Qwen3-VL vision encoder and merger for one preprocessed image grid.
func (q *Qwen3Tower) Features(pv []float32, grid [3]int) ([]float32, error) {
	enc, err := q.Encoder()
	if err != nil {
		return nil, err
	}
	if q.Deep > 0 {
		return q.fb.Run("Qwen3-VL", q.Plan.Require, &q.Acc, func(acc multimodal.GridTowerAccelerator) ([]float32, error) {
			merged, deep, err := multimodal.Qwen3TowerFeaturesDeepstack(enc, acc, pv, [][3]int{grid})
			if err != nil {
				return nil, err
			}
			for _, d := range deep {
				merged = append(merged, d...)
			}
			return merged, nil
		})
	}
	return q.fb.Run("Qwen3.5", q.Plan.Require, &q.Acc, func(acc multimodal.GridTowerAccelerator) ([]float32, error) {
		return multimodal.Qwen3TowerFeatures(enc, acc, pv, [][3]int{grid})
	})
}

// Qwen3TowerFamily reports whether model_type mt uses Qwen3Tower and whether it carries DeepStack sets.
func Qwen3TowerFamily(mt string) (ok, deepstack bool) {
	switch mt {
	case "qwen3_5", "qwen3_5_moe":
		return true, false
	case "qwen3_vl", "qwen3_vl_moe":
		return true, true
	}
	return false, false
}

// IsQwen35VisionDir reports whether dir holds a usable Qwen3.5+ or Qwen3-VL vision checkpoint.
func IsQwen35VisionDir(dir string) bool {
	mt := towerfit.VisionModelType(dir)
	ok, deep := Qwen3TowerFamily(mt)
	if !ok {
		return false
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Vision *struct {
			Depth     int   `json:"depth"`
			Deepstack []int `json:"deepstack_visual_indexes"`
		} `json:"vision_config"`
	}
	if json.Unmarshal(raw, &c) != nil || c.Vision == nil || c.Vision.Depth <= 0 {
		return false
	}
	if deep != (len(c.Vision.Deepstack) > 0) {
		return false
	}
	_, err = multimodal.LoadQwen3PreprocessConfig(dir)
	return err == nil
}

// SetupQwen35Vision validates dir, reads and caps its Qwen3.5+ preprocessor config, and returns a lazy Qwen3Tower.
func SetupQwen35Vision(dir string, int8Tower bool) (*Qwen3Tower, multimodal.QwenPreprocessConfig, error) {
	if !IsQwen35VisionDir(dir) {
		return nil, multimodal.QwenPreprocessConfig{}, fmt.Errorf("%s is not a Qwen3.5+ checkpoint with a usable vision tower (needs vision_config with deepstack_visual_indexes [], and a preprocessor_config.json with size.shortest_edge/longest_edge)", dir)
	}
	pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
	if err != nil {
		return nil, pp, fmt.Errorf("qwen3.5 preprocessor config (%s): %w", dir, err)
	}
	if limit := towerfit.Qwen3MaxImageTokens * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit {
		pp.MaxPixels = limit
	}
	return &Qwen3Tower{Dir: dir, Quant: int8Tower, Deep: deepstackSets(dir)}, pp, nil
}

func deepstackSets(dir string) int {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return 0
	}
	var c struct {
		Vision struct {
			Deepstack []int `json:"deepstack_visual_indexes"`
		} `json:"vision_config"`
	}
	if json.Unmarshal(raw, &c) != nil {
		return 0
	}
	return len(c.Vision.Deepstack)
}

// SetupQwen35MMProj validates a GGUF mmproj path against modelType and textHidden and returns a lazy Qwen3Tower.
func SetupQwen35MMProj(path, modelType string, textHidden int, int8Tower bool) (*Qwen3Tower, multimodal.QwenPreprocessConfig, error) {
	if modelType != "qwen3_5" && modelType != "qwen3_5_moe" {
		return nil, multimodal.QwenPreprocessConfig{}, fmt.Errorf("-vision %s: a GGUF mmproj is supported for Qwen3.5+ models only; the model is %q", path, modelType)
	}
	vc, err := vision.ReadQwen3MMProjConfig(path)
	if err != nil {
		return nil, multimodal.QwenPreprocessConfig{}, fmt.Errorf("-vision %s: %w", path, err)
	}
	if vc.OutHiddenSize != textHidden {
		return nil, multimodal.QwenPreprocessConfig{}, fmt.Errorf("-vision %s: the projector emits %d wide and the model's hidden width is %d: this mmproj belongs to another model size", path, vc.OutHiddenSize, textHidden)
	}
	pp := multimodal.Qwen35FamilyPreprocessConfig()
	if limit := towerfit.Qwen3MaxImageTokens * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit {
		pp.MaxPixels = limit
	}
	return &Qwen3Tower{Dir: path, MMProj: true, Quant: int8Tower}, pp, nil
}

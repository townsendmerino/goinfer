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

// GlmOcrTower lazily loads a GLM-OCR vision encoder on first image request.
type GlmOcrTower struct {
	Dir   string
	Quant bool
	Plan  GridTowerPlan
	once  sync.Once
	Enc   *vision.GlmOcrVisionEncoder
	Acc   multimodal.GridTowerAccelerator
	fb    towerfit.DeviceFallback
	Err   error
}

// Encoder loads and returns the underlying GlmOcrVisionEncoder, initializing its device accelerator once.
func (g *GlmOcrTower) Encoder() (*vision.GlmOcrVisionEncoder, error) {
	g.once.Do(func() {
		g.Enc, g.Err = vision.LoadGlmOcrVisionEncoder(g.Dir, g.Quant)
		if g.Err != nil {
			g.Err = fmt.Errorf("load glm-ocr vision tower (%s): %w", g.Dir, g.Err)
			return
		}
		if g.Plan.Device != "" {
			acc, err := multimodal.NewGlmOcrTower(g.Plan.Device, g.Enc)
			g.Acc, g.Err = g.Plan.Started("GLM-OCR", acc, err)
			if g.Acc != nil {
				ReleaseTowerBlocks("GLM-OCR", g.Enc)
			}
		}
	})
	return g.Enc, g.Err
}

// Features runs the GLM-OCR vision encoder and tail merger for one preprocessed image grid.
func (g *GlmOcrTower) Features(pv []float32, grid [3]int) ([]float32, error) {
	enc, err := g.Encoder()
	if err != nil {
		return nil, err
	}
	return g.fb.Run("GLM-OCR", g.Plan.Require, &g.Acc, func(acc multimodal.GridTowerAccelerator) ([]float32, error) {
		return multimodal.GlmOcrTowerFeatures(enc, acc, pv, [][3]int{grid})
	})
}

// IsGlmOcrVisionDir reports whether dir holds a usable GLM-OCR vision checkpoint.
func IsGlmOcrVisionDir(dir string) bool {
	if towerfit.VisionModelType(dir) != "glm_ocr" {
		return false
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Vision *struct {
			Depth int `json:"depth"`
		} `json:"vision_config"`
	}
	if json.Unmarshal(raw, &c) != nil || c.Vision == nil || c.Vision.Depth <= 0 {
		return false
	}
	_, err = multimodal.LoadGlmOcrPreprocessConfig(dir)
	return err == nil
}

// SetupGlmOcrVision validates dir, loads and caps its GLM-OCR preprocessor config, and returns a lazy GlmOcrTower.
func SetupGlmOcrVision(dir string, int8Tower bool, maxPixels int) (*GlmOcrTower, multimodal.QwenPreprocessConfig, error) {
	if !IsGlmOcrVisionDir(dir) {
		return nil, multimodal.QwenPreprocessConfig{}, fmt.Errorf("%s is not a GLM-OCR checkpoint with a usable vision tower (needs model_type glm_ocr, a vision_config, and a preprocessor_config.json with size.shortest_edge/longest_edge)", dir)
	}
	pp, err := multimodal.LoadGlmOcrPreprocessConfig(dir)
	if err != nil {
		return nil, pp, fmt.Errorf("glm-ocr preprocessor config (%s): %w", dir, err)
	}
	if pp, err = multimodal.CapGlmOcrPixels(pp, maxPixels); err != nil {
		return nil, pp, fmt.Errorf("-vision-max-pixels: %w", err)
	}
	return &GlmOcrTower{Dir: dir, Quant: int8Tower}, pp, nil
}

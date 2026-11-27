package serveapp

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
	"github.com/townsendmerino/goinfer/multimodal"
)

func (s *server) loadQwen35VisionTower(dir string, int8Tower bool, backend string, require bool) error {
	tower, pp, err := towers.SetupQwen35Vision(dir, int8Tower)
	if err != nil {
		return err
	}
	if tower.Plan, err = towers.PlanGridTower("Qwen3.5", multimodal.Qwen3Towers(), int8Tower, backend, require); err != nil {
		return err
	}
	return s.attachQwen35Tower(tower, pp, dir)
}

func (s *server) loadQwen35MMProj(path string, int8Tower bool, backend string, require bool) error {
	for _, lm := range s.models {
		c := lm.model.Config()
		tower, pp, err := towers.SetupQwen35MMProj(path, c.ModelType, c.HiddenDim, int8Tower)
		if err != nil {
			return err
		}
		if tower.Plan, err = towers.PlanGridTower("Qwen3.5", multimodal.Qwen3Towers(), int8Tower, backend, require); err != nil {
			return err
		}
		return s.attachQwen35Tower(tower, pp, path)
	}
	return fmt.Errorf("-vision %s: no model is loaded", path)
}

func (s *server) attachQwen35Tower(tower *qwen3Tower, pp multimodal.QwenPreprocessConfig, dir string) error {
	for _, lm := range s.models {
		q := &qwen3Vision{QwenGrid: qwenGrid{PP: pp, Merge: pp.MergeSize, ImgTok: -1}, Qwen3Tower: tower}
		lm.vision = q
		if id, ok := lm.tk.TokenID(multimodal.QwenImagePad); ok {
			q.ImgTok = id
		}
		if q.ImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.QwenImagePad)
		}
		family := "Qwen3.5"
		if tower.Deep > 0 {
			family = "Qwen3-VL"
		}
		fmt.Fprintf(os.Stderr, "%s vision for %q: tower (%s) loads on first image (merge %d, image-pad id %d, <= %d tokens/image) from %s\n",
			family, lm.name, tower.Plan.Where, q.Merge, q.ImgTok, towerfit.Qwen3MaxImageTokens, dir)
	}
	return nil
}

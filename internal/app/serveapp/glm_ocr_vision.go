package serveapp

import (
	"fmt"
	"os"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
	"github.com/townsendmerino/goinfer/multimodal"
)

func (s *server) loadGlmOcrVisionTower(dir string, int8Tower bool, maxPixels int, backend string, require bool) error {
	tower, pp, err := towers.SetupGlmOcrVision(dir, int8Tower, maxPixels)
	if err != nil {
		return err
	}
	if tower.Plan, err = towers.PlanGridTower("GLM-OCR", multimodal.GlmOcrTowers(), int8Tower, backend, require); err != nil {
		return err
	}
	for _, lm := range s.models {
		g := &glmVision{QwenGrid: qwenGrid{PP: pp, Merge: pp.MergeSize, ImgTok: -1}, GlmOcrTower: tower}
		lm.vision = g
		if id, ok := lm.tk.TokenID(multimodal.GlmOcrImagePad); ok {
			g.ImgTok = id
		}
		if g.ImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.GlmOcrImagePad)
		}
		q := "f32"
		if int8Tower {
			q = "int8"
		}
		fmt.Fprintf(os.Stderr, "GLM-OCR vision for %q: tower (%s, %s) loads on first image (merge %d, image id %d, pixel budget %d..%d px = at most %d image tokens) from %s\n",
			lm.name, q, tower.Plan.Where, g.Merge, g.ImgTok, pp.MinPixels, pp.MaxPixels, pp.MaxPixels/(pp.PatchSize*pp.PatchSize*pp.MergeSize*pp.MergeSize), dir)
	}
	return nil
}

func (lm *loadedModel) imageFitsContext(n int, grid [3]int) error {
	if lm.model == nil {
		return nil
	}
	ctx := lm.contextWindow(false)
	g := lm.qwenGrid()
	if ctx > 0 && n+24 >= ctx {
		return fmt.Errorf("image_too_large_for_context: this image is %d image tokens (a %dx%d-patch grid) but the context window is %d tokens; "+
			"send a smaller image (the model's own ceiling is %d tokens, 4.8 MP) or raise the context", n, grid[2], grid[1], ctx,
			g.PP.MaxPixels/(g.PP.PatchSize*g.PP.PatchSize*g.Merge*g.Merge))
	}
	return nil
}

func (lm *loadedModel) glmOcrExtractionTurn(rf *respFormat, turns []chat.Turn) error {
	if lm.glmOcr() == nil || rf == nil || rf.Type != "json_schema" || rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
		return nil
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return nil
	}
	tmpl, err := constrain.TemplateFromSchema(rf.JSONSchema.Schema)
	if err != nil {
		return fmt.Errorf("response_format json_schema: cannot build the extraction template: %w", err)
	}
	turns[idx].Content, _ = multimodal.GlmOcrExtractionText(turns[idx].Content, tmpl)
	return nil
}

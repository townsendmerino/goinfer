package serveapp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/multimodal"
)

// GLM-OCR image serving (O3, docs/tasks/task-glm-ocr-2026-10.md). The route is the Qwen2.5-VL / Qwen3.5+ one —
// preprocess -> tower -> GenerateQwenVL with m-RoPE, the merged tower rows replacing the <|image|> run — with these
// differences, all of which live here or in the loadedModel.glm branches of vision_serve.go:
//   - the tower is aikit's GlmOcrVisionEncoder, loaded on first use (like qwen3Tower);
//   - the preprocessing config halves the file's pixel bounds (multimodal.LoadGlmOcrPreprocessConfig);
//   - the image block and the template are GLM's, image first (multimodal.GlmOcrImageBlock, chat.GlmOCR);
//   - NO per-image token cap here, unlike Qwen3.5's qwen3MaxImageTokens: the processor's own ceiling is 6,144 tokens
//     (4.82 MP) and the serve-side default is O4's decision (the owner picks it), so serve accepts what the processor
//     does and an image that does not fit the context is refused by name (imageFitsContext), never truncated.
//
// The tower stays f32 unless -vision-quant int8 is given: serve's usual "a GPU backend implies an int8 tower" rule is for
// the resident GPU encoders. Under --backend metal the f32 tower runs on Metal (S2, planGridTower); its int8 form is not
// gated on the real checkpoint.

// glmOcrTower is a GLM-OCR vision tower that loads on first use; safe for concurrent callers, a failed load is
// remembered rather than retried on every request.
type glmOcrTower struct {
	dir   string
	quant bool
	plan  gridTowerPlan // where the tower runs (S2): the device it is built on at first use
	once  sync.Once
	enc   *vision.GlmOcrVisionEncoder
	acc   multimodal.GridTowerAccelerator // nil: aikit's CPU tower
	err   error
}

func (g *glmOcrTower) encoder() (*vision.GlmOcrVisionEncoder, error) {
	g.once.Do(func() {
		g.enc, g.err = vision.LoadGlmOcrVisionEncoder(g.dir, g.quant)
		if g.err != nil {
			g.err = fmt.Errorf("load glm-ocr vision tower (%s): %w", g.dir, g.err)
			return
		}
		if g.plan.device != "" {
			acc, err := multimodal.NewGlmOcrTower(g.plan.device, g.enc)
			g.acc, g.err = g.plan.started("GLM-OCR", acc, err)
		}
	})
	return g.enc, g.err
}

// features runs the tower (on its device when the plan put it there) and aikit's tail: the merged image embeddings.
func (g *glmOcrTower) features(pv []float32, grid [3]int) ([]float32, error) {
	enc, err := g.encoder()
	if err != nil {
		return nil, err
	}
	return multimodal.GlmOcrTowerFeatures(enc, g.acc, pv, [][3]int{grid})
}

// isGlmOcrVisionDir reports whether dir is a GLM-OCR checkpoint that carries a usable vision tower: model_type glm_ocr, a
// non-empty vision_config, and a preprocessor config LoadGlmOcrPreprocessConfig accepts. AUTO-discovery only: a stripped
// text-only copy is simply not a vision model, not an error.
func isGlmOcrVisionDir(dir string) bool {
	if visionModelType(dir) != "glm_ocr" {
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

// setupGlmOcrVision does everything about a GLM-OCR tower that can be decided without the weights: validates the
// directory, reads the (halved) preprocessor config, and returns the not-yet-loaded tower. Split from
// loadGlmOcrVisionTower so the config and the laziness are testable without a tokenizer.
func setupGlmOcrVision(dir string, int8Tower bool, maxPixels int) (*glmOcrTower, multimodal.QwenPreprocessConfig, error) {
	if !isGlmOcrVisionDir(dir) {
		return nil, multimodal.QwenPreprocessConfig{}, fmt.Errorf("%s is not a GLM-OCR checkpoint with a usable vision tower (needs model_type glm_ocr, a vision_config, and a preprocessor_config.json with size.shortest_edge/longest_edge)", dir)
	}
	pp, err := multimodal.LoadGlmOcrPreprocessConfig(dir)
	if err != nil {
		return nil, pp, fmt.Errorf("glm-ocr preprocessor config (%s): %w", dir, err)
	}
	// -vision-max-pixels lowers the budget (the model's own ceiling is the default and the most it accepts), because the tower is CPU f32 and its cost is
	// superlinear in the image (docs/measurements/glm-ocr-tower-cost-2026-10.md).
	if pp, err = multimodal.CapGlmOcrPixels(pp, maxPixels); err != nil {
		return nil, pp, fmt.Errorf("-vision-max-pixels: %w", err)
	}
	return &glmOcrTower{dir: dir, quant: int8Tower}, pp, nil
}

// loadGlmOcrVisionTower attaches a GLM-OCR tower to the single loaded model. Everything checkable without the tower's
// weights is checked here, at startup; the weights load on first image.
func (s *server) loadGlmOcrVisionTower(dir string, int8Tower bool, maxPixels int, backend string, require bool) error {
	tower, pp, err := setupGlmOcrVision(dir, int8Tower, maxPixels)
	if err != nil {
		return err
	}
	if tower.plan, err = planGridTower("GLM-OCR", multimodal.GlmOcrTowers(), int8Tower, backend, require); err != nil {
		return err
	}
	for _, lm := range s.models {
		lm.glm = tower
		lm.qwenPP = pp
		lm.qwenMerge = pp.MergeSize
		lm.qwenImgTok = -1
		if id, ok := lm.tk.TokenID(multimodal.GlmOcrImagePad); ok {
			lm.qwenImgTok = id
		}
		if lm.qwenImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.GlmOcrImagePad)
		}
		q := "f32"
		if int8Tower {
			q = "int8"
		}
		fmt.Fprintf(os.Stderr, "GLM-OCR vision for %q: tower (%s, %s) loads on first image (merge %d, image id %d, pixel budget %d..%d px = at most %d image tokens) from %s\n",
			lm.name, q, tower.plan.where, lm.qwenMerge, lm.qwenImgTok, pp.MinPixels, pp.MaxPixels, pp.MaxPixels/(pp.PatchSize*pp.PatchSize*pp.MergeSize*pp.MergeSize), dir)
	}
	return nil
}

// imageFitsContext refuses, by name, an image whose token run (plus the text around it) cannot fit the context this
// request will run in, BEFORE the tower runs (it is minutes on a CPU). Without it the same request would fail later and
// less specifically in prepare ("prompt is N tokens but the model's context window is C"), after the image was resized;
// the point here is the remedy, and that an image is never silently truncated to fit.
func (lm *loadedModel) imageFitsContext(n int, grid [3]int) error {
	if lm.model == nil {
		return nil
	}
	ctx := lm.contextWindow(false)
	// The 24 covers the template's own tokens and a one-line task prompt; prepare's exact check still runs afterwards.
	if ctx > 0 && n+24 >= ctx {
		return fmt.Errorf("image_too_large_for_context: this image is %d image tokens (a %dx%d-patch grid) but the context window is %d tokens; "+
			"send a smaller image (the model's own ceiling is %d tokens, 4.8 MP) or raise the context", n, grid[2], grid[1], ctx,
			lm.qwenPP.MaxPixels/(lm.qwenPP.PatchSize*lm.qwenPP.PatchSize*lm.qwenMerge*lm.qwenMerge))
	}
	return nil
}

// glmOcrExtractionTurn is O5 (docs/tasks/task-glm-ocr-2026-10.md): a GLM-OCR image request that carries response_format
// json_schema is an EXTRACTION request. The model's extraction prompt is a JSON template (an object of empty values), not a
// schema, so the template is built from the request's own schema (constrain.TemplateFromSchema: the same document the
// grammar is compiled from, so the two cannot disagree) and becomes the last user turn's text, with the card's instruction in
// front of it. The grammar itself is installed by prepare, exactly as on the text route (the vision route has always passed
// the request's sampling, response_format included, through prepare and driveVL).
//
// The rule (multimodal.GlmOcrExtractionText): the template prompt REPLACES the user's text only when that text is empty or a
// bare task prompt ("Text Recognition:", "Table Recognition:", "Formula Recognition:"); any other text is the user's own prompt
// and is sent unchanged, still under the grammar. json_object (no schema) has no template to build and is left alone.
func (lm *loadedModel) glmOcrExtractionTurn(rf *respFormat, turns []chat.Turn) error {
	if lm.glm == nil || rf == nil || rf.Type != "json_schema" || rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
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

// gridTowerPlan is where a Qwen3.5+ or GLM-OCR tower will run (S2 of docs/tasks/task-multimodal-support-2026-10.md):
// device is the registered accelerator to build at first use ("" = aikit's CPU tower), where the banner's word for it.
type gridTowerPlan struct {
	device, where string
	require       bool
}

// planGridTower decides at startup, before the tower's weights load, where a grid tower runs, by the rules
// chooseGemma4Tower uses: the Metal tower under --backend metal when this binary registers one and the tower is
// float32; the CPU otherwise, with the reason. -require-backend turns each CPU fallback into a refusal.
func planGridTower(family string, registered []string, int8Tower bool, backend string, require bool) (gridTowerPlan, error) {
	if backend != "metal" {
		return gridTowerPlan{where: "CPU"}, nil
	}
	if int8Tower {
		if require {
			return gridTowerPlan{}, fmt.Errorf("-require-backend: the %s Metal tower is float32; -vision-quant int8 keeps it on the CPU", family)
		}
		return gridTowerPlan{where: "CPU (-vision-quant int8; the Metal tower is float32)"}, nil
	}
	if !slices.Contains(registered, "metal") {
		if require {
			return gridTowerPlan{}, fmt.Errorf("-require-backend: this binary has no Metal %s tower", family)
		}
		return gridTowerPlan{where: "CPU (no Metal tower in this binary)"}, nil
	}
	return gridTowerPlan{device: "metal", where: "Metal", require: require}, nil
}

// started settles the plan once the tower's weights are loaded: the accelerator, or the CPU with a note when the device
// declined (an error under -require-backend).
func (p *gridTowerPlan) started(family string, acc multimodal.GridTowerAccelerator, err error) (multimodal.GridTowerAccelerator, error) {
	if err == nil {
		return acc, nil
	}
	if p.require {
		return nil, fmt.Errorf("-require-backend: the %s tower could not start on %s: %w", family, p.device, err)
	}
	fmt.Fprintf(os.Stderr, "vision: the %s tower runs on the CPU: %s declined it: %v\n", family, p.device, err)
	p.where = "CPU (" + p.device + " declined)"
	return nil, nil
}

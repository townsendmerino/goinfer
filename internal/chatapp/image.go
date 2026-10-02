package chatapp

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// --image (O5, docs/tasks/task-glm-ocr-2026-10.md): one image, one answer, in the terminal.
//
//	goinfer-chat --model ~/models/glm-ocr --image invoice.png --schema invoice.schema.json
//
// goinfer-chat is in-process, so this runs the same pipeline goinfer-serve's GLM-OCR image route runs, minus the HTTP:
// preprocess (the checkpoint's pixel budget, halved) -> the vision tower (f32, on the CPU) -> GenerateQwenVL, with the
// image block spliced into the glm_ocr chat template as SPECIAL text (multimodal.SpliceImageBlock, the helper serve uses).
// GLM-OCR is the only vision family goinfer-chat carries; goinfer-serve handles the others.
//
// With --schema the user turn is the model's EXTRACTION prompt: the card's instruction plus the schema's JSON template
// (constrain.TemplateFromSchema), and the schema's grammar constrains the reply. The same replace-only-if-bare rule serve
// applies (multimodal.GlmOcrExtractionText) decides whether -p text is kept: -p "Table Recognition:" or no -p at all is
// replaced by the template prompt; any other -p text is the user's own prompt and is sent as written.

// imageInput is an image attached to the session, preprocessed once at startup (so a bad file fails before the model loads).
type imageInput struct {
	raw    []byte
	dir    string
	pp     multimodal.QwenPreprocessConfig
	pixels []float32
	grid   [3]int
	nImg   int
	tower  *vision.GlmOcrVisionEncoder // loaded on first use
	// towerTime is how long the tower took (load excluded), so the reply's tok/s can exclude it.
	towerTime time.Duration
}

// Defaults an --image run applies when the flag was not given explicitly: OCR is run greedy (the checkpoint's own
// generation_config is do_sample false), and an extraction reply is a few hundred tokens, more than the REPL's 512 when an
// invoice has many lines.
const (
	imageDefaultTemp = 0.0
	imageDefaultMax  = 2048
)

// checkImageFlags is the flag-level validation, pure so it is testable without a model: --image is one image for one answer.
func checkImageFlags(image, model, batch string) error {
	if image == "" {
		return nil
	}
	if batch != "" {
		return fmt.Errorf("--image attaches one image to one answer; it cannot be combined with --batch (put image parts in a batch line's own messages through goinfer-serve)")
	}
	if model == "" {
		return fmt.Errorf("--image needs --model <a GLM-OCR checkpoint directory> (zai-org/GLM-OCR)")
	}
	return nil
}

// isGlmOcrDir reports whether dir is a GLM-OCR checkpoint (model_type glm_ocr with a vision_config and a preprocessor config).
func isGlmOcrDir(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	s := string(raw)
	if !strings.Contains(s, `"model_type": "glm_ocr"`) && !strings.Contains(s, `"model_type":"glm_ocr"`) {
		return false
	}
	_, err = multimodal.LoadGlmOcrPreprocessConfig(dir)
	return err == nil && strings.Contains(s, `"vision_config"`)
}

// loadImageInput reads and preprocesses the image for the GLM-OCR checkpoint at dir. Everything that can fail cheaply
// fails here, before the multi-second model load.
func loadImageInput(dir, imagePath string) (*imageInput, error) {
	fi, err := os.Stat(dir)
	if err != nil || !fi.IsDir() || !isGlmOcrDir(dir) {
		return nil, fmt.Errorf("--image: %q is not a GLM-OCR checkpoint directory (model_type glm_ocr with its vision tower and preprocessor_config.json); goinfer-chat's image input is GLM-OCR only, goinfer-serve handles the other vision families", dir)
	}
	raw, err := os.ReadFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("--image: %w", err)
	}
	pp, err := multimodal.LoadGlmOcrPreprocessConfig(dir)
	if err != nil {
		return nil, fmt.Errorf("--image: %w", err)
	}
	pixels, grid, err := multimodal.QwenPreprocess(raw, pp)
	if err != nil {
		return nil, fmt.Errorf("--image %s: %w", imagePath, err)
	}
	return &imageInput{raw: raw, dir: dir, pp: pp, pixels: pixels, grid: grid, nImg: multimodal.QwenMergedTokens(grid, pp.MergeSize)}, nil
}

// features runs the tower (f32, CPU) over the preprocessed image; loads it on first use.
func (im *imageInput) features() ([]float32, error) {
	if im.tower == nil {
		progress("loading the GLM-OCR vision tower…")
		tw, err := vision.LoadGlmOcrVisionEncoder(im.dir, false)
		if err != nil {
			return nil, fmt.Errorf("load glm-ocr vision tower (%s): %w", im.dir, err)
		}
		im.tower = tw
	}
	t0 := time.Now()
	progress(fmt.Sprintf("reading the image: %d image tokens through the vision tower (CPU, f32; about a minute per megapixel)…", im.nImg))
	feats, err := im.tower.Forward(im.pixels, [][3]int{im.grid})
	if err == nil {
		im.towerTime = time.Since(t0)
		progress(fmt.Sprintf("vision tower done in %s", im.towerTime.Round(time.Second)))
	}
	return feats, err
}

// imageTurns is the history this request sends: the last user turn's text after the extraction rule. With a schema the text is
// replaced by the template prompt only when it is empty or a bare task prompt; with no schema an empty text becomes plain
// text recognition (what serve does for an image with no text part).
func (s *session) imageTurns(turns []chat.Turn) ([]chat.Turn, error) {
	out := append([]chat.Turn(nil), turns...)
	idx := -1
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Role == "user" {
			idx = i
			break
		}
	}
	if idx < 0 {
		out = append(out, chat.Turn{Role: "user"})
		idx = len(out) - 1
	}
	if s.schema != nil {
		tmpl, err := constrain.TemplateFromSchema(s.schema)
		if err != nil {
			return nil, fmt.Errorf("--schema: cannot build the extraction template: %w", err)
		}
		out[idx].Content, _ = multimodal.GlmOcrExtractionText(out[idx].Content, tmpl)
	} else if strings.TrimSpace(out[idx].Content) == "" {
		out[idx].Content = multimodal.GlmOcrDefaultPrompt
	}
	return out, nil
}

// imageSystem is the system prompt an image request sends: none, unless the user asked for one. The REPL's default is a
// coding-assistant prompt, and an OCR model is prompted with a task string alone.
func (s *session) imageSystem() string {
	if sys := strings.TrimSpace(s.system); sys != defaultSystem {
		return sys
	}
	return ""
}

// imagePrompt builds the ids of the image prompt: the glm_ocr template with the user turn's text after the extraction rule
// (imageTurns) behind the image block, the block spliced in as SPECIAL text so its sentinels become real tokens. It returns
// the ids, where the image run sits in them, and the <|image|> token id.
func (s *session) imagePrompt(turns []chat.Turn) (ids []int, imgPos, imgLen, imageToken int, err error) {
	im := s.img
	if s.tmpl == nil || s.tmpl.Name() != "glm_ocr" {
		return nil, 0, 0, 0, fmt.Errorf("--image: the loaded model's chat template is %v, not glm_ocr", s.tmpl)
	}
	imageToken, ok := s.tk.TokenID(multimodal.GlmOcrImagePad)
	if !ok {
		return nil, 0, 0, 0, fmt.Errorf("--image: the tokenizer has no %s token", multimodal.GlmOcrImagePad)
	}
	turns, err = s.imageTurns(turns)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	block := multimodal.GlmOcrImageBlock(im.nImg)
	for i := len(turns) - 1; i >= 0; i-- {
		if turns[i].Role == "user" {
			turns[i].Content = block + turns[i].Content
			break
		}
	}
	segs, err := multimodal.SpliceImageBlock(s.tmpl.RenderSegments(s.imageSystem(), turns), block)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	ids, err = s.tk.EncodeSegments(segs, false)
	if err != nil {
		return nil, 0, 0, 0, fmt.Errorf("encode: %w", err)
	}
	imgPos, imgLen = multimodal.FindImageRun(ids, imageToken)
	if imgLen != im.nImg {
		return nil, 0, 0, 0, fmt.Errorf("image placeholder run = %d image tokens, want %d (template mismatch)", imgLen, im.nImg)
	}
	return ids, imgPos, imgLen, imageToken, nil
}

// startImageGen builds the image prompt and starts the generation through decoder.GenerateQwenVL.
func (s *session) startImageGen(ctx context.Context, turns []chat.Turn, maxTok int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, error) {
	im := s.img
	ids, imgPos, imgLen, imageToken, err := s.imagePrompt(turns)
	if err != nil {
		return nil, nil, err
	}
	if mp := s.model.Config().MaxPositions; mp > 0 && len(ids)+1 >= mp {
		return nil, nil, fmt.Errorf("image_too_large_for_context: the prompt is %d tokens (%d of them the image) and the model's context is %d; use a smaller image", len(ids), im.nImg, mp)
	}
	stream, gen := s.model.GenerateQwenVL(ctx, ids, imgPos, imgLen, multimodal.HashImageBytes(im.raw), im.features,
		[][3]int{im.grid}, im.pp.MergeSize, imageToken, maxTok, sp)
	return stream, gen, nil
}

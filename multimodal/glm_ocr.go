package multimodal

import (
	"fmt"
	"strings"
)

// GLM-OCR (zai-org/GLM-OCR, model_type glm_ocr) image plumbing: the prompt block, the task prompts and the preprocessing config. The
// pixel path itself is QwenPreprocess unchanged (GLM-OCR's Glm46VImageProcessor is the same smart-resize + CLIP-normalize +
// merge-block patchify the Qwen towers take; docs/tasks/task-glm-ocr-2026-10.md), so only the config differs.

// GLM-OCR image-block sentinels. The checkpoint's chat template renders an image as
// "<|begin_of_image|><|image|><|end_of_image|>" and the processor expands the single <|image|> to n of them
// (n = t·h·w / merge²). Unlike Qwen's, the image comes FIRST in the user turn and the task prompt follows it with no
// newline between (`...<|end_of_image|>Text Recognition:`). <|begin_of_image|>/<|end_of_image|> are TEXT positions for
// m-RoPE; only <|image|> (59280) is an image position.
const (
	GlmOcrImageStart = "<|begin_of_image|>"
	GlmOcrImagePad   = "<|image|>"
	GlmOcrImageEnd   = "<|end_of_image|>"
)

// GlmOcrImageBlock is the placeholder for one image: the start/end sentinels around n image-pad placeholders.
func GlmOcrImageBlock(n int) string {
	return GlmOcrImageStart + strings.Repeat(GlmOcrImagePad, n) + GlmOcrImageEnd
}

// The task prompts the model card lists. The user turn is the image followed by exactly one of these. The information-extraction
// prompt is a JSON template, not one of these (GlmOcrExtractionInstruction).
const (
	GlmOcrPromptText    = "Text Recognition:"
	GlmOcrPromptFormula = "Formula Recognition:"
	GlmOcrPromptTable   = "Table Recognition:"
)

// GlmOcrDefaultPrompt is what an image request with no text part gets: plain text recognition.
const GlmOcrDefaultPrompt = GlmOcrPromptText

// GlmOcrExtractionInstruction opens the model's information-extraction prompt (the model card's own, verbatim: "output the information
// in the image in the following JSON format"). It is followed by a newline and a JSON TEMPLATE, an object whose values are empty
// strings, NOT a JSON Schema (docs/tasks/task-glm-ocr-2026-10.md); build the template with constrain.TemplateFromSchema /
// TemplateFromStruct so the prompt and the grammar come from one source.
const GlmOcrExtractionInstruction = "请按下列JSON格式输出图中信息:"

// GlmOcrExtractionPrompt is the user-turn text for extraction: the instruction, a newline, the template.
func GlmOcrExtractionPrompt(template string) string {
	return GlmOcrExtractionInstruction + "\n" + template
}

// GlmOcrExtractionText applies the extraction-prompt rule for a request that asks for schema-bound output ON AN IMAGE: the template
// prompt REPLACES the text the user sent only when that text is empty (nothing but whitespace) or is exactly one of the three bare
// task prompts (a client that always sends "Text Recognition:" next to a schema means "read the image into this schema", and the task
// prompt would contradict the grammar). Any other text is the user's own prompt (their own extraction prompt, a question, a different
// instruction) and is used UNCHANGED: the grammar still constrains the reply, but the user owns what the model is told. replaced
// reports which happened.
func GlmOcrExtractionText(userText, template string) (text string, replaced bool) {
	switch strings.TrimSpace(userText) {
	case "", GlmOcrPromptText, GlmOcrPromptFormula, GlmOcrPromptTable:
		return GlmOcrExtractionPrompt(template), true
	}
	return userText, false
}

// LoadGlmOcrPreprocessConfig reads a GLM-OCR checkpoint's preprocessor_config.json into the QwenPreprocessConfig that QwenPreprocess takes.
//
// THE PIXEL BOUNDS ARE HALVED, ON PURPOSE. The file says size.shortest_edge 12544 and size.longest_edge 9633792, and
// LoadQwen3PreprocessConfig would pass them through as MinPixels/MaxPixels. But Glm46VImageProcessor calls smart_resize with
// num_frames = temporal_patch_size = 2 and tests t_bar·h_bar·w_bar against the bounds, so they bound TWO frames of a still image: one
// frame's budget is bound / temporal_patch_size (6,272 .. 4,816,896 px, i.e. at most 24,576 patches and 6,144 image tokens, not the
// 12,000 the unhalved bound would allow). Passing the file's values unhalved silently doubles the budget and gives a larger grid than
// HF's for any image over ~4.8 MP. The division is by the file's own temporal_patch_size, not a literal 2.
// TestLoadGlmOcrPreprocessConfig_halvesPixelBounds pins this and fails if the unhalved values come through.
//
// Everything else is LoadQwen3PreprocessConfig's: patch 14, merge 2, temporal 2, CLIP mean/std, and the required-field refusals (a
// file missing the size keys is an error, not a default).
func LoadGlmOcrPreprocessConfig(dir string) (QwenPreprocessConfig, error) {
	cfg, err := LoadQwen3PreprocessConfig(dir)
	if err != nil {
		return cfg, fmt.Errorf("multimodal(glm_ocr): %w", err)
	}
	return halveForTemporalFrames(cfg), nil
}

// halveForTemporalFrames converts pixel bounds that apply to TemporalPatchSize stacked frames into one frame's bounds.
func halveForTemporalFrames(cfg QwenPreprocessConfig) QwenPreprocessConfig {
	cfg.MinPixels /= cfg.TemporalPatchSize
	cfg.MaxPixels /= cfg.TemporalPatchSize
	return cfg
}

// CapGlmOcrPixels lowers the pixel budget to maxPixels (a pixel count, one frame). It never raises it: the checkpoint's own
// ceiling (4.82 MP, 6,144 image tokens) is the most the tower and the context were gated for. maxPixels <= 0 leaves the config
// alone; a cap below the config's own floor is an error rather than a silent clamp, because the resize could not honour both.
func CapGlmOcrPixels(cfg QwenPreprocessConfig, maxPixels int) (QwenPreprocessConfig, error) {
	if maxPixels <= 0 || maxPixels >= cfg.MaxPixels {
		return cfg, nil
	}
	if maxPixels < cfg.MinPixels {
		return cfg, fmt.Errorf("multimodal(glm_ocr): a pixel cap of %d is below the model's own minimum of %d pixels", maxPixels, cfg.MinPixels)
	}
	cfg.MaxPixels = maxPixels
	return cfg, nil
}

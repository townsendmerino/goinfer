package multimodal

import (
	"fmt"
	"strings"
)

// GLM-OCR (zai-org/GLM-OCR, model_type glm_ocr) image plumbing: the prompt block, the task prompts and the
// preprocessing config. The pixel path itself is QwenPreprocess unchanged — GLM-OCR's Glm46VImageProcessor is the
// same smart-resize + CLIP-normalize + merge-block patchify the Qwen towers take (docs/tasks/task-glm-ocr-2026-10.md
// O0), so only the config differs.

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

// The task prompts the model card lists. The user turn is the image followed by exactly one of these (the card's three;
// Ollama's page also names "Figure Recognition:", which the card does not). The information-extraction prompt is a JSON
// template, not one of these (task O5).
const (
	GlmOcrPromptText    = "Text Recognition:"
	GlmOcrPromptFormula = "Formula Recognition:"
	GlmOcrPromptTable   = "Table Recognition:"
)

// GlmOcrDefaultPrompt is what an image request with no text part gets: plain text recognition.
const GlmOcrDefaultPrompt = GlmOcrPromptText

// LoadGlmOcrPreprocessConfig reads a GLM-OCR checkpoint's preprocessor_config.json into the QwenPreprocessConfig that
// QwenPreprocess takes.
//
// THE PIXEL BOUNDS ARE HALVED, ON PURPOSE. The file says size.shortest_edge 12544 and size.longest_edge 9633792, and
// LoadQwen3PreprocessConfig would pass them through as MinPixels/MaxPixels. But Glm46VImageProcessor calls smart_resize
// with num_frames = temporal_patch_size = 2 and tests t_bar·h_bar·w_bar against the bounds, so they bound TWO frames of a
// still image: one frame's budget is bound / temporal_patch_size (6,272 .. 4,816,896 px, i.e. at most 24,576 patches and
// 6,144 image tokens, not the 12,000 the unhalved bound would allow). Passing the file's values unhalved silently doubles
// the budget and gives a larger grid than HF's for any image over ~4.8 MP (task doc §1 correction, O0). The division is by
// the file's own temporal_patch_size, not a literal 2. TestLoadGlmOcrPreprocessConfig_halvesPixelBounds pins this and
// fails if the unhalved values come through.
//
// Everything else is LoadQwen3PreprocessConfig's: patch 14, merge 2, temporal 2, CLIP mean/std, and the required-field
// refusals (a file missing the size keys is an error, not a default).
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

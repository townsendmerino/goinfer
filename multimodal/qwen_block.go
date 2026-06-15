package multimodal

import "strings"

// Qwen2.5-VL image-block sentinels. An image is rendered into the prompt as
// VisionStart + n × ImagePad + VisionEnd; the tokenizer maps ImagePad to the
// image_token_id (151655) run that the decoder's embed-by-vector seam
// (decoder.GenerateQwenVL / prefillLogitsQwenVL) replaces with the ViT+merger
// features. n = grid_h*grid_w / spatial_merge_size² for one image. Shared so the
// placeholder is assembled identically wherever a Qwen image turn is built.
const (
	QwenImagePad    = "<|image_pad|>"
	QwenVisionStart = "<|vision_start|>"
	QwenVisionEnd   = "<|vision_end|>"
)

// QwenImageBlock returns the placeholder string for one image: the vision-start/end
// sentinels wrapping n image-pad placeholders (n = merged-patch count). The caller
// renders this inside a user turn; the tokenizer turns the pad run into
// image-token ids that the embed-by-vector forward overrides.
func QwenImageBlock(n int) string {
	return QwenVisionStart + strings.Repeat(QwenImagePad, n) + QwenVisionEnd
}

// QwenMergedTokens returns the number of decoder image tokens (merged patches) for
// a grid (t,h,w in patch units) at the given spatial_merge_size — t*h*w / merge².
func QwenMergedTokens(grid [3]int, merge int) int {
	return grid[0] * (grid[1] / merge) * (grid[2] / merge)
}

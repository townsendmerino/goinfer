package multimodal

import "strings"

// Gemma 4 image-block sentinels: the real tokenizer's literal strings (google/gemma-4-E2B-it's tokenizer.json added_tokens and
// tokenizer_config.json boi_token/image_token/eoi_token), with irregular bracket placement, not the symmetric "<start_of_image>" shape
// Gemma 3 uses. An image is rendered into the prompt as Gemma4ImageBlockStart + n × Gemma4ImageSoftToken + Gemma4ImageBlockEnd; the
// tokenizer maps Gemma4ImageSoftToken to a run of ids that the decoder's embed-by-vector seam (decoder.GenerateGemma4VL /
// prefillLogitsGemma4VL) replaces with the projected vision features.
const (
	Gemma4ImageSoftToken  = "<|image|>"
	Gemma4ImageBlockStart = "<|image>"
	Gemma4ImageBlockEnd   = "<image|>"
)

// Gemma4ImageBlock returns the placeholder string for one image: the BOI/EOI
// sentinels wrapping n soft-token placeholders. n is DATA-DEPENDENT (computed
// per image from its own preprocessed patch grid via Gemma4PooledTokens),
// unlike Gemma 3's fixed Projector.MMTokens() — Gemma4AspectRatioSize floors
// each axis to a pooling_kernel_size*patch_size multiple, so the pooled count
// is <= the checkpoint's vision_soft_tokens_per_image budget, not equal to it
// in general.
func Gemma4ImageBlock(n int) string {
	return Gemma4ImageBlockStart + strings.Repeat(Gemma4ImageSoftToken, n) + Gemma4ImageBlockEnd
}

// Gemma4PooledTokens returns the number of pooled soft-token embeddings vision.Gemma4Encoder.Forward will produce for a patch grid
// described by positionIDs (vision.Gemma4Preprocess's per-patch (x,y) output): (gridW/k)*(gridH/k), mirroring the encoder's
// average-pooling bucket count exactly, so the caller can size the prompt's placeholder run BEFORE running the (CPU-only, costly)
// encoder.
func Gemma4PooledTokens(positionIDs [][2]int, poolingKernelSize int) int {
	maxX, maxY := 0, 0
	for _, p := range positionIDs {
		if p[0] > maxX {
			maxX = p[0]
		}
		if p[1] > maxY {
			maxY = p[1]
		}
	}
	gridW, gridH := maxX+1, maxY+1
	return (gridW / poolingKernelSize) * (gridH / poolingKernelSize)
}

// FindImageRun (Gemma3ImageBlock's own function, gemma3_block.go) is already
// family-agnostic and is reused unchanged for Gemma 4.

// Gemma 4's audio placeholders (docs/tasks/task-multimodal-support-2026-10.md, S5): HF's processor (processing_gemma4.py,
// replace_audio_token) emits BOA + n × the audio soft token + EOA, with no newline on either side, as for an image.
const (
	Gemma4AudioSoftToken  = "<|audio|>"
	Gemma4AudioBlockStart = "<|audio>"
	Gemma4AudioBlockEnd   = "<audio|>"
)

// Gemma4AudioBlock returns the placeholder string for one audio clip of n soft tokens (audio.Gemma4SoftTokens).
func Gemma4AudioBlock(n int) string {
	return Gemma4AudioBlockStart + strings.Repeat(Gemma4AudioSoftToken, n) + Gemma4AudioBlockEnd
}

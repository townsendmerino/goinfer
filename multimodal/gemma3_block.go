package multimodal

import "strings"

// Gemma 3 image-block sentinels. An image is rendered into the prompt as
// ImageBlockStart + MMTokens × ImageSoftToken + ImageBlockEnd; the tokenizer
// maps ImageSoftToken to the image_token_index (262144) run that the decoder's
// embed-by-vector seam (decoder.GenerateVL / prefillLogitsVL) replaces with the
// projected vision features. Shared by cmd/serve and demo/agent so the image
// placeholder is assembled identically wherever a Gemma 3 image turn is built.
const (
	ImageSoftToken  = "<image_soft_token>"
	ImageBlockStart = "<start_of_image>"
	ImageBlockEnd   = "<end_of_image>"
)

// Gemma3ImageBlock returns the placeholder string for one image: the BOI/EOI
// sentinels wrapping n soft-token placeholders (n = Projector.MMTokens()). The
// caller renders this inside a user turn (Gemma 3 leads the user content with the
// image); the tokenizer turns the soft-token run into image-token ids.
//
// Bare — no surrounding newlines. Most callers want Gemma3PromptBlock instead (the shape a
// checkpoint's own processor actually produces); this exists separately because
// spliceImageBlock-style callers only need it as a substring to LOCATE the block within a
// rendered prompt, not to match a real processor's output byte-for-byte.
func Gemma3ImageBlock(n int) string {
	return ImageBlockStart + strings.Repeat(ImageSoftToken, n) + ImageBlockEnd
}

// Gemma3PromptBlock is the text a Gemma 3 image turn actually splices into the prompt: n soft
// tokens wrapped in Gemma3ImageBlock, itself wrapped in "\n\n" on both sides (M-38,
// audit-2026-09-10) — Gemma 3's own processor (processing_gemma3.py, verified against the real
// transformers source) does exactly this: f"\n\n{boi_token}{image_tokens}{eoi_token}\n\n". Gemma's
// SPM vocab has distinct \n\n/\n\n\n pieces, so the id stream around the sentinel differs from
// HF/llama.cpp's without the wrapping. Shared by internal/serveapp and demo/agent so both build
// the identical shape.
func Gemma3PromptBlock(n int) string {
	return "\n\n" + Gemma3ImageBlock(n) + "\n\n"
}

// FindImageRun returns the start index and length of the first run of tok in ids
// — the image-soft-token placeholder run that the embed-by-vector forward
// overrides. n == 0 when tok does not occur.
func FindImageRun(ids []int, tok int) (pos, n int) {
	for i := 0; i < len(ids); i++ {
		if ids[i] == tok {
			pos = i
			for i < len(ids) && ids[i] == tok {
				n++
				i++
			}
			return pos, n
		}
	}
	return 0, 0
}

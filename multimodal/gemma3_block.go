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
func Gemma3ImageBlock(n int) string {
	return ImageBlockStart + strings.Repeat(ImageSoftToken, n) + ImageBlockEnd
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

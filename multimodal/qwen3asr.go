package multimodal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// QwenASRPrompt builds the token ids of a Qwen3-ASR transcription turn (system is the system message's text, usually empty: the template places it in the system turn and ignores any user text), the layout Qwen3ASRProcessor produces for one audio clip with the checkpoint's chat template (a fixed layout: the
// template has no other branch for a lone audio part): an empty system turn, a user turn holding <|audio_start|>, n <|audio_pad|> placeholders and <|audio_end|>, and the assistant turn
// opened. A non-empty language forces it by prefilling the assistant turn with "language <Name><asr_text>", as the processor's apply_transcription_request does; empty lets the model detect it
// and write "language <Name><asr_text>" itself. It returns the ids and the position of the first placeholder.
func QwenASRPrompt(tk *tokenizer.Tokenizer, system string, n int, language string) (ids []int, audioPos int, err error) {
	pad, ok := tk.TokenID("<|audio_pad|>")
	if !ok {
		return nil, 0, fmt.Errorf("multimodal: the tokenizer has no <|audio_pad|> token (not a Qwen3-ASR tokenizer)")
	}
	if n <= 0 {
		return nil, 0, fmt.Errorf("multimodal: %d audio tokens", n)
	}
	pre, err := tk.Encode("<|im_start|>system\n"+system+"<|im_end|>\n<|im_start|>user\n<|audio_start|>", false)
	if err != nil {
		return nil, 0, err
	}
	post := "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"
	if language != "" {
		post += "language " + language + "<asr_text>"
	}
	tail, err := tk.Encode(post, false)
	if err != nil {
		return nil, 0, err
	}
	ids = append(ids, pre...)
	audioPos = len(ids)
	for range n {
		ids = append(ids, pad)
	}
	return append(ids, tail...), audioPos, nil
}

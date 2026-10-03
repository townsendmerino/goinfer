package decide

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// PlainTokenizer adapts a goinfer tokenizer. Text a caller supplies (a state, an option) is always encoded as a plain
// content segment, so special-token text written inside it (an "<|im_start|>" in a customer's message) stays literal
// instead of becoming a control token — the parse_special split serve's own prompts use (M25). HF's reference encodes
// bare-v1 with special parsing on, so the two differ only on such text.
type PlainTokenizer struct {
	Tok  *tokenizer.Tokenizer
	tmpl *chat.Template // nil: the model has no recognized chat template, so chat-v1 is unavailable
}

// NewPlainTokenizer wraps tk and detects its chat template (for chat-v1). A model with no recognized template still
// serves bare-v1.
func NewPlainTokenizer(tk *tokenizer.Tokenizer) PlainTokenizer {
	p := PlainTokenizer{Tok: tk}
	if t, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has}); err == nil {
		p.tmpl = t
	}
	return p
}

// EncodePlain implements Tokenizer.
func (p PlainTokenizer) EncodePlain(s string) ([]int, error) {
	return p.Tok.EncodeSegments([]tokenizer.Segment{{Text: s}}, false)
}

// DecodePlain implements StateDecoder: the tokenizer's decode, with invalid UTF-8 replaced exactly as transformers'
// byte-level decoder replaces it (a cut can split a character).
func (p PlainTokenizer) DecodePlain(ids []int) (string, error) {
	s, err := p.Tok.Decode(ids)
	if err != nil {
		return "", err
	}
	return pyReplaceInvalidUTF8([]byte(s)), nil
}

// EncodeChat implements Tokenizer: the message as one user turn with the assistant turn opened, and, for a template
// with <think>/</think>, the empty think block its enable_thinking=False rendering emits, so the next token is the
// answer rather than a think marker. The suffix is resolved through this tokenizer and verified, never pinned: the
// ids differ between Qwen vocabularies (decoder/dflash_accept_test.go's noThinkSuffix records why).
func (p PlainTokenizer) EncodeChat(user string) ([]int, error) {
	if p.tmpl == nil {
		return nil, errors.New("decide: this model has no recognized chat template; use the bare-v1 template")
	}
	ids, err := p.Tok.EncodeSegments(p.tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: user}}), false)
	if err != nil {
		return nil, err
	}
	open, oOK := p.Tok.TokenID("<think>")
	close, cOK := p.Tok.TokenID("</think>")
	if oOK && cOK {
		suffix, err := p.Tok.Encode("<think>\n\n</think>\n\n", false)
		if err != nil {
			return nil, err
		}
		if len(suffix) != 4 || suffix[0] != open || suffix[2] != close {
			return nil, fmt.Errorf("decide: the no-think suffix encoded to %v, not [<think> _ </think> _]; this template is not Qwen3-style", suffix)
		}
		ids = append(ids, suffix...)
	}
	return ids, nil
}

// ModelPrefill returns a Prefill over a loaded model: one generation of a single token, with a logit processor that
// copies the logits at the first generated position — the model's distribution at the last prompt position. It runs
// through Model.Generate, so every backend, the resident prefix reuse and the resident KV slots apply as they do to
// any request. The one sampled token is discarded.
func ModelPrefill(m *decoder.Model) Prefill {
	return func(ctx context.Context, ids []int) ([]float32, error) {
		var got []float32
		sp := decoder.SamplingParams{LogitProcessor: func(gen []int, logits []float32) {
			if got == nil {
				got = slices.Clone(logits)
			}
		}}
		ch, g := m.Generate(ctx, ids, 1, sp)
		for range ch {
		}
		if err := g.Err(); err != nil {
			return nil, err
		}
		if got == nil {
			return nil, errors.New("decide: the model produced no logits for the prompt")
		}
		return got, nil
	}
}

// ModelPrefillMany is the many-prompt form of ModelPrefill for label scoring (Route A): the next-token logits after each prompt, with the prefix the prompts share
// prefilled once. It is nil unless the model can share (decoder.Model.CanSharePrefix: the CPU Qwen3.5 path): on any other model, a resident one above all, the
// ordinary one-prompt prefill is the right path and the caller keeps it.
func ModelPrefillMany(m *decoder.Model) PrefillMany {
	if !m.CanSharePrefix() {
		return nil
	}
	return m.PromptLogitsMany
}

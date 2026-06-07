// Package chat renders a conversation into the exact prompt string a model's
// chat template expects — no Jinja engine. goinfer loads a handful of families
// (Gemma 3/4, ChatML/Qwen, Llama-3, Mistral); each has a small native Go
// renderer here, byte-exact against HuggingFace's apply_chat_template (see the
// testdata/chat_goldens fixtures).
//
// Detect picks the renderer from the GGUF/HF tokenizer.chat_template string
// (fingerprinted against the known families); for a bare checkpoint with no
// template, it falls back to the special-token heuristic. An unrecognized
// template is an explicit error — the caller then does a raw completion.
package chat

import (
	"errors"
	"strings"
	"time"
)

// Turn is one conversation message. Role is "user" or "assistant"; the system
// prompt is passed separately to Render (families place it differently).
type Turn struct {
	Role    string
	Content string
}

// Stops are the strings that end a model turn for this family (e.g.
// "<end_of_turn>"). The caller resolves them to token ids via its tokenizer and
// passes them as stop ids to the sampler.
type Stops struct {
	Strings []string
}

// Template renders a conversation into a family's prompt string and reports its
// turn-stop markers.
type Template struct {
	name   string
	render func(system string, turns []Turn) string
	stops  []string
}

// Name is the family identifier ("gemma3", "gemma4", "chatml", "llama3", "mistral").
func (t *Template) Name() string { return t.name }

// Render builds the complete prompt string (including any leading BOS marker the
// family's template emits — encode with addBOS=false) for the system prompt and
// turns, ending with the generation prompt for the model to continue.
func (t *Template) Render(system string, turns []Turn) string { return t.render(system, turns) }

// Stops returns the turn-stop marker strings for this family.
func (t *Template) Stops() Stops { return Stops{Strings: append([]string(nil), t.stops...)} }

// Meta is what Detect inspects: the model's chat-template string (from
// GGUF/HF tokenizer metadata, possibly empty) and a vocab-membership probe used
// only for the bare-checkpoint fallback.
type Meta struct {
	ChatTemplate string
	HasToken     func(string) bool
}

// ErrUnknownTemplate means neither the chat-template string nor the vocab
// markers matched a known family. The caller should fall back to feeding the raw
// text as a completion.
var ErrUnknownTemplate = errors.New("chat: unrecognized chat template (raw-completion fallback)")

// Detect resolves a Template for a model. It first fingerprints the
// chat-template string; if that's empty, it falls back to the special-token
// heuristic for bare checkpoints. Returns ErrUnknownTemplate when nothing matches.
func Detect(meta Meta) (*Template, error) {
	if t := meta.ChatTemplate; t != "" {
		switch {
		// Order matters: Gemma 4's template also mentions turns/channels, so test
		// its distinctive markers before the generic ones.
		case strings.Contains(t, "<|turn>") || strings.Contains(t, "<|channel>"):
			return Gemma4(), nil
		case strings.Contains(t, "<start_of_turn>"):
			return Gemma3(), nil
		case strings.Contains(t, "<|start_header_id|>"):
			return Llama3(), nil
		case strings.Contains(t, "<|im_start|>"):
			return ChatML(), nil
		case strings.Contains(t, "[INST]"):
			return Mistral(), nil
		}
		return nil, ErrUnknownTemplate
	}
	// Bare checkpoint: detect from the special tokens present in the vocab.
	if has := meta.HasToken; has != nil {
		switch {
		case has("<|im_start|>"):
			return ChatML(), nil
		case has("<start_of_turn>"):
			return Gemma3(), nil
		case has("<|turn>"):
			return Gemma4(), nil
		case has("<|start_header_id|>"):
			return Llama3(), nil
		}
	}
	return nil, ErrUnknownTemplate
}

// timeNow is the clock Llama-3's date preamble reads; overridable in tests.
var timeNow = time.Now

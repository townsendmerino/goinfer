// Package chat renders a conversation into the exact prompt string a model's
// chat template expects — no Jinja engine. goinfer loads a handful of families
// (Gemma 3/4, ChatML/Qwen, Llama-3, Mistral); each has a small native Go
// renderer here, checked against HuggingFace's apply_chat_template (see the
// testdata/chat_goldens fixtures).
//
// WHAT "BYTE-EXACT" COVERS, precisely (N-37 — the claim here used to be unqualified):
//
//   - The `sys_user` and `sys_multi` shapes are byte-exact for every family.
//   - The NO-SYSTEM shape is byte-exact only where a family's template has no default system
//     prompt. ChatML is the exception and it is DELIBERATE: Qwen 2.5's template inserts "You
//     are Qwen, created by Alibaba Cloud…" when the conversation has no system turn, and this
//     renderer emits no system turn at all. Adopting that string would be wrong — ChatML() is
//     the GENERIC ChatML renderer, shared with families that are not Qwen and have no such
//     default. The divergence is pinned by TestChatML_noSystem_documentedDivergence against a
//     golden rendered from Qwen's own template, so it cannot drift unnoticed in either
//     direction.
//   - Tool rendering is byte-exact for GEMMA 4 ONLY, whose tool syntax is a micro-language the
//     model parses. For the JSON families the embedded tool JSON's spacing follows Jinja's
//     tojson and is checked structurally — see TestRenderTools_declarations (M-20).
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

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Segment is a span of the rendered prompt tagged special (structural markers —
// tokenize WITH the added-token trie) or not (untrusted content — tokenize WITHOUT
// it). RenderSegments emits these so EncodeSegments can keep a marker string typed
// by a user from becoming a real control token (M25).
type Segment = tokenizer.Segment

// Turn is one conversation message. Role is "user", "assistant", or "tool" (a
// tool result); the system prompt is passed separately to Render. For tool
// calling: an assistant turn may carry ToolCalls (what the model asked for), and
// a "tool" turn carries the result of one call (ToolName + ToolCallID + Content).
type Turn struct {
	Role       string
	Content    string
	ToolCalls  []ToolCall // assistant turns: the calls the model made
	ToolName   string     // tool turns: the function this result is for
	ToolCallID string     // tool turns: the id of the call being answered
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
	render func(system string, turns []Turn) []Segment
	stops  []string
}

// Name is the family identifier ("gemma3", "gemma4", "chatml", "llama3", "mistral").
func (t *Template) Name() string { return t.name }

// Render builds the complete prompt string (including any leading BOS marker the
// family's template emits — encode with addBOS=false) for the system prompt and
// turns, ending with the generation prompt for the model to continue. It is the
// concatenation of RenderSegments; for tokenization prefer RenderSegments +
// EncodeSegments so untrusted content can't forge control tokens (M25).
func (t *Template) Render(system string, turns []Turn) string {
	var b strings.Builder
	for _, s := range t.render(system, turns) {
		b.WriteString(s.Text)
	}
	return b.String()
}

// RenderSegments builds the prompt as tagged spans: the family's structural markers
// as Special segments (tokenized with the added-token trie), untrusted message/tool
// content as non-special ones (tokenized without it). Feed to Tokenizer.
// EncodeSegments. On legitimate input the token stream equals Encode(Render(...)).
func (t *Template) RenderSegments(system string, turns []Turn) []Segment {
	return t.render(system, turns)
}

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
		// Harmony BEFORE Gemma 4: both mention channels, and harmony's "<|channel|>"
		// does NOT contain Gemma's "<|channel>" (the trailing pipe breaks the substring),
		// so the two are distinguishable — but only if harmony's own marker is tested,
		// and only in this order.
		case strings.Contains(t, "<|start|>") && strings.Contains(t, "<|message|>"):
			return Harmony(), nil
		case strings.Contains(t, "<|turn>") || strings.Contains(t, "<|channel>"):
			return Gemma4(), nil
		case strings.Contains(t, "<start_of_turn>"):
			return Gemma3(), nil
		case strings.Contains(t, "<|start_header_id|>"):
			return Llama3(), nil
		// Mellum2 IS ChatML; its distinctive normalize_content macro lets Detect
		// name it "mellum2" (banner/serve) before the generic <|im_start|> branch.
		case strings.Contains(t, "normalize_content") && strings.Contains(t, "<|im_start|>"):
			return Mellum2(), nil
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
		case has("<|start|>") && has("<|message|>") && has("<|channel|>"):
			return Harmony(), nil
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

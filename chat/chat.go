// Package chat renders a conversation into the exact prompt string a model's chat template expects, with no Jinja engine, and
// parses a reply back into reasoning, answer and tool calls. Each family (Gemma 3/4, ChatML/Qwen, Llama-3, Mistral, Ministral 3,
// Phi-3, Harmony/gpt-oss, GLM-OCR) has a small native Go renderer, checked against HuggingFace's apply_chat_template
// (testdata/chat_goldens, testdata/chat_think_goldens).
//
// What "byte-exact" covers, precisely:
//
//   - A system prompt with one or more turns, for every family.
//   - The no-system shape. A ChatML template that declares a default system message (Qwen 2.5's, Qwen2.5-VL's) gets it, read
//     from the checkpoint's own template by Detect (default_system.go), so its no-system rendering is that template's byte for
//     byte. The bare ChatML() renderer, shared with families that declare none, emits no system turn
//     (TestChatML_noSystem_documentedDivergence). Ministral's default system message is deliberately not replicated (see Ministral).
//   - Tool rendering is byte-exact where the tool syntax is a micro-language the model parses or the template's own form is selected
//     (Gemma 4, Harmony, Qwen3.5's XML). For the JSON families the embedded tool JSON's spacing follows Jinja's tojson and is
//     checked structurally (TestRenderTools_declarations).
//
// Detect picks the renderer from the GGUF/HF tokenizer.chat_template string, falling back to a special-token heuristic for a bare
// checkpoint with no template. An unrecognized template is an explicit error; the caller then does a raw completion.
package chat

import (
	"errors"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Segment is a span of the rendered prompt tagged special (structural markers: tokenize WITH the added-token trie) or not
// (untrusted content: tokenize WITHOUT it). RenderSegments emits these so EncodeSegments can keep a marker string typed by a user
// from becoming a real control token.
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
	// Reasoning is an assistant turn's own reasoning, as the client replays it (OpenAI reasoning_content / reasoning,
	// Anthropic thinking blocks). A family with a history rule renders it the way the model's own template does (history.go):
	// kept for the turns of the tool loop in progress, dropped for turns before the last user query.
	Reasoning string
	// ToolLoop marks a user turn that only ACCOMPANIES tool results (an Anthropic user message carries its tool_result blocks
	// and any reminder text together). The history rule must not read it as a new user query, or every tool loop from such a
	// client would lose its reasoning on the very turn after the first tool result.
	ToolLoop bool
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

	// defaultSystem is the system message the checkpoint's own template inserts when the conversation has none (ChatML
	// families only, read from the template by chatMLDefaultSystem); "" for none. Every render path applies it (systemOr).
	defaultSystem string

	// reason is this checkpoint's declared thinking behaviour (reasoning.go); nil = the family does not think in its
	// template, or the template's control was not recognised. think is the mode WithThinking selected; the zero value
	// ThinkAsIs renders exactly what render renders.
	reason *Reasoning
	think  ThinkMode

	// effort is a Harmony (gpt-oss) template's reasoning effort, written on its `Reasoning:` line (templates.go); "" elsewhere.
	effort string

	// nativeTools: the template declares a tool form goinfer can reproduce byte for byte — Qwen3.5's XML (qwen_xml_tools.go) or Gemma 4's
	// canonical template (gemma4_tools.go); toolFormat is which prompt WithToolFormat selected (the zero value is `auto`).
	nativeTools bool
	toolFormat  ToolFormat
	// nativeByDefault: for this template the native form is what `auto` selects — true only where a pre-registered A/B adopted it.
	nativeByDefault bool
	// lenientFenced: WithLenientToolCalls — also read ONE fenced JSON call at the end of a reply as a call (fenced_tool_calls.go). Off by default.
	lenientFenced bool

	// groupsToolResults: the template puts CONSECUTIVE tool results in one user turn (Qwen2.5, Qwen3, Qwen3.5 do), read from its text
	// (detectGroupedToolResults). goinfer's Hermes renderer writes one user turn per result for a template that does not.
	groupsToolResults bool
}

// Name is the family identifier ("chatml", "mellum2", "gemma3", "gemma4", "harmony", "llama3",
// "mistral", "ministral", "phi3", "phi3_orig").
func (t *Template) Name() string { return t.name }

// Render builds the complete prompt string, including any leading BOS marker the family's template emits (encode with
// addBOS=false), for the system prompt and turns, ending with the generation prompt. It is the concatenation of RenderSegments;
// for tokenization prefer RenderSegments + EncodeSegments so untrusted content cannot forge control tokens.
func (t *Template) Render(system string, turns []Turn) string {
	var b strings.Builder
	for _, s := range t.RenderSegments(system, turns) {
		b.WriteString(s.Text)
	}
	return b.String()
}

// RenderSegments builds the prompt as tagged spans: the family's structural markers
// as Special segments (tokenized with the added-token trie), untrusted message/tool
// content as non-special ones (tokenized without it). Feed to Tokenizer.
// EncodeSegments. On legitimate input the token stream equals Encode(Render(...)).
func (t *Template) RenderSegments(system string, turns []Turn) []Segment {
	system = t.systemOr(system)
	if t.reason != nil && t.think != ThinkAsIs {
		return t.renderThinking(system, turns)
	}
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
		// Order matters: the more specific fingerprint is tested before its generic sibling. Harmony before Gemma 4 (both
		// mention channels; Harmony's "<|channel|>" does not contain Gemma's "<|channel>", so each is distinguishable only if its own marker is
		// tested), Mellum2 before ChatML, and the three declines below before Mistral and ChatML.
		case strings.Contains(t, "<|start|>") && strings.Contains(t, "<|message|>"):
			return Harmony(), nil
		case strings.Contains(t, "<|turn>") || strings.Contains(t, "<|channel>"):
			g := Gemma4()
			g.reason = detectGemma4Reasoning(t)
			g.nativeTools = g.reason != nil && g.reason.hist == histGemma4 // the canonical template, whose loop the native renderer ports
			g.nativeByDefault = g.nativeTools                              // ADOPTED: docs/measurements/gemma4-tool-text-order-2026-09-30/RESULTS.md
			return g, nil
		case strings.Contains(t, "<start_of_turn>"):
			return Gemma3(), nil
		case strings.Contains(t, "<|start_header_id|>"):
			return Llama3(), nil
		// GLM-OCR: "[gMASK]<sop>" with the <|user|>/<|assistant|> roles and its image markers. The image markers are
		// part of the fingerprint on purpose (see GlmOCR): GLM-4.5's text template opens the same way and is not this one.
		case strings.Contains(t, "[gMASK]<sop>") && strings.Contains(t, "<|begin_of_image|>") && strings.Contains(t, "<|user|>"):
			return GlmOCR(), nil
		// Mellum2 IS ChatML; its distinctive normalize_content macro lets Detect
		// name it "mellum2" (banner/serve) before the generic <|im_start|> branch.
		case strings.Contains(t, "normalize_content") && strings.Contains(t, "<|im_start|>"):
			// Mellum2.1's template adds Qwen3's thinking control and history rule to the same ChatML body; 2.0's has neither, so the detectors return nil for it.
			m := Mellum2()
			m.defaultSystem = chatMLDefaultSystem(t)
			m.reason = detectChatMLReasoning(t)
			m.nativeTools = declaresQwen35XMLTools(t, m.reason)
			m.groupsToolResults = detectGroupedToolResults(t)
			return m, nil
		// Each of these three would otherwise match a generic branch below and be silently misrendered:
		//   - SmolLM3: ChatML markers, but the template always writes its own "## Metadata" system preamble the caller never asked for;
		//     declined, not rendered as plain ChatML.
		//   - Olmo 3: ChatML markers, but its tools are <functions>/<function_calls> XML, not the Hermes <tool_call> JSON dialect; declined.
		//   - Ministral 3: contains "[INST]" like Mistral(), but is [SYSTEM_PROMPT]-based; routed to Ministral() (see there).
		// Details: docs/code-notes/chat.md#Detect.fingerprints.
		case strings.Contains(t, "## Metadata"):
			return nil, ErrUnknownTemplate
		case strings.Contains(t, "<function_calls>"):
			return nil, ErrUnknownTemplate
		case strings.Contains(t, "[SYSTEM_PROMPT]"):
			return Ministral(), nil
		// Phi-3: "<|user|>" / "<|end|>" markers. Its current template has a system branch; the
		// first release's (still in its q4 GGUF) has none and opens with bos_token instead.
		case strings.Contains(t, "<|user|>") && strings.Contains(t, "<|end|>") && strings.Contains(t, "<|system|>"):
			return Phi3(), nil
		case strings.Contains(t, "<|user|>") && strings.Contains(t, "<|end|>") && strings.Contains(t, "bos_token"):
			return Phi3Orig(), nil
		case strings.Contains(t, "<|im_start|>"):
			c := ChatML()
			c.defaultSystem = chatMLDefaultSystem(t)
			c.reason = detectChatMLReasoning(t)
			c.nativeTools = declaresQwen35XMLTools(t, c.reason)
			c.groupsToolResults = detectGroupedToolResults(t)
			return c, nil
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
		case has("[SYSTEM_PROMPT]") && has("[INST]"):
			// Ministral 3 (and the Mistral 3 VL saves) ship no chat template; their tekken vocab carries [SYSTEM_PROMPT] as a control token,
			// which Mistral v0.3's [INST]-only vocab does not. Without this branch an image request on one is refused.
			return Ministral(), nil
		}
	}
	return nil, ErrUnknownTemplate
}

// timeNow is the clock Llama-3's date preamble reads; overridable in tests.
var timeNow = time.Now

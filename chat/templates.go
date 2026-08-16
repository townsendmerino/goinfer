package chat

import "strings"

// Each constructor returns the family's renderer. The render funcs are written
// to match HuggingFace apply_chat_template byte-for-byte (testdata/chat_goldens).
//
// Renderers emit []Segment, not a raw string (Render concatenates them). A Special
// segment is a genuine single special token of the family (its control markers); a
// non-special segment is a whole gap BETWEEN two special tokens — trusted structure
// (role names, newlines, date preambles) and untrusted content together. Because the
// non-special segments are exactly the gaps the whole-string encoder would form,
// EncodeSegments reproduces Encode(Render(...)) on legitimate input while refusing to
// promote a marker string a user typed into a real control token (M25). The rule when
// editing a renderer: sp() ONLY genuine special tokens; everything else via ct().

// segBuf accumulates render segments.
type segBuf struct{ segs []Segment }

// sp appends a structural special-token span (tokenized WITH the trie). It must be a
// single genuine special token of the family, so the segment boundary lands on a real
// token break and no cross-boundary BPE merge is lost.
func (b *segBuf) sp(text string) { b.segs = append(b.segs, Segment{Text: text, Special: true}) }

// ct appends a content span (tokenized WITHOUT the trie) — a whole inter-special gap,
// trusted structure and untrusted content alike, so any marker string inside stays
// literal text.
func (b *segBuf) ct(text string) {
	if text != "" {
		b.segs = append(b.segs, Segment{Text: text, Special: false})
	}
}

// Gemma3 — "<bos>" then per turn "<start_of_turn>{role}\n{content}<end_of_turn>\n"
// (assistant→model); no system role, so the system is folded into the first user
// turn ("{system}\n\n{content}"). Generation prompt: "<start_of_turn>model\n".
func Gemma3() *Template {
	return &Template{name: "gemma3", stops: []string{"<end_of_turn>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		b.sp("<bos>")
		firstUser := true
		for _, t := range turns {
			role, content := "user", t.Content
			if t.Role == "assistant" {
				role = "model"
			} else if firstUser {
				firstUser = false
				if system != "" {
					content = system + "\n\n" + content
				}
			}
			b.sp("<start_of_turn>")
			b.ct(role + "\n" + content)
			b.sp("<end_of_turn>")
			b.ct("\n")
		}
		b.sp("<start_of_turn>")
		b.ct("model\n")
		return b.segs
	}}
}

// Gemma4 — Gemma 3's successor with new markers: "<bos>", a real system turn
// "<|turn>system\n{system}<turn|>\n", per turn "<|turn>{role}\n{content}<turn|>\n"
// (assistant→model), ending with the generation prompt plus Gemma 4's thinking
// scaffold: "<|turn>model\n<|channel>thought\n<channel|>".
func Gemma4() *Template {
	return &Template{name: "gemma4", stops: []string{"<turn|>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		b.sp("<bos>")
		if system != "" {
			b.sp("<|turn>")
			b.ct("system\n" + system)
			b.sp("<turn|>")
			b.ct("\n")
		}
		for _, t := range turns {
			role := "user"
			if t.Role == "assistant" {
				role = "model"
			}
			b.sp("<|turn>")
			b.ct(role + "\n" + t.Content)
			b.sp("<turn|>")
			b.ct("\n")
		}
		b.sp("<|turn>")
		b.ct("model\n")
		b.sp("<|channel>")
		b.ct("thought\n")
		b.sp("<channel|>")
		return b.segs
	}}
}

// Harmony (gpt-oss) — "<|start|>{role}<|message|>{content}<|end|>", with a REQUIRED
// system message that gpt-oss's template synthesizes rather than taking from the caller,
// and a generation prompt of a bare "<|start|>assistant".
//
// Three things make this family unlike the others here, all of them load-bearing:
//
//  1. THE SYSTEM MESSAGE IS SYNTHESIZED, NOT PASSED THROUGH. gpt-oss's own template emits a
//     fixed identity line, a knowledge cutoff, TODAY'S DATE, a reasoning-effort line and the
//     valid-channel declaration — whether or not the caller supplied a system prompt. A
//     caller-supplied system prompt is a DEVELOPER message in harmony, which is a separate
//     role, so it is rendered as one rather than replacing the preamble.
//  2. THE DATE IS LIVE. `Current date:` comes from strftime_now in the upstream template, so
//     it is read from timeNow (the same injectable clock Llama-3's preamble uses) and a
//     byte-exactness test must pin the clock.
//  3. THE CHANNEL SET IS DECLARED, NOT OPTIONAL. "Channel must be included for every message"
//     — gpt-oss always answers on a channel (analysis / commentary / final), so there is no
//     non-thinking form of this prompt the way Qwen3 and Gemma-4 both have. Callers that want
//     only the answer must strip the analysis channel from the OUTPUT; it cannot be suppressed
//     in the prompt.
//
// Reasoning effort defaults to "medium", matching the upstream template's own default.
func Harmony() *Template {
	return &Template{name: "harmony", stops: []string{"<|return|>", "<|end|>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		b.sp("<|start|>")
		b.ct("system")
		b.sp("<|message|>")
		b.ct("You are ChatGPT, a large language model trained by OpenAI.\n" +
			"Knowledge cutoff: 2024-06\n" +
			"Current date: " + timeNow().Format("2006-01-02") + "\n\n" +
			"Reasoning: medium\n\n" +
			"# Valid channels: analysis, commentary, final. Channel must be included for every message.")
		b.sp("<|end|>")
		if system != "" {
			b.sp("<|start|>")
			b.ct("developer")
			b.sp("<|message|>")
			b.ct("# Instructions\n\n" + system + "\n\n")
			b.sp("<|end|>")
		}
		for _, t := range turns {
			role := "user"
			if t.Role == "assistant" {
				role = "assistant"
			}
			b.sp("<|start|>")
			b.ct(role)
			b.sp("<|message|>")
			b.ct(t.Content)
			b.sp("<|end|>")
		}
		b.sp("<|start|>")
		b.ct("assistant")
		return b.segs
	}}
}

// ChatML (Qwen and most byte-level families) — per turn
// "<|im_start|>{role}\n{content}<|im_end|>\n", a leading system turn when given,
// generation prompt "<|im_start|>assistant\n". No BOS in the template.
func ChatML() *Template {
	return &Template{name: "chatml", stops: []string{"<|im_end|>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		if system != "" {
			b.sp("<|im_start|>")
			b.ct("system\n" + system)
			b.sp("<|im_end|>")
			b.ct("\n")
		}
		for _, t := range turns {
			b.sp("<|im_start|>")
			b.ct(t.Role + "\n" + t.Content)
			b.sp("<|im_end|>")
			b.ct("\n")
		}
		b.sp("<|im_start|>")
		b.ct("assistant\n")
		return b.segs
	}}
}

// Mellum2 (JetBrains Mellum2) renders ChatML — its chat template is ChatML
// byte-for-byte (<|im_start|>/<|im_end|> turns, stop <|im_end|>, Hermes
// <tool_call> tools), verified vs HF apply_chat_template
// (testdata/chat_goldens/mellum2.json). It is a named alias so Detect / cmd/serve
// / demo/chat identify it as "mellum2" distinctly; the render, stops, and tools
// are the ChatML path. Detect fingerprints its distinctive normalize_content
// macro before the generic <|im_start|> ChatML branch.
func Mellum2() *Template {
	t := ChatML()
	t.name = "mellum2"
	return t
}

// Llama3 — "<|begin_of_text|>", an always-present system block carrying the
// date preamble (then the system text), per turn
// "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>", and the
// assistant generation header. The "Today Date" is the current date.
func Llama3() *Template {
	return &Template{name: "llama3", stops: []string{"<|eot_id|>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		date := timeNow().Format("02 Jan 2006")
		b.sp("<|begin_of_text|>")
		b.sp("<|start_header_id|>")
		b.ct("system")
		b.sp("<|end_header_id|>")
		b.ct("\n\nCutting Knowledge Date: December 2023\nToday Date: " + date + "\n\n" + system)
		b.sp("<|eot_id|>")
		for _, t := range turns {
			b.sp("<|start_header_id|>")
			b.ct(t.Role)
			b.sp("<|end_header_id|>")
			b.ct("\n\n" + t.Content)
			b.sp("<|eot_id|>")
		}
		b.sp("<|start_header_id|>")
		b.ct("assistant")
		b.sp("<|end_header_id|>")
		b.ct("\n\n")
		return b.segs
	}}
}

// Mistral — "<s>" once, each user turn "[INST] {content}[/INST]", each assistant
// turn " {content}</s>". No system role: the system is folded into the LAST user
// turn ("{system}\n\n{content}").
//
// NOTE: unlike the families above, Mistral's structural markers are version-
// dependent — [INST]/[/INST] are plain text in v0.1 but real special tokens in
// v0.3+, and <s>/</s> placement interleaves with content inside a single encoder
// gap. Statically deciding the special/content split would risk changing the
// tokenization of legitimate prompts, so this renderer emits ONE Special segment
// (identical to whole-string Encode — no regression) and forgoes the injection
// hardening the others get. Splitting it safely needs the loaded tokenizer's
// added-vocabulary, a follow-up.
func Mistral() *Template {
	return &Template{name: "mistral", stops: []string{"</s>"}, render: func(system string, turns []Turn) []Segment {
		lastUser := -1
		for i, t := range turns {
			if t.Role != "assistant" {
				lastUser = i
			}
		}
		var sb strings.Builder
		sb.WriteString("<s>")
		for i, t := range turns {
			if t.Role == "assistant" {
				sb.WriteString(" " + t.Content + "</s>")
				continue
			}
			content := t.Content
			if i == lastUser && system != "" {
				content = system + "\n\n" + content
			}
			sb.WriteString("[INST] " + content + "[/INST]")
		}
		return []Segment{{Text: sb.String(), Special: true}}
	}}
}

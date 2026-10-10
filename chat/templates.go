package chat

import "strings"

// Each constructor returns the family's renderer, written to match HuggingFace apply_chat_template byte for byte
// (testdata/chat_goldens).
//
// Renderers emit []Segment, not a raw string (Render concatenates them). A Special segment is one genuine special token of the
// family (its control markers); a non-special segment is a whole gap BETWEEN two special tokens: trusted structure (role names,
// newlines, date preambles) and untrusted content together. Those gaps are exactly what the whole-string encoder would form, so
// EncodeSegments reproduces Encode(Render(...)) on legitimate input while refusing to promote a marker string a user typed into a
// control token. The rule when editing a renderer: sp() ONLY genuine special tokens; everything else via ct().

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
		return gemma4Segments(system, turns, false, false, false)
	}}
}

// gemma4Segments renders the Gemma 4 conversation. think=false is the family's generic rendering (its generation prompt
// carries the closed thinking scaffold); think=true is the template's enable_thinking=true form: a system turn that
// opens with the "<|think|>" marker (present even when the caller gave no system prompt) and a generation prompt that
// leaves the thinking channel for the model to open. old=true is the earlier template: its thinking-off prompt has no closed channel at all,
// and it keeps a turn's reasoning only when the turn makes a tool call (a call in this text-only path never happens, so none).
func gemma4Segments(system string, turns []Turn, think, hist, old bool) []Segment {
	var b segBuf
	b.sp("<bos>")
	if system != "" || think {
		b.sp("<|turn>")
		if think {
			b.ct("system\n")
			b.sp("<|think|>")
			b.ct("\n" + system)
		} else {
			b.ct("system\n" + system)
		}
		b.sp("<turn|>")
		b.ct("\n")
	}
	lu := lastUserIndex(turns)
	for i, t := range turns {
		role := "user"
		if t.Role == "assistant" {
			role = "model"
		}
		// The managed rendering follows the template: consecutive assistant messages are ONE model turn — no turn header for a message
		// that follows an assistant message, and no turn end for one that is followed by another (`continue_same_model_turn`,
		// `continues_into_next`). The generic rendering keeps a turn per message, byte for byte as before the history rule existed.
		joinsPrev := role == "model" && hist && i > 0 && turns[i-1].Role == "assistant"
		joinsNext := role == "model" && hist && i+1 < len(turns) && turns[i+1].Role == "assistant"
		if !joinsPrev {
			b.sp("<|turn>")
		}
		if role == "model" && hist {
			if !joinsPrev {
				b.ct("model\n")
			}
			if t.Reasoning != "" && i > lu && !old { // the tool loop in progress keeps its reasoning; earlier turns do not
				b.sp("<|channel>")
				b.ct("thought\n" + t.Reasoning + "\n")
				b.sp("<channel|>")
			}
			b.ct(stripChannels(t.Content))
		} else {
			b.ct(role + "\n" + t.Content)
		}
		if !joinsNext {
			b.sp("<turn|>")
			b.ct("\n")
		}
	}
	b.sp("<|turn>")
	b.ct("model\n")
	if !think && !old {
		b.sp("<|channel>")
		b.ct("thought\n")
		b.sp("<channel|>")
	}
	return b.segs
}

// Harmony (gpt-oss) is "<|start|>{role}<|message|>{content}<|end|>", with a required system message that gpt-oss's template
// synthesizes rather than takes from the caller, and a generation prompt of a bare "<|start|>assistant".
//
// Three things make this family unlike the others, all load-bearing:
//
//  1. THE SYSTEM MESSAGE IS SYNTHESIZED, NOT PASSED THROUGH. The template emits a fixed identity line, a knowledge cutoff, today's
//     date, a reasoning-effort line and the valid-channel declaration whether or not the caller supplied a system prompt. A
//     caller's system prompt is a DEVELOPER message (a separate role), rendered as one rather than replacing the preamble.
//  2. THE DATE IS LIVE. `Current date:` comes from strftime_now upstream, so it is read from timeNow (the injectable clock
//     Llama-3's preamble also uses); a byte-exactness test must pin the clock.
//  3. THE CHANNEL SET IS DECLARED, NOT OPTIONAL ("Channel must be included for every message"). There is no non-thinking form of
//     this prompt as Qwen3 and Gemma 4 have; a caller that wants only the answer strips the analysis channel from the OUTPUT.
//
// Reasoning effort defaults to "medium", the upstream template's default.
//
// STOPS ARE UPSTREAM'S, NOT "EVERY END MARKER": <|return|>, <|endoftext|> and <|call|> (gpt-oss's generation_config eos_token_id).
// <|end|> is NOT a stop: it closes each MESSAGE and one reply is several (the analysis message ends in <|end|>, then the model opens
// the final channel and ends the turn with <|return|>, or <|call|> for a tool call). Stopping on <|end|> ends every reply after its
// thinking, so serve streams only the analysis channel. Evidence: docs/code-notes/chat.md#Harmony.
func Harmony() *Template { return harmonyTemplate("medium") }

// harmonyTemplate is Harmony with the system block's `Reasoning:` line set to effort ("low" | "medium" | "high" — the values gpt-oss's
// own template takes as its reasoning_effort).
func harmonyTemplate(effort string) *Template {
	return &Template{name: "harmony", effort: effort, stops: []string{"<|return|>", "<|call|>", "<|endoftext|>"}, render: func(system string, turns []Turn) []Segment {
		return harmonySegments(effort, system, turns, nil)
	}}
}

// NormalizeReasoningEffort reports whether s is a reasoning effort gpt-oss's template understands — low, medium or high, any case —
// and returns it in the template's spelling. Nothing else is an effort here: "none" is the off switch of the families that have
// one (it acts through WithThinking, not through this), and an unrecognised value is ignored by callers, never an error.
func NormalizeReasoningEffort(s string) (string, bool) {
	switch e := strings.ToLower(strings.TrimSpace(s)); e {
	case "low", "medium", "high":
		return e, true
	}
	return "", false
}

// WithReasoningEffort returns a copy of a Harmony (gpt-oss) template that writes effort on its `Reasoning:` line — the only family
// whose own template takes the knob. For every other template, or an effort that is not low/medium/high, it returns t unchanged,
// which is what keeps a client that sends a bare reasoning_effort to every endpoint (dsh does) from changing any other model's prompt.
func (t *Template) WithReasoningEffort(effort string) *Template {
	e, ok := NormalizeReasoningEffort(effort)
	if t == nil || t.name != "harmony" || !ok {
		return t
	}
	c := harmonyTemplate(e)
	c.think = t.think
	return c
}

// ChatML (Qwen and most byte-level families) — per turn
// "<|im_start|>{role}\n{content}<|im_end|>\n", a leading system turn when given,
// generation prompt "<|im_start|>assistant\n". No BOS in the template.
func ChatML() *Template {
	return &Template{name: "chatml", stops: []string{"<|im_end|>"}, render: func(system string, turns []Turn) []Segment {
		return chatMLSegments(system, turns, histNone)
	}}
}

// chatMLSegments renders the conversation as ChatML. hist is the family's history rule for assistant turns (history.go);
// histNone is the generic rendering — each turn's content as given — and is what every ChatML family without a recognised
// thinking control gets, byte for byte as before the history rule existed.
func chatMLSegments(system string, turns []Turn, hist histKind) []Segment {
	var b segBuf
	if system != "" {
		b.sp("<|im_start|>")
		b.ct("system\n" + system)
		b.sp("<|im_end|>")
		b.ct("\n")
	}
	lq := lastQueryIndex(turns)
	for i, t := range turns {
		b.sp("<|im_start|>")
		if t.Role == "assistant" && hist != histNone {
			block, reasoning, content := qwenAssistant(hist, t, i, lq, len(turns))
			if block {
				b.ct("assistant\n")
				b.sp("<think>") // the markers are control tokens; the reasoning between them is untrusted text
				b.ct("\n" + reasoning + "\n")
				b.sp("</think>")
				b.ct("\n\n" + content)
			} else {
				b.ct("assistant\n" + content)
			}
		} else {
			b.ct(t.Role + "\n" + t.Content)
		}
		b.sp("<|im_end|>")
		b.ct("\n")
	}
	b.sp("<|im_start|>")
	b.ct("assistant\n")
	return b.segs
}

// Phi3 is microsoft/Phi-3-mini-4k-instruct's current template: per turn "<|{role}|>\n{content}<|end|>\n", a leading system turn when
// given, generation prompt "<|assistant|>\n". No BOS in the template, and the HF tokenizer adds none (add_bos_token false).
func Phi3() *Template {
	return &Template{name: "phi3", stops: []string{"<|end|>", "<|endoftext|>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		if system != "" {
			b.sp("<|system|>")
			b.ct("\n" + system)
			b.sp("<|end|>")
			b.ct("\n")
		}
		for _, t := range turns {
			b.sp("<|" + t.Role + "|>")
			b.ct("\n" + t.Content)
			b.sp("<|end|>")
			b.ct("\n")
		}
		b.sp("<|assistant|>")
		b.ct("\n")
		return b.segs
	}}
}

// Phi3Orig — the template Phi-3-mini-4k-instruct shipped with in its first release, and the
// one its q4 GGUF still carries: "<s>", then per user turn
// "<|user|>\n{content}<|end|>\n<|assistant|>\n" (the generation prompt is part of every user
// turn) and per assistant turn "{content}<|end|>\n". It has NO system branch, so a system
// prompt is dropped, as it is by HF and llama.cpp rendering the same template. Emits the BOS
// itself: encode with addBOS=false.
func Phi3Orig() *Template {
	return &Template{name: "phi3_orig", stops: []string{"<|end|>", "<|endoftext|>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		b.sp("<s>")
		for _, t := range turns {
			if t.Role == "assistant" {
				b.ct(t.Content)
				b.sp("<|end|>")
				b.ct("\n")
				continue
			}
			b.sp("<|user|>")
			b.ct("\n" + t.Content)
			b.sp("<|end|>")
			b.ct("\n")
			b.sp("<|assistant|>")
			b.ct("\n")
		}
		return b.segs
	}}
}

// GlmOCR is zai-org/GLM-OCR's template (the GLM-4.1V family's `[gMASK]<sop>` shape): "[gMASK]<sop>", an optional
// "<|system|>\n{system}", then per turn "<|user|>\n{content}" or "<|assistant|>\n<think></think>\n{content}" (the template writes an
// EMPTY think block for a prior assistant turn with no reasoning, and this renderer carries none), and the generation prompt
// "<|assistant|>\n". No end-of-turn marker: the next role marker closes a turn, and a reply ends at <|endoftext|> or the next
// <|user|> (the checkpoint's eos_token_id is [59246, 59253], hence the stops).
//
// Byte-for-byte against HF's apply_chat_template for text turns (testdata/chat_goldens/glm_ocr.json), and for the image shape through
// multimodal.GlmOcrImageBlock. NOT rendered: tools (`<|observation|>` turns, the `<tool_call>` XML), `enable_thinking`'s
// `<think></think>` generation-prompt suffix and `/nothink` (the OCR model is prompted with a task string, not a conversation).
// Detect matches only this checkpoint's template (the image markers are in the fingerprint), so GLM-4.5's text template, which
// shares the `[gMASK]<sop>` opening and renders differently, is not captured by it.
func GlmOCR() *Template {
	return &Template{name: "glm_ocr", stops: []string{"<|endoftext|>", "<|user|>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		b.sp("[gMASK]")
		b.sp("<sop>")
		if system != "" {
			b.sp("<|system|>")
			b.ct("\n" + system)
		}
		for _, t := range turns {
			switch t.Role {
			case "assistant":
				b.sp("<|assistant|>")
				b.ct("\n")
				b.sp("<think>")
				b.sp("</think>")
				b.ct("\n" + strings.TrimSpace(t.Content)) // the template writes '\n' + content.strip() when it is non-blank
			default:
				b.sp("<|user|>")
				b.ct("\n" + t.Content)
			}
		}
		b.sp("<|assistant|>")
		b.ct("\n")
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

// Mistral is "<s>" once, each user turn "[INST] {content}[/INST]", each assistant turn " {content}</s>". No system role: the system is
// folded into the LAST user turn ("{system}\n\n{content}").
//
// Unlike the families above its markers are version-dependent ([INST]/[/INST] are plain text in v0.1 but special tokens in v0.3+, and
// <s>/</s> interleave with content inside one encoder gap), so a static special/content split could change the tokenization of
// legitimate prompts. It emits ONE Special segment (identical to whole-string Encode) and so gets no injection hardening; splitting
// it safely needs the loaded tokenizer's added vocabulary.
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

// Ministral is Ministral 3's own template (mistralai/Ministral-3-8B-Instruct-2512's chat_template.jinja): "<s>" once, an explicit system
// message as its own "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]" block, no "[INST] " space before user content (unlike Mistral()), and
// no leading space before an assistant turn. Detect must not route it to Mistral(): it shares the "[INST]" substring but is a
// different template version.
//
// The system block is rendered FIRST, right after "<s>": the real template writes it wherever a system message falls, and the
// caller's separate system parameter can only be the first message.
//
// NOT implemented: the template's default system message (injected only when the caller gives none: a long, Mistral-product-branded
// prompt naming the model and "Le Chat"). It is deliberately not replicated, as ChatML() leaves out Qwen's default; injecting
// Mistral's product identity into a self-hosted response would be wrong, not just an omission. So the no-system case is NOT
// byte-exact for this family, and the goldens cover explicit-system cases only.
//
// Tool calling is NOT wired (SupportsTools declines it): the real format ([TOOL_CALLS]name[ARGS]{json}, no JSON-array wrapping)
// differs from the "mistral" dialect enough that reusing it would mis-render and mis-parse silently rather than be visibly incomplete.
//
// Every marker is a single control token of Ministral 3's tokenizer (<s> 1, </s> 2, [INST] 3, [/INST] 4, [SYSTEM_PROMPT] 17,
// [/SYSTEM_PROMPT] 18), so they are sp() and the system and turn texts ct(): a marker a user types stays literal, and an image block in
// a user turn is a content gap the vision splice can find (as one Special segment the whole prompt would hide it, and every Pixtral
// request would be refused).
func Ministral() *Template {
	return &Template{name: "ministral", stops: []string{"</s>"}, render: func(system string, turns []Turn) []Segment {
		var b segBuf
		b.sp("<s>")
		if system != "" {
			b.sp("[SYSTEM_PROMPT]")
			b.ct(system)
			b.sp("[/SYSTEM_PROMPT]")
		}
		for _, t := range turns {
			if t.Role == "assistant" {
				b.ct(t.Content)
				b.sp("</s>")
				continue
			}
			b.sp("[INST]")
			b.ct(t.Content)
			b.sp("[/INST]")
		}
		return b.segs
	}}
}

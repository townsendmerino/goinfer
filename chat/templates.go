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
//
// STOPS ARE UPSTREAM'S, NOT "EVERY END MARKER". gpt-oss-20b's generation_config.json lists
// eos_token_id [200002, 199999, 200012] = <|return|>, <|endoftext|>, <|call|>. <|end|> is NOT a stop:
// it closes each MESSAGE, and one reply is several — the analysis message ends in <|end|>, then the
// model opens <|start|>assistant<|channel|>final<|message|> and ends the turn with <|return|> (or
// <|call|> for a tool call). Stopping on <|end|> ended every reply after its thinking: through serve,
// gpt-oss streamed only its analysis channel and never an answer (found 2026-09-14 by the web UI's W6
// capture, docs/tasks/task-web-ui-2026-09.md).
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

// Phi3 — microsoft/Phi-3-mini-4k-instruct's current template: per turn
// "<|{role}|>\n{content}<|end|>\n", a leading system turn when given, generation prompt
// "<|assistant|>\n". No BOS in the template, and the HF tokenizer adds none
// (add_bos_token false).
//
// Before this renderer existed Detect matched no Phi-3 template, so chat and serve fed Phi-3
// a raw completion with no turn markers at all: the model wrote its own "<|assistant|>" and
// answered the benchmark's filler prompt with newlines (found 2026-09-25,
// docs/measurements/peer-claim-2026-09-25.md).
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

// GlmOCR — zai-org/GLM-OCR's template (the GLM-4.1V family's `[gMASK]<sop>` shape): "[gMASK]<sop>", an optional
// "<|system|>\n{system}", then per turn "<|user|>\n{content}" or "<|assistant|>\n<think></think>\n{content}" (a prior
// assistant turn; the template writes an EMPTY think block there when the turn carries no reasoning, and this renderer
// carries none), and the generation prompt "<|assistant|>\n". No end-of-turn marker: the next role marker closes a turn,
// and a reply ends at <|endoftext|> or the next <|user|> (the checkpoint's eos_token_id is [59246, 59253], hence the stops).
//
// Verified byte-for-byte against HF's apply_chat_template on the checkpoint at revision 2e85a628
// (testdata/chat_goldens/glm_ocr.json, text turns), and for the image shape by the O3 gate
// (multimodal.GlmOcrImageBlock + the golden's input_ids). NOT rendered: tools (`<|observation|>` turns and the
// `<tool_call>` XML), `enable_thinking`'s `<think></think>` generation-prompt suffix and `/nothink` — the OCR model is
// prompted with a task string, not a conversation, and none of those is a path it is used on. Detect matches only this
// checkpoint's template (the image markers are in the fingerprint), so GLM-4.5's text template, which shares the
// `[gMASK]<sop>` opening and is rendered differently, is not captured by it.
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

// Ministral — Ministral 3's own template (mistralai/Ministral-3-8B-Instruct-2512's real
// chat_template.jinja, fetched and read 2026-09-16, not guessed from Mistral v0.3's shape): "<s>"
// once, an EXPLICIT system message as its own "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]" block —
// no "[INST] " space before user content (unlike v0.3's Mistral(), which has one), and no eos
// leading space before an assistant turn either.
//
// M-36 (audit-2026-09-10): this family was previously misdetected as Mistral() via the shared
// "[INST]" substring — a different template version with a different id stream on every turn
// (no [SYSTEM_PROMPT] at all, "[INST] " WITH a space, system folded into the last user turn
// instead of rendered as its own block).
//
// System placement: the real template emits [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] wherever a
// role=="system" message naturally falls in the conversation (validated to be first or absent by
// its own role-ordering pass in the common case) — this renders it FIRST, right after "<s>" and
// before any turn, which is exactly what the real template produces when the caller's system
// message is the conversation's first message, the only shape goinfer's system-as-a-separate-
// parameter Render(system, turns) interface can represent (there is no per-turn system Turn).
//
// NOT implemented: the real template's default system message (injected only when the caller
// provides NONE at all — a long, Mistral-product-branded prompt naming "Ministral-3-8B-
// Instruct-2512" and "Le Chat" by identity, with {today}/{yesterday} date substitutions) is
// deliberately NOT replicated here, the same documented divergence ChatML() already has for
// Qwen's own default system message (chat.go's own "byte-exact" scope note) — injecting
// Mistral's own product identity into a self-hosted goinfer response would be actively wrong, not
// just an omission. The no-system-message case is therefore NOT byte-exact for this family
// (goldens below cover explicit-system cases only, same convention as ChatML's).
//
// Tool calling is NOT wired for this family (SupportsTools() declines it, chat/tools.go): the
// real wire format ([TOOL_CALLS]name[ARGS]{json} per call, no JSON-array wrapping) differs from
// the existing "mistral" tool dialect (a JSON array of call objects) enough that reusing it would
// silently mis-render/mis-parse rather than simply be incomplete — declining is the honest choice
// until that dialect is built for real, not a guess dressed up as support.
func Ministral() *Template {
	return &Template{name: "ministral", stops: []string{"</s>"}, render: func(system string, turns []Turn) []Segment {
		var sb strings.Builder
		sb.WriteString("<s>")
		if system != "" {
			sb.WriteString("[SYSTEM_PROMPT]" + system + "[/SYSTEM_PROMPT]")
		}
		for _, t := range turns {
			if t.Role == "assistant" {
				sb.WriteString(t.Content + "</s>")
				continue
			}
			sb.WriteString("[INST]" + t.Content + "[/INST]")
		}
		return []Segment{{Text: sb.String(), Special: true}}
	}}
}

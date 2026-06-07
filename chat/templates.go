package chat

import "strings"

// Each constructor returns the family's renderer. The render funcs are written
// to match HuggingFace apply_chat_template byte-for-byte (testdata/chat_goldens).

// Gemma3 — "<bos>" then per turn "<start_of_turn>{role}\n{content}<end_of_turn>\n"
// (assistant→model); no system role, so the system is folded into the first user
// turn ("{system}\n\n{content}"). Generation prompt: "<start_of_turn>model\n".
func Gemma3() *Template {
	return &Template{name: "gemma3", stops: []string{"<end_of_turn>"}, render: func(system string, turns []Turn) string {
		var b strings.Builder
		b.WriteString("<bos>")
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
			b.WriteString("<start_of_turn>" + role + "\n" + content + "<end_of_turn>\n")
		}
		b.WriteString("<start_of_turn>model\n")
		return b.String()
	}}
}

// Gemma4 — Gemma 3's successor with new markers: "<bos>", a real system turn
// "<|turn>system\n{system}<turn|>\n", per turn "<|turn>{role}\n{content}<turn|>\n"
// (assistant→model), ending with the generation prompt plus Gemma 4's thinking
// scaffold: "<|turn>model\n<|channel>thought\n<channel|>".
func Gemma4() *Template {
	return &Template{name: "gemma4", stops: []string{"<turn|>"}, render: func(system string, turns []Turn) string {
		var b strings.Builder
		b.WriteString("<bos>")
		if system != "" {
			b.WriteString("<|turn>system\n" + system + "<turn|>\n")
		}
		for _, t := range turns {
			role := "user"
			if t.Role == "assistant" {
				role = "model"
			}
			b.WriteString("<|turn>" + role + "\n" + t.Content + "<turn|>\n")
		}
		b.WriteString("<|turn>model\n<|channel>thought\n<channel|>")
		return b.String()
	}}
}

// ChatML (Qwen and most byte-level families) — per turn
// "<|im_start|>{role}\n{content}<|im_end|>\n", a leading system turn when given,
// generation prompt "<|im_start|>assistant\n". No BOS in the template.
func ChatML() *Template {
	return &Template{name: "chatml", stops: []string{"<|im_end|>"}, render: func(system string, turns []Turn) string {
		var b strings.Builder
		if system != "" {
			b.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
		}
		for _, t := range turns {
			b.WriteString("<|im_start|>" + t.Role + "\n" + t.Content + "<|im_end|>\n")
		}
		b.WriteString("<|im_start|>assistant\n")
		return b.String()
	}}
}

// Llama3 — "<|begin_of_text|>", an always-present system block carrying the
// date preamble (then the system text), per turn
// "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>", and the
// assistant generation header. The "Today Date" is the current date.
func Llama3() *Template {
	return &Template{name: "llama3", stops: []string{"<|eot_id|>"}, render: func(system string, turns []Turn) string {
		var b strings.Builder
		date := timeNow().Format("02 Jan 2006")
		b.WriteString("<|begin_of_text|>")
		b.WriteString("<|start_header_id|>system<|end_header_id|>\n\n")
		b.WriteString("Cutting Knowledge Date: December 2023\nToday Date: " + date + "\n\n")
		b.WriteString(system + "<|eot_id|>")
		for _, t := range turns {
			b.WriteString("<|start_header_id|>" + t.Role + "<|end_header_id|>\n\n" + t.Content + "<|eot_id|>")
		}
		b.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
		return b.String()
	}}
}

// Mistral — "<s>" once, each user turn "[INST] {content}[/INST]", each assistant
// turn " {content}</s>". No system role: the system is folded into the LAST user
// turn ("{system}\n\n{content}").
func Mistral() *Template {
	return &Template{name: "mistral", stops: []string{"</s>"}, render: func(system string, turns []Turn) string {
		lastUser := -1
		for i, t := range turns {
			if t.Role != "assistant" {
				lastUser = i
			}
		}
		var b strings.Builder
		b.WriteString("<s>")
		for i, t := range turns {
			if t.Role == "assistant" {
				b.WriteString(" " + t.Content + "</s>")
				continue
			}
			content := t.Content
			if i == lastUser && system != "" {
				content = system + "\n\n" + content
			}
			b.WriteString("[INST] " + content + "[/INST]")
		}
		return b.String()
	}}
}

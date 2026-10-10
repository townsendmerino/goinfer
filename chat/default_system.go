package chat

import "regexp"

// chatMLDefaultSystemRe finds a ChatML template's DEFAULT system block: `<|im_start|>system` + a newline (a real one, or
// the two characters `\n` inside a Jinja string literal) + plain literal text + `<|im_end|>`. A block built from the
// conversation (`'<|im_start|>system\n' + messages[0]['content'] + ...`) carries quotes, braces or a `+`, which the text
// class excludes, so only a template's own fixed default matches.
var chatMLDefaultSystemRe = regexp.MustCompile(`<\|im_start\|>system(?:\\n|\n)([^<>{}'"+\\]+?)<\|im_end\|>`)

// chatMLDefaultSystem returns the system message a ChatML template inserts when the conversation has none, "" when it
// inserts none. Qwen 2.5's templates do ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."; Qwen2.5-VL's
// "You are a helpful assistant."); Qwen 3's do not. Taken from the checkpoint's own template, never assumed per family:
// ChatML() is generic, and a default is only right for the model whose template declares it (owner, 2026-10-09: "every
// template with one").
func chatMLDefaultSystem(tmpl string) string {
	if m := chatMLDefaultSystemRe.FindStringSubmatch(tmpl); m != nil {
		return m[1]
	}
	return ""
}

// systemOr is the system message to render: the caller's, or the template's default when the caller sent none.
func (t *Template) systemOr(system string) string {
	if system == "" {
		return t.defaultSystem
	}
	return system
}

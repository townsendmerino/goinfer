package chat

import (
	"encoding/json"
	"os"
	"testing"
)

// The checkpoint's own default system message (owner, 2026-10-09: "every template with one"): Detect reads it from the
// template, and the no-system rendering is then Qwen 2.5's own, byte for byte (the chatml golden's no_system case was
// rendered by that template). The generic ChatML() still has no default: TestChatML_noSystem_documentedDivergence.
func TestDetect_chatMLDefaultSystemIsTheTemplates(t *testing.T) {
	g := loadToolGoldenPlain(t, "chatml")
	if g == nil {
		return
	}
	raw, _ := os.ReadFile("../testdata/chat_goldens/chatml.json")
	var src struct {
		ChatTemplate string `json:"chat_template"`
	}
	if err := json.Unmarshal(raw, &src); err != nil || src.ChatTemplate == "" {
		t.Fatalf("chatml.json carries no chat_template: %v", err)
	}
	tm, err := Detect(Meta{ChatTemplate: src.ChatTemplate})
	if err != nil {
		t.Fatal(err)
	}
	want := goldenCase(g, "no_system")
	if got := tm.Render("", []Turn{{Role: "user", Content: "What is the capital of France?"}}); got != want {
		t.Errorf("no-system rendering differs from Qwen 2.5's own template:\n got %q\nwant %q", got, want)
	}
	// An explicit system message is the caller's, never replaced by the default.
	if got, want := tm.Render("Be brief.", []Turn{{Role: "user", Content: "hi"}}), ChatML().Render("Be brief.", []Turn{{Role: "user", Content: "hi"}}); got != want {
		t.Errorf("an explicit system message changed: %q, want %q", got, want)
	}
}

func TestChatMLDefaultSystem_extraction(t *testing.T) {
	for _, c := range []struct{ name, tmpl, want string }{
		// Qwen2.5-VL's chat_template.json, decoded: raw template text, a real newline.
		{"raw text", "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}", "You are a helpful assistant."},
		// Qwen 2.5's text template: inside a Jinja string literal, the two characters \n.
		{"jinja literal", `{{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}`, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
		// Qwen 3's: the system block is built from the conversation only. No default.
		{"from messages", `{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}`, ""},
		{"no system block", "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>", ""},
	} {
		if got := chatMLDefaultSystem(c.tmpl); got != c.want {
			t.Errorf("%s: %q, want %q", c.name, got, c.want)
		}
	}
}

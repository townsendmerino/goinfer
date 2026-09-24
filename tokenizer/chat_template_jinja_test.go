package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// A checkpoint saved by recent transformers keeps its chat template in chat_template.jinja beside
// tokenizer_config.json, not under the config's chat_template key. The loader must pick it up, or
// chat.Detect sees no template (docs/tool-call-coverage.md, finding 1): Granite-4.0-H and Laguna
// XS.2 fell to raw completion, and the M-36 fingerprints could be bypassed.
func TestChatTemplateJinja(t *testing.T) {
	blob := []byte(`{"model":{"type":"BPE","vocab":{"<s>":0,"a":1},"merges":[]},` +
		`"decoder":{"type":"ByteLevel"},"pre_tokenizer":{"type":"ByteLevel"}}`)
	const jinja = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>{% endfor %}"
	cases := []struct {
		name, config, jinja, want string
	}{
		{"jinja only, with a config", `{"bos_token":"<s>"}`, jinja, jinja},
		{"jinja only, no config file", "", jinja, jinja},
		{"config key wins over the file", `{"chat_template":"FROM-CONFIG"}`, jinja, "FROM-CONFIG"},
		{"neither", `{"bos_token":"<s>"}`, "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), blob, 0o644); err != nil {
				t.Fatal(err)
			}
			if c.config != "" {
				if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(c.config), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if c.jinja != "" {
				if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(c.jinja), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			tk, err := Load(dir)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if got := tk.ChatTemplate(); got != c.want {
				t.Errorf("ChatTemplate() = %q, want %q", got, c.want)
			}
		})
	}
	// No-sibling mode (a .giw's embedded tokenizer.json, M-14) must still read nothing from disk,
	// the jinja file included — even with one sitting in the process CWD.
	dir := t.TempDir()
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("chat_template.jinja", []byte(jinja), 0o644); err != nil {
		t.Fatal(err)
	}
	tk, err := LoadJSONBytes(blob)
	if err != nil {
		t.Fatal(err)
	}
	if tk.ChatTemplate() != "" {
		t.Error("LoadJSONBytes adopted a chat_template.jinja from the CWD (M-14)")
	}
}

package tokenizer

import "testing"

// TestChatStyle pins the vocab-marker detection the demos use to pick a chat
// template — including Gemma 4's new <|turn>/<turn|> markers (it replaced Gemma
// 3's <start_of_turn>/<end_of_turn>).
func TestChatStyle(t *testing.T) {
	for _, tc := range []struct {
		name  string
		vocab map[string]int32
		want  ChatStyle
	}{
		{"chatml", map[string]int32{"<|im_start|>": 1, "<|im_end|>": 2}, ChatStyleChatML},
		{"gemma3", map[string]int32{"<start_of_turn>": 105, "<end_of_turn>": 106}, ChatStyleGemma},
		{"gemma4", map[string]int32{"<|turn>": 105, "<turn|>": 106}, ChatStyleGemma4},
		{"none", map[string]int32{"hello": 1}, ChatStyleNone},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tk := &Tokenizer{vocab: tc.vocab}
			if got := tk.ChatStyle(); got != tc.want {
				t.Errorf("ChatStyle() = %d, want %d", got, tc.want)
			}
		})
	}
}

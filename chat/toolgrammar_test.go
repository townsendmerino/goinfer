package chat

import (
	"encoding/json"
	"testing"
)

// TestTemplate_ToolGrammar: the auto-mode tool masker leaves prose alone and arms once the template's call opener is written
// (ChatML), arms on a call object's name key for a family with no opener (Llama 3), and is refused for a template with no
// JSON call form (Gemma 4).
func TestTemplate_ToolGrammar(t *testing.T) {
	tokens := [][]byte{[]byte("Sure"), []byte(","), []byte(" here"), []byte("<tool_call>"), []byte("\n"), []byte(`{"name": "`), []byte("x")}
	tools := []Tool{{Name: "lookup", Parameters: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)}}

	lazy, err := ChatML().ToolGrammar(tools, tokens, nil)
	if err != nil {
		t.Fatalf("ChatML: %v", err)
	}
	if lazy.Gate([]int{0, 1, 2}) {
		t.Error("ChatML: armed on prose")
	}
	if !lazy.Gate([]int{0, 1, 2, 3}) {
		t.Error("ChatML: not armed after <tool_call>")
	}

	lazy, err = Llama3().ToolGrammar(tools, tokens, nil)
	if err != nil {
		t.Fatalf("Llama 3: %v", err)
	}
	if lazy.Gate([]int{6}) {
		t.Error("Llama 3: armed on an output that does not begin as a call")
	}
	if lazy, err = Llama3().ToolGrammar(tools, tokens, nil); err != nil || !lazy.Gate([]int{5}) {
		t.Errorf("Llama 3: not armed on an output that begins with a call's name key (err %v)", err)
	}

	if _, err := Gemma4().ToolGrammar(tools, tokens, nil); err == nil {
		t.Error("Gemma 4 has no JSON call form; ToolGrammar should refuse it")
	}
}

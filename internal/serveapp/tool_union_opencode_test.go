package serveapp

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// agentToolsModel is a model with Mellum2.1's template (the chatml <tool_call> form) and the scripted tokenizer, and
// agentTools the tool list of tack's E1 pilot: opencode's built-in tools and ken-mcp's.
func agentToolsModel(t *testing.T) *loadedModel {
	t.Helper()
	tk := scriptTokenizer(t)
	lm := &loadedModel{tk: tk, vocab: tk.Size(), name: "m"} // vocab sizes the token-bytes table the masker reads
	lm.tmpl = mellum21Template(t)
	if _, _, _, _, ok := lm.tmpl.ToolCallWrapper(); !ok {
		t.Fatal("the template has no constrainable call form: not the path under test")
	}
	return lm
}

func agentTools(t *testing.T) []chat.Tool {
	t.Helper()
	raw, err := os.ReadFile("../../constrain/testdata/opencode-ken-tools.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Tools []struct {
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	var tools []chat.Tool
	for _, tl := range doc.Tools {
		tools = append(tools, chat.Tool{Name: tl.Function.Name, Description: tl.Function.Description, Parameters: tl.Function.Parameters})
	}
	return tools
}

// TestToolUnion_armsOnAgentTools: a request carrying the agent tool list gets the call constraint the route promises:
// under auto the lazy union, gated on the opener; under required the union from the first token. Before the tool path
// compiled leniently, these tools' schemas failed it and every call went unconstrained.
func TestToolUnion_armsOnAgentTools(t *testing.T) {
	lm, tools := agentToolsModel(t), agentTools(t)
	var auto genRequest
	constrainToolUnion(lm, &auto, "auto", tools)
	if auto.sp.LogitProcessor == nil || auto.sp.LogitProcessorGate == nil {
		t.Errorf("auto over %d agent tools: no gated call constraint", len(tools))
	}
	var req genRequest
	constrainToolUnion(lm, &req, "required", tools)
	if req.masker == nil {
		t.Errorf("required over %d agent tools: no call constraint", len(tools))
	}
}

// TestToolUnion_autoLeavesProseFree: under auto with the agent tool list, a reply can still be plain text. Fed a prose
// reply token by token, as the decoder feeds them, the gate stays shut and the processor leaves every logit as it was. Only
// once the model has written the call opener does the gate open and the processor mask.
func TestToolUnion_autoLeavesProseFree(t *testing.T) {
	lm, tools := agentToolsModel(t), agentTools(t)
	var gr genRequest
	constrainToolUnion(lm, &gr, "auto", tools)
	gate, proc := gr.sp.LogitProcessorGate, gr.sp.LogitProcessor
	if gate == nil || proc == nil {
		t.Fatal("auto has no gated processor: the constraint would apply from the first token")
	}
	vocab := len(lm.cachedTokenBytes())
	if vocab < 100 {
		t.Fatalf("a token-bytes table of %d entries: the logit checks below would be vacuous", vocab)
	}
	base := make([]float32, vocab)
	for i := range base {
		base[i] = float32(i%97) / 97
	}
	masked := func(gen []int) int {
		l := append([]float32(nil), base...)
		proc(gen, l)
		n := 0
		for i := range l {
			if l[i] != base[i] {
				n++
			}
		}
		return n
	}
	prose, err := lm.tk.Encode("I will read main.go first, then explain what prepare does and why top_p 0 is greedy.", false)
	if err != nil {
		t.Fatal(err)
	}
	var gen []int
	for i, id := range prose {
		if gate(gen) {
			t.Fatalf("the gate opened after %d prose tokens", i)
		}
		if n := masked(gen); n != 0 {
			t.Fatalf("after %d prose tokens the processor changed %d logits: prose is constrained", i, n)
		}
		gen = append(gen, id)
	}
	opener, err := lm.tk.Encode("<tool_call>\n{\"name\": \"", false)
	if err != nil {
		t.Fatal(err)
	}
	gen = append(gen, opener...)
	if !gate(gen) {
		t.Fatal("the call opener did not open the gate")
	}
	if n := masked(gen); n == 0 {
		t.Error("after the opener the processor masked nothing")
	}
}

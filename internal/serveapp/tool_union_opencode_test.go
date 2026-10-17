package serveapp

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// TestToolUnion_armsOnAgentTools: a request carrying opencode's and ken-mcp's tools (tack's E1 pilot) gets the call
// constraint the route promises. Under auto, the lazy union arms on the opener; under required, the union runs from the
// first token. Before the tool path compiled leniently, these tools' schemas failed it, and every call went unconstrained.
func TestToolUnion_armsOnAgentTools(t *testing.T) {
	lm := &loadedModel{tk: scriptTokenizer(t), name: "m"} // Mellum2.1's template: the chatml <tool_call> form
	lm.tmpl = mellum21Template(t)
	if _, _, _, _, ok := lm.tmpl.ToolCallWrapper(); !ok {
		t.Fatal("the tiny model's template has no constrainable call form: not the path under test")
	}
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

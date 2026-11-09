package chat

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/constrain"
)

// ToolGrammar is the constrained decoding for a turn that offers tools and lets the model answer in prose (tool_choice
// "auto"): a lazy masker that masks nothing until this template's call opener arrives, then holds the call to the offered
// tools' schemas, and disarms after each complete call. A family whose call has no opener (Llama 3's bare JSON) arms instead
// when the output begins as a call object. tokens is the vocabulary's bytes (constrain.TokenBytes) and stopIDs the ids that
// end the turn (StopIDs). Set decoder.SamplingParams.LogitProcessor to its Process and LogitProcessorGate to its Gate. An
// error means this template has no JSON call form to constrain to, or a tool's schema does not compile; decode unconstrained.
// A turn that must call a tool is constrain.ToolCallsGrammar with ToolCallWrapper's prefix, held from the first token.
func (t *Template) ToolGrammar(tools []Tool, tokens [][]byte, stopIDs []int) (*constrain.LazyMasker, error) {
	prefix, suffix, argsKey, array, ok := t.ToolCallWrapper()
	if !ok {
		return nil, fmt.Errorf("chat: template %q has no JSON tool-call form to constrain", t.name)
	}
	specs := make([]constrain.ToolSpec, len(tools))
	for i, tl := range tools {
		specs[i] = constrain.ToolSpec{Name: tl.Name, Parameters: tl.Parameters}
	}
	trigger := strings.TrimRight(prefix, " \t\r\n")
	g, err := constrain.ToolCallsGrammar(trigger, suffix, argsKey, array, specs)
	if err != nil {
		return nil, err
	}
	m := constrain.NewMasker(g, tokens, stopIDs)
	if trigger == "" {
		return constrain.NewCallKeyLazyMasker(m, "<|python_tag|>"), nil
	}
	return constrain.NewLazyMasker(m, trigger), nil
}

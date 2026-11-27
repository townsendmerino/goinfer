package chat

import "github.com/townsendmerino/goinfer/internal/chat/gemma4tools"

const gq = gemma4tools.Quote

var (
	detectGemma4ContentBeforeCalls = gemma4tools.DetectContentBeforeCalls
	gemmaArgs                      = gemma4tools.Args
	gemmaBodyToJSON                = gemma4tools.BodyToJSON
)

func (t *Template) renderGemma4Tools(system string, turns []Turn, tools []Tool) []Segment {
	old := t.reason != nil && t.reason.oldGemma4
	return gemma4tools.RenderTools(t.has, system, toCoreTurns(turns), toCoreTools(tools), gemma4tools.Opts{
		Think:    t.gemma4Think(),
		Hist:     t.historyKind() == histGemma4 || old,
		Old:      old,
		Scaffold: t.reason.offScaffold(),
	})
}

func (t *Template) renderGemma4NativeTools(system string, turns []Turn, tools []Tool, think bool) []Segment {
	return gemma4tools.RenderNativeTools(t.has, system, toCoreTurns(turns), toCoreTools(tools), gemma4tools.NativeOpts{
		Think:              think,
		Scaffold:           t.reason.offScaffold(),
		ContentBeforeCalls: t.contentBeforeCalls,
	})
}

func parseGemma4Tools(out string) ([]ToolCall, string) {
	calls, lead := gemma4tools.ParseTools(out)
	return fromCoreCalls(calls), lead
}

package chat

import "github.com/townsendmerino/goinfer/internal/chat/chatcore"

type histKind = chatcore.HistKind

const (
	histNone     = chatcore.HistNone
	histQwen3    = chatcore.HistQwen3
	histQwen35   = chatcore.HistQwen35
	histGemma4   = chatcore.HistGemma4
	histMellum21 = chatcore.HistMellum21
)

var (
	detectHistoryKind        = chatcore.DetectHistoryKind
	detectGemma4History      = chatcore.DetectGemma4History
	splitTagged              = chatcore.SplitTagged
	stripChannels            = chatcore.StripChannels
	detectGroupedToolResults = chatcore.DetectGroupedToolResults
)

func (t *Template) historyKind() histKind {
	if t == nil || t.reason == nil || t.think == ThinkAsIs {
		return histNone
	}
	return t.reason.hist
}

func lastQueryIndex(turns []Turn) int {
	return chatcore.LastQueryIndex(toCoreTurns(turns))
}

func lastUserIndex(turns []Turn) int {
	return chatcore.LastUserIndex(toCoreTurns(turns))
}

func qwenAssistant(kind histKind, t Turn, idx, lq, n int) (block bool, reasoning, content string) {
	return chatcore.QwenAssistant(kind, toCoreTurn(t), idx, lq, n)
}

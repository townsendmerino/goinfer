package serveapp

import (
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

func (lm *loadedModel) prefixBoundary(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if lm.model == nil || !lm.model.KVSnapshotsSupported() {
		return 0
	}
	return lm.prefixBoundaryOf(tm, system, turns, tools, ids)
}

func (lm *loadedModel) recurrentKeep(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if !lm.keepsRecurrentSnapshots() {
		return 0
	}
	return wirecore.RecurrentKeep(lm.tk, tm, system, turns, tools, ids)
}

func (lm *loadedModel) recurrentMsgBoundary(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	if !lm.keepsRecurrentSnapshots() {
		return 0
	}
	return wirecore.RecurrentMsgBoundary(lm.tk, tm, system, turns, tools, ids)
}

func (lm *loadedModel) keepsRecurrentSnapshots() bool {
	if lm.scripted != nil {
		return lm.scriptedRecurrent
	}
	return lm.model != nil && lm.model.KeepsRecurrentSnapshots()
}

func (lm *loadedModel) prefixBoundaryOf(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) int {
	return wirecore.PrefixBoundary(lm.tk, tm, system, turns, tools, ids)
}

func (lm *loadedModel) prepareTurn(sm sampling, tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool, ids []int) (genRequest, error) {
	sm.FixedTokens = func() int { return lm.fixedPromptTokens(tm, system, tools) }
	gr, err := lm.prepare(sm, ids, lm.residentPath())
	if err != nil {
		return gr, err
	}
	gr.boundary = lm.prefixBoundary(tm, system, turns, tools, ids)
	gr.keep = lm.recurrentKeep(tm, system, turns, tools, ids)
	gr.msgBoundary = lm.recurrentMsgBoundary(tm, system, turns, tools, ids)
	return gr, nil
}

package serveapp

import (
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

type budgetFlag = streamframe.BudgetFlag

var (
	parseBudgetFlag = streamframe.ParseBudgetFlag
	resolveBudget   = streamframe.ResolveBudget
)

func (c *config) reasoningBudgetFlag() budgetFlag {
	f, _ := parseBudgetFlag(c.reasoningBudget)
	return f
}

func (s *server) routeThink(lm *loadedModel, gr *genRequest, tm *chat.Template, turns []chat.Turn, ts thinkSettings, onReasoning func(string)) {
	gr.think = newThinkOut(tm, turns, ts, onReasoning)
	s.applyBudget(lm, gr, tm, turns, ts)
}

func (s *server) applyBudget(lm *loadedModel, gr *genRequest, tm *chat.Template, turns []chat.Turn, ts thinkSettings) {
	if !tm.ThinkingPossible() || gr.masker != nil || lm.spec || lm.blockSpec != nil || lm.tk == nil {
		return
	}
	limit, ok := resolveBudget(s.cfg.reasoningBudgetFlag(), ts.Budget, gr.maxTokens)
	if !ok {
		return
	}
	b := tm.NewReasoningBudgetFor(lm.tk, turns, limit)
	if gr.callPostCheck {
		b = tm.NewCallBudgetFor(lm.tk, turns, limit)
	}
	if b == nil {
		return
	}
	prev, prevGate := gr.sp.LogitProcessor, gr.sp.LogitProcessorGate
	switch {
	case prev == nil:
		gr.sp.LogitProcessor, gr.sp.LogitProcessorGate = b.Process, b.Gate
	case prevGate != nil:
		gr.sp.LogitProcessor = func(g []int, l []float32) { prev(g, l); b.Process(g, l) }
		gr.sp.LogitProcessorGate = func(g []int) bool { return prevGate(g) || b.Gate(g) }
	}
}

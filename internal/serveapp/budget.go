package serveapp

// The reasoning budget: a ceiling on how long a reply may think, enforced by FORCING the block closed.
//
// Without it a thinking reply that runs out of max_tokens while still inside its block has no answer at all — content is
// empty, finish_reason is length — and a small max_tokens with thinking on (the default for Qwen3.5-9B and Qwen3) does
// exactly that. With it, once the block has used its budget the next token is forced to be the block's closing token; the
// model then writes its answer in the tokens that remain.
//
//	budget for a request = min(what the request asked for, what leaves room to answer)
//	  asked for:   thinking_token_budget (OpenAI-style body), Anthropic thinking.budget_tokens, else -reasoning-budget N
//	  room:        max_tokens minus max(2, max_tokens/4) — thinking never takes more than three quarters of the turn
//
// -reasoning-budget auto (the default) applies only the room rule, so a request that says nothing is still guaranteed an
// answer section; unlimited applies no ceiling of its own (an explicit request budget still holds); N caps every thinking
// reply at N tokens (and still the room rule).
//
// HOW: a gated logit processor (decoder.SamplingParams.LogitProcessor / LogitProcessorGate). Its gate is false until the
// budget is due, so every step before that keeps the on-device fast paths — a reply that finishes thinking early pays nothing.
// It is a pure function of the generated ids: it finds the block by the family's single open/close tokens, counts the tokens
// inside it, and once the count reaches the budget with the close token not yet written, masks every logit but the close
// token's. Skipped — with the budget simply not applying — when speculative decoding is on (any processor makes the spec paths
// fall back to plain decode, which would silently cost the operator the drafter on every thinking request), when a grammar
// masker governs the request, and when the tokenizer has no single token for the delimiters.

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
)

// budgetFlag is -reasoning-budget parsed.
type budgetFlag struct {
	unlimited bool
	n         int // > 0: a cap on every thinking reply; 0 with !unlimited means auto
}

func parseBudgetFlag(s string) (budgetFlag, error) {
	switch v := strings.ToLower(strings.TrimSpace(s)); v {
	case "", "auto":
		return budgetFlag{}, nil
	case "unlimited":
		return budgetFlag{unlimited: true}, nil
	default:
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return budgetFlag{}, fmt.Errorf("want auto, unlimited, or a positive number of tokens")
		}
		return budgetFlag{n: n}, nil
	}
}

func (c config) reasoningBudgetFlag() budgetFlag {
	f, _ := parseBudgetFlag(c.reasoningBudget)
	return f
}

// resolveBudget is the ceiling on reasoning tokens for a request whose turn is maxTokens long, or ok=false when none applies.
// explicit is the request's own budget (0 = none asked for).
func resolveBudget(f budgetFlag, explicit, maxTokens int) (limit int, ok bool) {
	if maxTokens < 4 {
		return 0, false // nothing worth protecting: the whole turn is a few tokens
	}
	room := maxTokens - max(2, maxTokens/4) // thinking leaves at least a quarter of the turn (min 2 tokens) for the answer
	limit = room
	if explicit > 0 {
		limit = min(explicit, room)
	} else if f.n > 0 {
		limit = min(f.n, room)
	} else if f.unlimited {
		return 0, false
	}
	return limit, limit >= 1
}

// reasoningBudget is the processor. opens means the PROMPT already ends inside an open block (counting starts at the first
// generated token); otherwise the block starts after the first open token the model writes.
type reasoningBudget struct {
	open, closeID int
	opens         bool
	limit         int
}

// due reports whether the next token must be the close token: inside an unclosed block that has used its budget.
func (b *reasoningBudget) due(generated []int) bool {
	start := 0
	if !b.opens {
		i := indexOfID(generated, b.open)
		if i < 0 {
			return false // the model has not opened a block (and may never)
		}
		start = i + 1
	}
	inside := generated[start:]
	if len(inside) < b.limit {
		return false
	}
	return indexOfID(inside, b.closeID) < 0
}

func indexOfID(ids []int, id int) int {
	for i, v := range ids {
		if v == id {
			return i
		}
	}
	return -1
}

// Process is a decoder.SamplingParams.LogitProcessor. It is safe to call on every step (the vision decode loop ignores the
// gate): it does nothing unless the budget is due.
func (b *reasoningBudget) Process(generated []int, logits []float32) {
	if !b.due(generated) || b.closeID < 0 || b.closeID >= len(logits) {
		return
	}
	neg := float32(math.Inf(-1))
	for i := range logits {
		logits[i] = neg
	}
	logits[b.closeID] = 0
}

// Gate is a decoder.SamplingParams.LogitProcessorGate.
func (b *reasoningBudget) Gate(generated []int) bool { return b.due(generated) }

// routeThink wires one generation's reasoning: the splitter that separates it (think.go) and, where it applies, the budget
// that bounds it. Every route calls it in place of newThinkOut.
func (s *server) routeThink(lm *loadedModel, gr *genRequest, tm *chat.Template, turns []chat.Turn, ts thinkSettings, onReasoning func(string)) {
	gr.think = newThinkOut(tm, turns, ts, onReasoning)
	s.applyBudget(lm, gr, tm, turns, ts)
}

func (s *server) applyBudget(lm *loadedModel, gr *genRequest, tm *chat.Template, turns []chat.Turn, ts thinkSettings) {
	if !tm.ThinkingPossible() || gr.masker != nil || lm.spec || lm.blockSpec != nil || lm.tk == nil {
		return
	}
	r := tm.Reasoning()
	openID, ok1 := lm.tk.TokenID(r.OpenToken())
	closeID, ok2 := lm.tk.TokenID(r.CloseToken())
	if !ok1 || !ok2 {
		return
	}
	limit, ok := resolveBudget(s.cfg.reasoningBudgetFlag(), ts.budget, gr.maxTokens)
	if !ok {
		return
	}
	b := &reasoningBudget{open: openID, closeID: closeID, opens: tm.PromptOpensThinkFor(turns), limit: limit}
	prev, prevGate := gr.sp.LogitProcessor, gr.sp.LogitProcessorGate
	switch {
	case prev == nil:
		gr.sp.LogitProcessor, gr.sp.LogitProcessorGate = b.Process, b.Gate
	case prevGate != nil: // the lazy tool union: run it first, then let a due budget force the close token over its mask
		gr.sp.LogitProcessor = func(g []int, l []float32) { prev(g, l); b.Process(g, l) }
		gr.sp.LogitProcessorGate = func(g []int) bool { return prevGate(g) || b.Gate(g) }
	}
	// prev != nil with no gate is an ungated constraint: it owns every step, and the budget stays out of its way.
}

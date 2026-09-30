package chat

import "math"

// The reasoning budget: a ceiling on how long a reply may think, enforced by forcing the block closed. Without it a thinking reply
// that runs out of max_tokens while still inside its block has no answer at all. With it, once the block has used its budget the
// next token is forced to be the block's closing token and the model writes its answer in the tokens that remain.
//
// ReasoningBudget is the processor (a decoder.SamplingParams.LogitProcessor plus its LogitProcessorGate, which keeps the decode fast
// paths on until the budget is due); BudgetRoom is the rule for how much of a turn thinking may take. serve (policy: flags and
// request fields), goinfer-chat and the demo agent all build on the two, so the rule is written once.

// BudgetRoom is the most reasoning tokens a turn of maxTokens may spend and still leave room to answer: three quarters of the
// turn, leaving at least 2 tokens. ok is false for a turn under 4 tokens, where there is nothing worth protecting.
func BudgetRoom(maxTokens int) (room int, ok bool) {
	if maxTokens < 4 {
		return 0, false
	}
	return maxTokens - max(2, maxTokens/4), true
}

// ReasoningBudget is the processor. opens means the PROMPT already ends inside an open block (counting starts at the first generated
// token); otherwise the block starts after the first open token the model writes.
type ReasoningBudget struct {
	open, closeID int
	opens         bool
	limit         int
}

// NewReasoningBudget builds the processor from the block's single open and close token ids.
func NewReasoningBudget(openID, closeID int, promptOpens bool, limit int) *ReasoningBudget {
	return &ReasoningBudget{open: openID, closeID: closeID, opens: promptOpens, limit: limit}
}

// TokenLookup is the part of a tokenizer the budget needs.
type TokenLookup interface {
	TokenID(text string) (int, bool)
}

// NewReasoningBudgetFor builds the budget for a reply to a prompt rendered by t from turns, or nil when it cannot apply: t has no
// reasoning spec, thinking is impossible under its mode, the tokenizer has no single token for the delimiters, or limit < 1.
func (t *Template) NewReasoningBudgetFor(tk TokenLookup, turns []Turn, limit int) *ReasoningBudget {
	if !t.ThinkingPossible() || limit < 1 || tk == nil {
		return nil
	}
	r := t.Reasoning()
	openID, ok1 := tk.TokenID(r.OpenToken())
	closeID, ok2 := tk.TokenID(r.CloseToken())
	if !ok1 || !ok2 {
		return nil
	}
	return NewReasoningBudget(openID, closeID, t.PromptOpensThinkFor(turns), limit)
}

// due reports whether the next token must be the close token: inside an unclosed block that has used its budget.
func (b *ReasoningBudget) due(generated []int) bool {
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

// Process is a decoder.SamplingParams.LogitProcessor. It is safe to call on every step (a decode loop that ignores the gate does):
// it does nothing unless the budget is due.
func (b *ReasoningBudget) Process(generated []int, logits []float32) {
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
func (b *ReasoningBudget) Gate(generated []int) bool { return b.due(generated) }

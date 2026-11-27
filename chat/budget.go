package chat

import "github.com/townsendmerino/goinfer/internal/chat/reasoning"

// ReasoningBudget caps reasoning length by forcing the closing delimiter or channel transition once limit tokens are spent.
type ReasoningBudget = reasoning.Budget

// TokenLookup looks up single-token vocabulary IDs.
type TokenLookup = reasoning.TokenLookup

// SegmentEncoder extends TokenLookup with multi-token segment encoding.
type SegmentEncoder = reasoning.SegmentEncoder

const (
	hmStart              = reasoning.HarmonyStart
	hmChannel            = reasoning.HarmonyChannel
	hmMessage            = reasoning.HarmonyMessage
	hmEnd                = reasoning.HarmonyEnd
	harmonyCallRecipient = reasoning.HarmonyCallRecipient
)

var (
	// BudgetRoom returns the maximum reasoning tokens a turn of maxTokens may spend while leaving room to answer.
	BudgetRoom = reasoning.BudgetRoom
	// NewReasoningBudget builds a delimiter-based reasoning budget from single open and close token IDs.
	NewReasoningBudget = reasoning.NewBudget
	// NewHarmonyBudget builds a channel-transition reasoning budget for gpt-oss replies.
	NewHarmonyBudget = reasoning.NewHarmonyBudget
	harmonyBudgetFor = reasoning.HarmonyBudgetFor
)

// NewReasoningBudgetFor builds the reasoning budget for replies to prompts rendered by t from turns, or nil when none applies.
func (t *Template) NewReasoningBudgetFor(tk TokenLookup, turns []Turn, limit int) *ReasoningBudget {
	if !t.ThinkingPossible() || limit < 1 || tk == nil {
		return nil
	}
	if t.name == "harmony" {
		return harmonyBudgetFor(tk, limit)
	}
	r := t.Reasoning()
	if r == nil {
		return nil
	}
	openID, ok1 := tk.TokenID(r.OpenToken())
	closeID, ok2 := tk.TokenID(r.CloseToken())
	if !ok1 || !ok2 {
		return nil
	}
	return NewReasoningBudget(openID, closeID, t.PromptOpensThinkFor(turns), limit)
}

// NewCallBudgetFor builds a reasoning budget for replies that must produce an unconstrained tool call.
func (t *Template) NewCallBudgetFor(tk SegmentEncoder, turns []Turn, limit int) *ReasoningBudget {
	b := t.NewReasoningBudgetFor(tk, turns, limit)
	if b == nil || t.name != "harmony" {
		return b
	}
	return reasoning.HarmonyCallBudgetFor(tk, b, limit)
}

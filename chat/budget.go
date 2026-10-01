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

	// Harmony (gpt-oss) variant, set by NewHarmonyBudget: the analysis message opens with a three-token HEADER and is closed, and
	// the answer opened, by a SEQUENCE the model would write itself — so forcing one close token is not enough (the model could
	// open another channel after it). head is what the reply must begin with for a block to exist; force is written, one token
	// per step, starting at generated-token index limit-len(force), so that it ends exactly at limit and the answer keeps the
	// rest of the turn. A natural <|end|> before that point means the model closed its own analysis: nothing is forced.
	head, force []int
	endID       int
}

// NewReasoningBudget builds the processor from the block's single open and close token ids.
func NewReasoningBudget(openID, closeID int, promptOpens bool, limit int) *ReasoningBudget {
	return &ReasoningBudget{open: openID, closeID: closeID, opens: promptOpens, limit: limit}
}

// NewHarmonyBudget builds the processor for a gpt-oss reply. head is the analysis header's ids (<|channel|> analysis <|message|>),
// endID the <|end|> that closes a message, force the ids written once the budget is due (<|end|> <|start|> assistant <|channel|>
// final <|message|>), and limit the number of generated tokens by which the answer must have been opened — header and forced
// sequence included, which is what makes the answer's share of the turn what BudgetRoom promised. nil when limit leaves no room
// for even one reasoning token.
func NewHarmonyBudget(head []int, endID int, force []int, limit int) *ReasoningBudget {
	if len(force) == 0 || limit-len(force) < len(head)+1 {
		return nil
	}
	return &ReasoningBudget{head: head, endID: endID, force: force, limit: limit}
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
	if t.name == "harmony" {
		return harmonyBudgetFor(tk, limit)
	}
	r := t.Reasoning()
	openID, ok1 := tk.TokenID(r.OpenToken())
	closeID, ok2 := tk.TokenID(r.CloseToken())
	if !ok1 || !ok2 {
		return nil
	}
	return NewReasoningBudget(openID, closeID, t.PromptOpensThinkFor(turns), limit)
}

// harmonyBudgetFor looks up the seven single-token words of the gpt-oss channel protocol; with any missing there is no budget.
func harmonyBudgetFor(tk TokenLookup, limit int) *ReasoningBudget {
	id := func(s string) int {
		v, ok := tk.TokenID(s)
		if !ok {
			return -1
		}
		return v
	}
	channel, analysis, message, end, start, assistant, final := id(hmChannel), id("analysis"), id(hmMessage), id(hmEnd), id(hmStart), id("assistant"), id("final")
	for _, v := range []int{channel, analysis, message, end, start, assistant, final} {
		if v < 0 {
			return nil
		}
	}
	return NewHarmonyBudget([]int{channel, analysis, message}, end, []int{end, start, assistant, channel, final, message}, limit)
}

// forcedHarmony returns the token that must be written next, or -1: the reply opened with the analysis header, the model has not
// closed that message by itself before the trigger point, and the forced sequence is not yet complete.
func (b *ReasoningBudget) forcedHarmony(generated []int) int {
	trigger := b.limit - len(b.force)
	n := len(generated)
	if n < trigger || n >= trigger+len(b.force) || len(generated) < len(b.head) {
		return -1
	}
	for i, h := range b.head {
		if generated[i] != h {
			return -1 // the reply did not open with an analysis message
		}
	}
	if indexOfID(generated[:trigger], b.endID) >= 0 {
		return -1 // the model closed its own analysis in time
	}
	return b.force[n-trigger]
}

// due reports whether the next token must be forced: inside an unclosed block that has used its budget.
func (b *ReasoningBudget) due(generated []int) bool {
	if b.force != nil {
		return b.forcedHarmony(generated) >= 0
	}
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
	next := b.closeID
	if b.force != nil {
		next = b.forcedHarmony(generated)
		if next < 0 {
			return
		}
	} else if !b.due(generated) {
		return
	}
	if next < 0 || next >= len(logits) {
		return
	}
	neg := float32(math.Inf(-1))
	for i := range logits {
		logits[i] = neg
	}
	logits[next] = 0
}

// Gate is a decoder.SamplingParams.LogitProcessorGate.
func (b *ReasoningBudget) Gate(generated []int) bool { return b.due(generated) }

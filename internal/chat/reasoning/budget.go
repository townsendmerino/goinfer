package reasoning

import (
	"math"
	"slices"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Harmony channel markers and call recipient prefix used by Harmony budgets.
const (
	HarmonyStart         = "<|start|>"
	HarmonyChannel       = "<|channel|>"
	HarmonyMessage       = "<|message|>"
	HarmonyEnd           = "<|end|>"
	HarmonyCallRecipient = "commentary to=functions."
)

// BudgetRoom returns the maximum reasoning tokens a turn of maxTokens may spend while leaving room to answer.
func BudgetRoom(maxTokens int) (room int, ok bool) {
	if maxTokens < 4 {
		return 0, false
	}
	return maxTokens - max(2, maxTokens/4), true
}

// TokenLookup looks up single-token vocabulary IDs.
type TokenLookup interface {
	TokenID(text string) (int, bool)
}

// SegmentEncoder extends TokenLookup with multi-token segment encoding.
type SegmentEncoder interface {
	TokenLookup
	EncodeSegments(segs []tokenizer.Segment, addBOS bool) ([]int, error)
}

// Budget caps reasoning length by forcing the closing delimiter or channel transition once limit tokens are spent.
type Budget struct {
	open, closeID int
	opens         bool
	limit         int

	head, force []int
	endID       int
}

// NewBudget builds a delimiter-based reasoning budget from single open and close token IDs.
func NewBudget(openID, closeID int, promptOpens bool, limit int) *Budget {
	return &Budget{open: openID, closeID: closeID, opens: promptOpens, limit: limit}
}

// NewHarmonyBudget builds a channel-transition reasoning budget for gpt-oss replies.
func NewHarmonyBudget(head []int, endID int, force []int, limit int) *Budget {
	if len(force) == 0 || limit-len(force) < len(head)+1 {
		return nil
	}
	return &Budget{head: head, endID: endID, force: force, limit: limit}
}

// HarmonyBudgetFor resolves the seven single-token Harmony channel words and returns a Budget for limit.
func HarmonyBudgetFor(tk TokenLookup, limit int) *Budget {
	id := func(s string) int {
		v, ok := tk.TokenID(s)
		if !ok {
			return -1
		}
		return v
	}
	channel, analysis, message, end, start, assistant, final := id(HarmonyChannel), id("analysis"), id(HarmonyMessage), id(HarmonyEnd), id(HarmonyStart), id("assistant"), id("final")
	for _, v := range []int{channel, analysis, message, end, start, assistant, final} {
		if v < 0 {
			return nil
		}
	}
	return NewHarmonyBudget([]int{channel, analysis, message}, end, []int{end, start, assistant, channel, final, message}, limit)
}

// HarmonyCallBudgetFor builds a Harmony budget whose forced transition opens a commentary function call.
func HarmonyCallBudgetFor(tk SegmentEncoder, plain *Budget, limit int) *Budget {
	if plain == nil {
		return nil
	}
	recipient, err := tk.EncodeSegments([]tokenizer.Segment{{Text: HarmonyCallRecipient}}, false)
	if err != nil || len(recipient) == 0 {
		return nil
	}
	id := func(s string) int { v, _ := tk.TokenID(s); return v }
	call := append([]int{id(HarmonyEnd), id(HarmonyStart), id("assistant"), id(HarmonyChannel)}, recipient...)
	return NewHarmonyBudget(plain.head, plain.endID, call, limit)
}

// Head returns the Harmony analysis header token IDs, if any.
func (b *Budget) Head() []int { return b.head }

// EndID returns the Harmony message-end token ID.
func (b *Budget) EndID() int { return b.endID }

// Force returns the forced token ID sequence for Harmony budgets.
func (b *Budget) Force() []int { return b.force }

// ForcedHarmony returns the Harmony token ID that must be emitted next, or -1 if none is due.
func (b *Budget) ForcedHarmony(generated []int) int {
	trigger := b.limit - len(b.force)
	n := len(generated)
	if n < trigger || n >= trigger+len(b.force) || len(generated) < len(b.head) {
		return -1
	}
	for i, h := range b.head {
		if generated[i] != h {
			return -1
		}
	}
	if slices.Contains(generated[:trigger], b.endID) {
		return -1
	}
	return b.force[n-trigger]
}

// Due reports whether the next generated token must be forced to close reasoning.
func (b *Budget) Due(generated []int) bool {
	if b.force != nil {
		return b.ForcedHarmony(generated) >= 0
	}
	start := 0
	if !b.opens {
		i := slices.Index(generated, b.open)
		if i < 0 {
			return false
		}
		start = i + 1
	}
	inside := generated[start:]
	if len(inside) < b.limit {
		return false
	}
	return !slices.Contains(inside, b.closeID)
}

// Process masks logits to the forced closing token when the budget is due.
func (b *Budget) Process(generated []int, logits []float32) {
	next := b.closeID
	if b.force != nil {
		next = b.ForcedHarmony(generated)
		if next < 0 {
			return
		}
	} else if !b.Due(generated) {
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

// Gate reports whether Process needs to run on the current step.
func (b *Budget) Gate(generated []int) bool { return b.Due(generated) }

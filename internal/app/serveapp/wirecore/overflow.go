package wirecore

import (
	"fmt"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/codec"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Overflow describes a prompt that reaches or exceeds its context window, distinguishing compactable conversations from fixed-scaffold overflows.
type Overflow struct {
	Prompt, Ctx, ModelWindow int
	Fixed                    int
	Reserved                 int
}

// Helpless reports whether compacting the conversation cannot make the request fit because the fixed scaffold plus reply reserve already fills Ctx.
func (e *Overflow) Helpless() bool {
	return e.Fixed >= 0 && e.Fixed+e.Reserved >= e.Ctx
}

// Message formats the dialect-appropriate overflow error ("anthropic" or OpenAI).
func (e *Overflow) Message(dialect string) string {
	if e.Helpless() {
		msg := fmt.Sprintf("this request cannot fit this server, and shortening the conversation will not help: its system prompt and tool definitions take %d tokens, a reply needs %d more, and the server holds %d positions",
			e.Fixed, e.Reserved, e.Ctx)
		if e.ModelWindow > e.Ctx {
			return msg + fmt.Sprintf("; restart goinfer-serve with -ctx %d or more (up to %d)", min(e.ModelWindow, e.Prompt+e.Reserved), e.ModelWindow)
		}
		return msg + "; that is the model's own size, so only a shorter system prompt or fewer tools can fit"
	}
	var msg string
	if dialect == "anthropic" {
		msg = fmt.Sprintf("prompt is too long: %d tokens > %d maximum", e.Prompt, e.Ctx)
	} else {
		msg = fmt.Sprintf("This model's maximum context length is %d tokens. However, your messages resulted in %d tokens (context_length_exceeded).", e.Ctx, e.Prompt)
	}
	if e.ModelWindow > e.Ctx {
		msg += fmt.Sprintf(" This is the server's GPU context (-ctx), not the model's limit (%d tokens): restart goinfer-serve with -ctx %d or more (up to %d) to accept it.",
			e.ModelWindow, e.Prompt+1, e.ModelWindow)
	}
	return msg
}

// OverflowReserve computes the reply token reserve used to judge whether conversation compaction can cure an overflow.
func OverflowReserve(maxTokens, ctx int) int {
	hi := max(ctx/4, 1)
	return min(max(maxTokens, min(1024, hi)), hi)
}

// FixedPromptTokens counts the tokens occupied by system, tools, and one empty user turn, returning -1 on error.
func FixedPromptTokens(tk *tokenizer.Tokenizer, tm *chat.Template, system string, tools []chat.Tool, encodeRaw func(string) ([]int, error)) int {
	empty := []chat.Turn{{Role: "user", Content: ""}}
	var ids []int
	var err error
	if tm != nil {
		ids, err = codec.EncodePrompt(tk, tm, system, empty, tools)
	} else if encodeRaw != nil {
		ids, err = encodeRaw(RawPrompt(system, empty))
	} else {
		return -1
	}
	if err != nil {
		return -1
	}
	return len(ids)
}

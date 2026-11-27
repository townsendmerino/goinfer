package chat

import "github.com/townsendmerino/goinfer/internal/chat/chatcore"

func toCoreTurn(t Turn) chatcore.Turn {
	var calls []chatcore.ToolCall
	if len(t.ToolCalls) > 0 {
		calls = make([]chatcore.ToolCall, len(t.ToolCalls))
		for i, c := range t.ToolCalls {
			calls[i] = chatcore.ToolCall{ID: c.ID, Name: c.Name, Arguments: c.Arguments}
		}
	}
	return chatcore.Turn{
		Role:       t.Role,
		Content:    t.Content,
		ToolCalls:  calls,
		ToolName:   t.ToolName,
		ToolCallID: t.ToolCallID,
		Reasoning:  t.Reasoning,
		ToolLoop:   t.ToolLoop,
	}
}

func toCoreTurns(turns []Turn) []chatcore.Turn {
	if len(turns) == 0 {
		return nil
	}
	out := make([]chatcore.Turn, len(turns))
	for i, t := range turns {
		out[i] = toCoreTurn(t)
	}
	return out
}

func toCoreTools(tools []Tool) []chatcore.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]chatcore.Tool, len(tools))
	for i, t := range tools {
		out[i] = chatcore.Tool{Name: t.Name, Description: t.Description, Parameters: t.Parameters}
	}
	return out
}

func fromCoreCall(c chatcore.ToolCall) ToolCall {
	return ToolCall{ID: c.ID, Name: c.Name, Arguments: c.Arguments}
}

func fromCoreCalls(calls []chatcore.ToolCall) []ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ToolCall, len(calls))
	for i, c := range calls {
		out[i] = fromCoreCall(c)
	}
	return out
}

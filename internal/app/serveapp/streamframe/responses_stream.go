package streamframe

import (
	"maps"
	"strings"
)

// RespStream drives the /v1/responses SSE item lifecycle (added, delta, done) with monotonically increasing
// sequence_number and output_index counters across reasoning, message, and function_call output items.
type RespStream struct {
	ss       *SSEWriter
	id       string
	seq      int
	idx      int
	msg      *respMsg
	rsn      *respMsg
	rsnDone  map[string]any
	mintBlob func(string) string
}

type respMsg struct {
	index int
	text  strings.Builder
}

// NewRespStream constructs a /v1/responses event stream writer for response id. mintBlob may be nil when carried-id signing is off.
func NewRespStream(ss *SSEWriter, id string, mintBlob func(string) string) *RespStream {
	return &RespStream{ss: ss, id: id, mintBlob: mintBlob}
}

// SetMintBlob sets the carried-ID blob minter invoked when the reasoning item closes.
func (r *RespStream) SetMintBlob(mintBlob func(string) string) { r.mintBlob = mintBlob }

// MessageOpen reports whether an assistant message output item has been opened and not yet closed.
func (r *RespStream) MessageOpen() bool { return r != nil && r.msg != nil }

// Emit sets p["type"] and p["sequence_number"] and writes one named SSE event.
func (r *RespStream) Emit(event string, p map[string]any) {
	p["type"] = event
	p["sequence_number"] = r.seq
	r.seq++
	SendEvent(r.ss, event, p)
}

func (r *RespStream) msgID() string { return r.id + "-msg" }
func (r *RespStream) rsnID() string { return r.id + "-rs" }

// ReasoningDelta streams a reasoning text fragment, lazily opening the reasoning output item on the first fragment.
func (r *RespStream) ReasoningDelta(t string) {
	if r.rsnDone != nil || r.msg != nil {
		return
	}
	if r.rsn == nil {
		r.rsn = &respMsg{index: r.idx}
		r.idx++
		r.Emit("response.output_item.added", map[string]any{"output_index": r.rsn.index, "item": map[string]any{
			"type": "reasoning", "id": r.rsnID(), "summary": []any{}}})
	}
	r.rsn.text.WriteString(t)
	r.Emit("response.reasoning_text.delta", map[string]any{"item_id": r.rsnID(), "output_index": r.rsn.index, "content_index": 0, "delta": t})
}

// CloseReasoning closes an open reasoning output item (if any) and returns its final item representation. Idempotent.
func (r *RespStream) CloseReasoning() map[string]any {
	if r.rsn != nil {
		text, idx := r.rsn.text.String(), r.rsn.index
		r.rsn = nil
		enc := ""
		if r.mintBlob != nil {
			enc = r.mintBlob(text)
		}
		r.rsnDone = ReasoningItem(r.rsnID(), text, enc)
		r.Emit("response.reasoning_text.done", map[string]any{"item_id": r.rsnID(), "output_index": idx, "content_index": 0, "text": text})
		r.Emit("response.output_item.done", map[string]any{"output_index": idx, "item": r.rsnDone})
	}
	return r.rsnDone
}

// ReasoningDoneItem returns the finalized reasoning item map once closed or emitted via CarrierOnly, or nil if none.
func (r *RespStream) ReasoningDoneItem() map[string]any { return r.rsnDone }

// CarrierOnly announces and closes an empty-text reasoning carrier item with encrypted_content enc at output_index 0.
// Returns false without writing anything if another item has already opened.
func (r *RespStream) CarrierOnly(enc string) bool {
	if r.idx != 0 || r.rsn != nil || r.rsnDone != nil || r.msg != nil {
		return false
	}
	idx := r.idx
	r.idx++
	r.rsnDone = ReasoningItem(r.rsnID(), "", enc)
	r.Emit("response.output_item.added", map[string]any{"output_index": idx, "item": map[string]any{
		"type": "reasoning", "id": r.rsnID(), "summary": []any{}}})
	r.Emit("response.output_item.done", map[string]any{"output_index": idx, "item": r.rsnDone})
	return true
}

func (r *RespStream) openMessage() {
	r.CloseReasoning()
	r.msg = &respMsg{index: r.idx}
	r.idx++
	r.Emit("response.output_item.added", map[string]any{"output_index": r.msg.index, "item": map[string]any{
		"type": "message", "id": r.msgID(), "role": "assistant", "status": "in_progress", "content": []any{}}})
	r.Emit("response.content_part.added", map[string]any{"item_id": r.msgID(), "output_index": r.msg.index, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}})
}

// Delta appends text to the assistant message item, opening the item first if needed.
func (r *RespStream) Delta(t string) {
	if r.msg == nil {
		r.openMessage()
	}
	r.msg.text.WriteString(t)
	r.Emit("response.output_text.delta", map[string]any{"item_id": r.msgID(), "output_index": r.msg.index, "content_index": 0, "delta": t})
}

// CloseMessage finishes the assistant message item (opening one around final if none was opened yet) and returns its completed map.
func (r *RespStream) CloseMessage(final string) map[string]any {
	if r.msg == nil {
		r.openMessage()
		if final != "" {
			r.Delta(final)
		}
	}
	text := r.msg.text.String()
	idx := r.msg.index
	r.msg = nil
	item := OutputMessage(r.msgID(), text)
	r.Emit("response.output_text.done", map[string]any{"item_id": r.msgID(), "output_index": idx, "content_index": 0, "text": text})
	r.Emit("response.content_part.done", map[string]any{"item_id": r.msgID(), "output_index": idx, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": text, "annotations": []any{}}})
	r.Emit("response.output_item.done", map[string]any{"output_index": idx, "item": item})
	return item
}

// FunctionCall announces, streams arguments for, and closes one function_call output item.
func (r *RespStream) FunctionCall(item map[string]any) {
	r.CloseReasoning()
	idx := r.idx
	r.idx++
	args, _ := item["arguments"].(string)
	open := make(map[string]any, len(item))
	maps.Copy(open, item)
	open["arguments"], open["status"] = "", "in_progress"
	r.Emit("response.output_item.added", map[string]any{"output_index": idx, "item": open})
	r.Emit("response.function_call_arguments.delta", map[string]any{"item_id": item["id"], "output_index": idx, "delta": args})
	r.Emit("response.function_call_arguments.done", map[string]any{"item_id": item["id"], "output_index": idx, "name": item["name"], "arguments": args})
	r.Emit("response.output_item.done", map[string]any{"output_index": idx, "item": item})
}

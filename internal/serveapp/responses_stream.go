package serveapp

import "strings"

// respStream writes a /v1/responses event stream with the item lifecycle OpenAI's clients read: every output item is
// announced (response.output_item.added), its parts and deltas follow, and it is closed (…done) before the next one
// opens, with sequence_number counting every event. Clients that assemble the reply from item events (Codex) see
// nothing of a stream that carries only response.output_text.delta.
//
// Items are numbered in the order the final response's output array lists them: the reasoning item, if the reply reasoned,
// then the message, if any, then each function call. One message item at most; it opens on its first delta and is closed
// before a function call.
type respStream struct {
	ss  *sseWriter
	id  string
	seq int
	idx int      // output_index of the next item
	msg *respMsg // the open message item, nil when none
	// rsn is the reasoning item (reasoning_items.go): opened by the first reasoning fragment, closed before any other item
	// opens; rsnDone is it as the final response lists it, once closed.
	rsn     *respMsg
	rsnDone map[string]any
}

type respMsg struct {
	index int
	text  strings.Builder
}

func newRespStream(ss *sseWriter, id string) *respStream { return &respStream{ss: ss, id: id} }

// emit writes one event; the payload gains its type and sequence_number.
func (r *respStream) emit(event string, p map[string]any) {
	p["type"] = event
	p["sequence_number"] = r.seq
	r.seq++
	sseEvent(r.ss, event, p)
}

func (r *respStream) msgID() string { return r.id + "-msg" }

func (r *respStream) rsnID() string { return r.id + "-rs" }

// reasoningDelta sends a piece of the reply's reasoning, opening the reasoning item first when it is not open. Reasoning
// after another item opened (a splitter never produces it) is kept out of the stream.
func (r *respStream) reasoningDelta(t string) {
	if r.rsnDone != nil || r.msg != nil {
		return
	}
	if r.rsn == nil {
		r.rsn = &respMsg{index: r.idx}
		r.idx++
		r.emit("response.output_item.added", map[string]any{"output_index": r.rsn.index, "item": map[string]any{
			"type": "reasoning", "id": r.rsnID(), "summary": []any{}}})
	}
	r.rsn.text.WriteString(t)
	r.emit("response.reasoning_text.delta", map[string]any{"item_id": r.rsnID(), "output_index": r.rsn.index, "content_index": 0, "delta": t})
}

// closeReasoning ends the reasoning item, if one is open, and returns it as the final response lists it (nil when the
// reply did not reason). Idempotent.
func (r *respStream) closeReasoning() map[string]any {
	if r.rsn != nil {
		text, idx := r.rsn.text.String(), r.rsn.index
		r.rsn = nil
		r.rsnDone = reasoningItem(r.rsnID(), text)
		r.emit("response.reasoning_text.done", map[string]any{"item_id": r.rsnID(), "output_index": idx, "content_index": 0, "text": text})
		r.emit("response.output_item.done", map[string]any{"output_index": idx, "item": r.rsnDone})
	}
	return r.rsnDone
}

func (r *respStream) openMessage() {
	r.closeReasoning()
	r.msg = &respMsg{index: r.idx}
	r.idx++
	r.emit("response.output_item.added", map[string]any{"output_index": r.msg.index, "item": map[string]any{
		"type": "message", "id": r.msgID(), "role": "assistant", "status": "in_progress", "content": []any{}}})
	r.emit("response.content_part.added", map[string]any{"item_id": r.msgID(), "output_index": r.msg.index, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}})
}

// delta sends a piece of the message's text, opening the message item first when it is not open.
func (r *respStream) delta(t string) {
	if r.msg == nil {
		r.openMessage()
	}
	r.msg.text.WriteString(t)
	r.emit("response.output_text.delta", map[string]any{"item_id": r.msgID(), "output_index": r.msg.index, "content_index": 0, "delta": t})
}

// closeMessage ends the message item and returns it as the final response lists it. When no delta opened one, the
// item is opened and closed here around final (one delta carries final, when it is not empty), so that every item in
// the final response was announced.
func (r *respStream) closeMessage(final string) map[string]any {
	if r.msg == nil {
		r.openMessage()
		if final != "" {
			r.delta(final)
		}
	}
	text := r.msg.text.String()
	idx := r.msg.index
	r.msg = nil
	item := outputMessage(r.msgID(), text)
	r.emit("response.output_text.done", map[string]any{"item_id": r.msgID(), "output_index": idx, "content_index": 0, "text": text})
	r.emit("response.content_part.done", map[string]any{"item_id": r.msgID(), "output_index": idx, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": text, "annotations": []any{}}})
	r.emit("response.output_item.done", map[string]any{"output_index": idx, "item": item})
	return item
}

// functionCall announces and closes one function_call item (a map as respondTools builds it). The arguments go out as
// one delta: the call is parsed from a finished buffer.
func (r *respStream) functionCall(item map[string]any) {
	r.closeReasoning()
	idx := r.idx
	r.idx++
	args, _ := item["arguments"].(string)
	open := make(map[string]any, len(item))
	for k, v := range item {
		open[k] = v
	}
	open["arguments"], open["status"] = "", "in_progress"
	r.emit("response.output_item.added", map[string]any{"output_index": idx, "item": open})
	r.emit("response.function_call_arguments.delta", map[string]any{"item_id": item["id"], "output_index": idx, "delta": args})
	r.emit("response.function_call_arguments.done", map[string]any{"item_id": item["id"], "output_index": idx, "name": item["name"], "arguments": args})
	r.emit("response.output_item.done", map[string]any{"output_index": idx, "item": item})
}

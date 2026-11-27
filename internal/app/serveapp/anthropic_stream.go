package serveapp

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

type anthropicThink = streamframe.AnthropicThink

var (
	anthropicEvent     = streamframe.SendEvent
	anthropicStreamErr = streamframe.SendAnthropicError
	streamTextBlock    = streamframe.StreamTextBlock
)

func anthropicSSEStart(w http.ResponseWriter) (*sseWriter, bool) {
	f, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "streaming unsupported")
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	return newSSEWriter(w, f), true
}

func (s *server) streamMessages(w http.ResponseWriter, r *http.Request, lm *loadedModel, gr genRequest, toolsActive bool, tools []chat.Tool, tm *chat.Template, turns []chat.Turn, ts thinkSettings, wantThinking bool, mint *carriedMint) {
	ss, ok := anthropicSSEStart(w)
	if !ok {
		return
	}
	streamCtx, cancelStream := ss.CancelOnFail(r.Context())
	defer cancelStream()
	id := "msg_" + reqID()
	gr.id = id
	anthropicEvent(ss, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": lm.name,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": 0},
		},
	})
	anthropicEvent(ss, "ping", map[string]any{"type": "ping"})

	var mintBlob func(string) string
	if mint != nil {
		mintBlob = mint.blob
	}
	th := streamframe.NewAnthropicThink(ss, wantThinking, mintBlob)
	s.routeThink(lm, &gr, tm, turns, ts, th.Push)

	if toolsActive {
		s.streamMessagesTools(streamCtx, w, ss, lm, gr, tools, th, mint)
		return
	}

	if wantThinking {
		streamMessagesThinking(ss, th, gr, func(onText func(string)) (turnResult, error) {
			return lm.drive(streamCtx, gr, s.gens, s.jobs, onText)
		})
		return
	}

	anthropicEvent(ss, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
	stopBeat := sseHeartbeat(ss)
	defer stopBeat()
	res, gerr := lm.drive(streamCtx, gr, s.gens, s.jobs, func(t string) {
		anthropicEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": t},
		})
	})
	stopBeat()
	if gerr != nil {
		anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		anthropicStreamErr(ss, "generation failed: "+gerr.Error())
		return
	}
	anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	reason, seq := anthropicStopReason(res.finish, res.stopHit)
	anthropicMessageEnd(ss, gr.clock, reason, seq, anthropicUsage(len(gr.promptIDs), res.reused.Tokens, res.nComp), res.cancelReason, emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp), false)
}

func (s *server) streamMessagesTools(ctx context.Context, w http.ResponseWriter, ss *sseWriter, lm *loadedModel, gr genRequest, tools []chat.Tool, th *anthropicThink, mint *carriedMint) {
	ts := streamframe.NewAnthropicTextStream(ss, th)
	t, gerr := s.runToolTurn(ctx, lm, gr, tools, ss, ts.Push)
	th.Close()
	if !ts.Open() {
		th.CarrierOnly(redactedCarrierBlock(mint))
	}
	if gerr != nil && !errors.Is(gerr, errProseDiverged) {
		anthropicStreamErr(ss, "generation failed: "+gerr.Error())
		return
	}
	if gerr != nil {
		anthropicStreamErr(ss, gerr.Error())
		return
	}
	if t.cancelReason != "" {
		ts.Close()
		reason, seq := anthropicStopReason(t.finish, t.stopHit)
		anthropicMessageEnd(ss, gr.clock, reason, seq, anthropicUsage(len(gr.promptIDs), t.reused.Tokens, t.nComp), t.cancelReason, nil, false)
		return
	}

	if len(t.calls) == 0 {
		ts.Finish(t.rest, true)
		reason, seq := anthropicStopReason(t.finish, t.stopHit)
		anthropicMessageEnd(ss, gr.clock, reason, seq, anthropicUsage(len(gr.promptIDs), t.reused.Tokens, t.nComp), "", emptyAnswerOf(gr, t.finish, t.stopHit, 0, t.nComp), false)
		return
	}

	idx := ts.Finish(t.rest, false)
	for _, c := range t.calls {
		tu := toolUseBlock(c)
		anthropicEvent(ss, "content_block_start", map[string]any{
			"type": "content_block_start", "index": idx,
			"content_block": map[string]any{"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": map[string]any{}},
		})
		args := strings.TrimSpace(string(c.Arguments))
		if args == "" {
			args = "{}"
		}
		anthropicEvent(ss, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": idx,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
		})
		anthropicEvent(ss, "content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
		idx++
	}
	anthropicMessageEnd(ss, gr.clock, "tool_use", nil, anthropicUsage(len(gr.promptIDs), t.reused.Tokens, t.nComp), "", nil, t.fromReasoning)
}

func streamMessagesThinking(ss *sseWriter, th *anthropicThink, gr genRequest, run func(onText func(string)) (turnResult, error)) {
	text := streamframe.NewAnthropicTextStream(ss, th)
	stopBeat := sseHeartbeat(ss)
	defer stopBeat()
	res, gerr := run(text.PushPlain)
	stopBeat()
	th.Close()
	if gerr != nil {
		text.Close()
		anthropicStreamErr(ss, "generation failed: "+gerr.Error())
		return
	}
	text.Close()
	if !th.Used() && !text.EverOpen() {
		streamTextBlock(ss, 0, "")
	}
	reason, seq := anthropicStopReason(res.finish, res.stopHit)
	anthropicMessageEnd(ss, gr.clock, reason, seq, anthropicUsage(len(gr.promptIDs), res.reused.Tokens, res.nComp), res.cancelReason, emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp), false)
}

func anthropicMessageEnd(ss *sseWriter, clock *jobs.TurnClock, reason string, seq any, usage map[string]any, cancelReason string, empty *emptyAnswer, fromReasoning bool) {
	md := clock.AddTurn(map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": reason, "stop_sequence": seq},
		"usage": usage,
	})
	if empty != nil {
		md["goinfer_empty_answer"] = empty
	}
	if fromReasoning {
		md[fromReasoningField] = true
	}
	anthropicEvent(ss, "message_delta", md)
	if cancelReason != "" {
		anthropicEvent(ss, "goinfer_cancelled", map[string]any{"type": "goinfer_cancelled", "reason": cancelReason})
	}
	anthropicEvent(ss, "message_stop", map[string]any{"type": "message_stop"})
}

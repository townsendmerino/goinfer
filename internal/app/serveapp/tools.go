package serveapp

import (
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
	"github.com/townsendmerino/goinfer/internal/codec"
)

var (
	toolUnionEnabled = os.Getenv("GOINFER_TOOL_UNION") != "0"
	toolUnionAuto    = toolUnionEnabled

	toAPICalls         = wirecore.ToAPICalls
	streamToolCalls    = wirecore.StreamToolCalls
	toolChoiceMode     = wirecore.ToolChoiceMode
	openAIUnionMode    = wirecore.OpenAIUnionMode
	forcedTool         = wirecore.ForcedTool
	endsWithToolResult = wirecore.EndsWithToolResult
)

func (s *server) handleChatTools(w http.ResponseWriter, r *http.Request, req chatReq) {
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveChatToolsWith(w, r, req, lm) })
}

func (s *server) serveChatToolsWith(w http.ResponseWriter, r *http.Request, req chatReq, lm *loadedModel) {
	if lm.tmpl == nil || !lm.tmpl.SupportsTools() {
		writeErr(w, http.StatusBadRequest, "this model has no tool-calling template")
		return
	}
	if err := checkToolCount(len(req.Tools)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages) + toolSchemaBytes(req.Tools)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	tools := make([]chat.Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	system, turns := messagesToTurns(req.Messages)
	ts, terr := s.resolveThink(req.Think())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if toolsConstrainedFromStart(forcedTool(req.ToolChoice, tools, endsWithToolResult(turns)), toolChoiceMode(req.ToolChoice) == "function", openAIUnionMode(req.ToolChoice), tools) {
		tm = lm.constrainedTemplate(ts)
	}
	ids, err := codec.EncodePrompt(lm.tk, tm, system, turns, tools)
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepareTurn(req.Sampling, tm, system, turns, tools, ids)
	if err != nil {
		writePrepareErr(w, err)
		return
	}
	forced := forcedTool(req.ToolChoice, tools, endsWithToolResult(turns))
	namedForce := toolChoiceMode(req.ToolChoice) == "function"
	if cerr := constrainForcedTool(lm, &gr, forced, namedForce, openAIUnionMode(req.ToolChoice), tools); cerr != nil {
		writeErr(w, http.StatusBadRequest, cerr.Error())
		return
	}
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	id := "chatcmpl-" + reqID()
	gr.id = id
	created := time.Now().Unix()

	var ss *sseWriter
	if req.Stream {
		var ok bool
		if ss, ok = sseStart(w); !ok {
			return
		}
	}
	var onProse func(string)
	var rb strings.Builder
	if ss != nil {
		onProse = func(out string) { sseSend(ss, chatChunk(id, created, lm.name, delta{Content: out}, nil)) }
		s.routeThink(lm, &gr, tm, turns, ts, func(t string) { sseSend(ss, chatChunk(id, created, lm.name, delta{ReasoningContent: t}, nil)) })
	} else {
		s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	}
	t, gerr := s.runToolTurn(r.Context(), lm, gr, tools, ss, onProse)
	if errors.Is(gerr, errProseDiverged) {
		sseErr(ss, gerr.Error())
		sseDone(ss)
		return
	}
	if gerr != nil {
		if ss != nil {
			if errors.Is(gerr, errNoForcedCall) {
				sendUsage(ss, gr.clock, req.StreamOptions, id, created, lm.name,
					newUsage(len(gr.promptIDs), t.nComp, t.reused))
			}
			sseErr(ss, "generation failed: "+gerr.Error())
			sseDone(ss)
			return
		}
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	finish, nComp, reused := t.finish, t.nComp, t.reused
	if t.cancelReason != "" {
		if ss != nil {
			sseSend(ss, map[string]any{"goinfer_cancelled": map[string]any{"id": id, "reason": t.cancelReason}})
			sendUsage(ss, gr.clock, req.StreamOptions, id, created, lm.name,
				newUsage(len(gr.promptIDs), nComp, reused))
			sseDone(ss)
			return
		}
		writeErr(w, statusCancelled, "generation cancelled: "+t.cancelReason)
		return
	}

	msg := map[string]any{"role": "assistant"}
	if len(t.calls) > 0 {
		msg["content"] = nil
		if t.lead != "" {
			msg["content"] = t.lead
		}
		msg["tool_calls"] = toAPICalls(t.calls)
		finish = "tool_calls"
	} else {
		msg["content"] = t.raw
	}
	if rb.Len() > 0 {
		msg["reasoning_content"] = rb.String()
	}
	choice := map[string]any{"index": 0, "message": msg, "finish_reason": finish}
	empty := emptyAnswerOf(gr, finish, t.stopHit, len(t.calls), nComp)
	if empty != nil {
		choice["goinfer_empty_answer"] = empty
	}
	if t.fromReasoning && len(t.calls) > 0 {
		choice[fromReasoningField] = true
	}
	usagev := newUsage(len(gr.promptIDs), nComp, reused)

	if req.Stream {
		d := map[string]any{"role": "assistant"}
		if len(t.calls) > 0 {
			d["tool_calls"] = streamToolCalls(t.calls)
			if t.rest != "" {
				sseSend(ss, chatChunk(id, created, lm.name, delta{Content: t.rest}, nil))
			}
		} else {
			d["content"] = t.rest
		}
		sseSend(ss, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": lm.name,
			"choices": []any{map[string]any{"index": 0, "delta": d, "finish_reason": nil}}})
		sseSend(ss, finishChunk(id, created, lm.name, finish, empty, t.fromReasoning && len(t.calls) > 0))
		sendUsage(ss, gr.clock, req.StreamOptions, id, created, lm.name, usagev)
		sseDone(ss)
		return
	}
	writeJSON(w, http.StatusOK, gr.clock.AddTurn(map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": lm.name,
		"choices": []any{choice}, "usage": usagev,
	}))
}

func toolsConstrainedFromStart(forced *chat.Tool, namedForce bool, union string, tools []chat.Tool) bool {
	return wirecore.ToolsConstrainedFromStart(forced, namedForce, union, tools, toolUnionEnabled)
}

func constrainForcedTool(lm *loadedModel, gr *genRequest, forced *chat.Tool, namedForce bool, union string, tools []chat.Tool) error {
	m, lazy, post, err := wirecore.BuildToolMasker(lm.tmpl, forced, namedForce, union, tools, toolUnionEnabled, toolUnionAuto, lm.cachedTokenBytes, slices.Concat(lm.eosIDs, lm.stopIDs))
	if err != nil {
		return err
	}
	if m != nil {
		gr.sp.LogitProcessor = m.Process
		gr.masker = m
	} else if lazy != nil {
		armLazy(lm, gr, lazy)
	}
	if post {
		gr.callPostCheck = true
	}
	return nil
}

func constrainToolUnion(lm *loadedModel, gr *genRequest, mode string, tools []chat.Tool) {
	_ = constrainForcedTool(lm, gr, nil, false, mode, tools)
}

func armLazy(lm *loadedModel, gr *genRequest, lazy *constrain.LazyMasker) {
	if lm.spec || lm.blockSpec != nil {
		gr.lazy = lazy
		return
	}
	gr.sp.LogitProcessor = lazy.Process
	gr.sp.LogitProcessorGate = lazy.Gate
}

func forcedCallEnforcement(tm *chat.Template) string {
	return wirecore.ForcedCallEnforcement(tm, toolUnionEnabled)
}

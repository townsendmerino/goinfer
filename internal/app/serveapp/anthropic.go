package serveapp

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/anthropicapi"
)

type (
	anthropicReq     = anthropicapi.Req
	anthropicMessage = anthropicapi.Message
	anthropicTool    = anthropicapi.Tool
)

var (
	writeAnthropicErr     = anthropicapi.WriteErr
	decodeAnthropicJSON   = anthropicapi.DecodeJSON
	anthropicText         = anthropicapi.Text
	anthropicInputBytes   = anthropicapi.InputBytes
	anthropicTurns        = anthropicapi.Turns
	anthropicTurnsCarried = anthropicapi.TurnsCarried
	anthropicImages       = anthropicapi.Images
	mergeAdjacent         = anthropicapi.MergeAdjacent
	anthropicTools        = anthropicapi.Tools
	anthropicToolMode     = anthropicapi.ToolMode
	anthropicForcedTool   = anthropicapi.ForcedTool
	thinkingBlock         = anthropicapi.ThinkingBlock
	textBlock             = anthropicapi.TextBlock
	anthropicStopReason   = anthropicapi.StopReason
	anthropicUsage        = anthropicapi.Usage
)

func anthropicToolsConstrainedFromStart(mode, name string, tools []chat.Tool) bool {
	return anthropicapi.ToolsConstrainedFromStart(mode, name, tools, toolUnionEnabled)
}

func redactedCarrierBlock(mint *carriedMint) map[string]any {
	var fn func(string) string
	if mint != nil {
		fn = mint.blob
	}
	return anthropicapi.RedactedCarrierBlock(fn)
}

func (s *server) anthropicModelNotFound(w http.ResponseWriter, name string) {
	writeAnthropicErr(w, http.StatusNotFound, "not_found_error",
		fmt.Sprintf("model %q not found (served: %s)", name, strings.Join(s.servedNames(), ", ")))
}

func applyToolConstraint(lm *loadedModel, gr *genRequest, mode, name string, tools []chat.Tool) error {
	return constrainForcedTool(lm, gr, anthropicForcedTool(mode, name, tools), mode == "tool", anthropicapi.UnionMode(mode), tools)
}

func (s *server) handleMessages(w http.ResponseWriter, r *http.Request) {
	var req anthropicReq
	if !decodeAnthropicJSON(w, r, &req) {
		return
	}
	if req.MaxTokens == nil || *req.MaxTokens <= 0 {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "max_tokens is required and must be > 0")
		return
	}
	if len(req.Messages) == 0 {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "messages must not be empty")
		return
	}
	if s.fastRoute(r, &req.Model) {
		req.ApplyFast()
	}
	s.withModelAnthropic(w, req.Model, func(lm *loadedModel) { s.serveMessagesWith(w, r, req, lm) })
}

func (s *server) serveMessagesWith(w http.ResponseWriter, r *http.Request, req anthropicReq, lm *loadedModel) {
	if err := lm.promptTooLargeForContext(anthropicInputBytes(&req)); err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	noteOmittedImages(w.Header(), omitAnthropicHistoryImages(req.Messages))
	imgs, ierr := anthropicImages(&req)
	if ierr != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", ierr.Error())
		return
	}
	if len(imgs) > 0 {
		if vmode, _ := anthropicToolMode(req.ToolChoice); len(req.Tools) > 0 && vmode != "none" {
			writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "tools are not supported together with image inputs; send images or tools, not both")
			return
		}
		s.serveVisionMessages(w, r, &req, lm, imgs)
		return
	}
	system, turns, carriers, aerr := anthropicTurnsCarried(&req)
	if aerr != nil {
		aerr.Write(w)
		return
	}

	mode, name := anthropicToolMode(req.ToolChoice)
	toolsActive := len(req.Tools) > 0 && mode != "none"
	if toolsActive {
		if err := checkToolCount(len(req.Tools)); err != nil {
			writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
	}
	var tools []chat.Tool
	var gr genRequest
	sm, serr := req.ToSampling()
	if serr != nil {
		serr.Write(w)
		return
	}
	ts, terr := s.resolveThink(req.ThinkRequest())
	if terr != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if sm.ConstrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	if toolsActive {
		if lm.tmpl == nil || !lm.tmpl.SupportsTools() {
			writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", "this model has no tool-calling template")
			return
		}
		tools = anthropicTools(req.Tools)
		if anthropicToolsConstrainedFromStart(mode, name, tools) {
			tm = lm.constrainedTemplate(ts)
		}
	}
	ids, idsOut, err := lm.encodePrompt(tm, system, turns, tools, s.idsCheckFor(lm, r, carriers))
	if err != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "encode: "+err.Error())
		return
	}
	s.noteIDs(r, idsOut)
	if gr, err = lm.prepareTurn(sm, tm, system, turns, tools, ids); err != nil {
		writeAnthropicPrepareErr(w, err)
		return
	}
	if toolsActive {
		if cerr := applyToolConstraint(lm, &gr, mode, name, tools); cerr != nil {
			writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", cerr.Error())
			return
		}
	}

	// A full queue returns 529 overloaded_error on the Anthropic surface.
	if ok, haltReason := lm.tryEnter(r.Context(), admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState); !ok {
		if haltReason != "" {
			writeAnthropicErr(w, http.StatusServiceUnavailable, "overloaded_error", "halted: "+haltReason)
			return
		}
		if r.Context().Err() != nil {
			return
		}
		w.Header().Set("Retry-After", "1")
		writeAnthropicErr(w, 529, "overloaded_error", fmt.Sprintf("model %q is busy; retry", lm.name))
		return
	}
	defer lm.exit()

	mint := s.mintFor(lm, r, gr)
	if req.Stream {
		s.streamMessages(w, r, lm, gr, toolsActive, tools, tm, turns, ts, req.WantsThinking(), mint)
		return
	}

	id := "msg_" + reqID()
	gr.id = id
	var rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	var (
		text          string
		calls         []chat.ToolCall
		lead          string
		res           turnResult
		gerr          error
		fromReasoning bool
	)
	if toolsActive {
		var t toolTurn
		t, gerr = s.runToolTurn(r.Context(), lm, gr, tools, nil, nil)
		text, calls, lead, res = t.raw, t.calls, t.lead, t.turnResult
		fromReasoning = t.fromReasoning && len(t.calls) > 0
	} else {
		var sb strings.Builder
		res, gerr = lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
		text = sb.String()
	}
	if gerr != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeAnthropicErr(w, statusCancelled, "cancelled", "generation cancelled: "+res.cancelReason)
		return
	}

	var mintBlob func(string) string
	if mint != nil {
		mintBlob = mint.blob
	}
	content, reason, seq := anthropicapi.BuildMessageContent(req.WantsThinking(), rb.String(), text, lead, calls, res.finish, res.stopHit, mintBlob)

	msg := map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         lm.name,
		"content":       content,
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         anthropicUsage(len(gr.promptIDs), res.reused.Tokens, res.nComp),
	}
	if e := emptyAnswerOf(gr, res.finish, res.stopHit, len(calls), res.nComp); e != nil {
		msg["goinfer_empty_answer"] = e
	}
	if fromReasoning {
		msg[fromReasoningField] = true
	}
	writeJSON(w, http.StatusOK, gr.clock.AddTurn(msg))
}

func (s *server) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	var req anthropicReq
	if !decodeAnthropicJSON(w, r, &req) {
		return
	}
	if s.fastRoute(r, &req.Model) {
		req.ApplyFast()
	}
	s.withModelAnthropic(w, req.Model, func(lm *loadedModel) { s.serveCountTokensWith(w, req, lm) })
}

func (s *server) serveCountTokensWith(w http.ResponseWriter, req anthropicReq, lm *loadedModel) {
	if err := lm.promptTooLargeForContext(anthropicInputBytes(&req)); err != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	system, turns, carriers, aerr := anthropicTurnsCarried(&req)
	if aerr != nil {
		aerr.Write(w)
		return
	}
	var tools []chat.Tool
	ts, terr := s.resolveThink(req.ThinkRequest())
	if terr != nil {
		writeAnthropicErr(w, http.StatusBadRequest, "invalid_request_error", terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	rf, ferr := req.OutputFormat()
	if ferr != nil {
		ferr.Write(w)
		return
	}
	if rf != nil {
		tm = lm.constrainedTemplate(ts)
	}
	if mode, name := anthropicToolMode(req.ToolChoice); len(req.Tools) > 0 && mode != "none" && lm.tmpl != nil && lm.tmpl.SupportsTools() {
		tools = anthropicTools(req.Tools)
		if anthropicToolsConstrainedFromStart(mode, name, tools) {
			tm = lm.constrainedTemplate(ts)
		}
	}
	ids, _, err := lm.encodePrompt(tm, system, turns, tools, s.idsCheckFor(lm, nil, carriers))
	if err != nil {
		writeAnthropicErr(w, http.StatusInternalServerError, "api_error", "encode: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"input_tokens": len(ids)})
}

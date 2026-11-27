package serveapp

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/responsesapi"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/streamframe"
)

type responseReq = responsesapi.Req

var (
	newResponseStore        = responsesapi.NewStore
	responseMessagesBytes   = responsesapi.MessagesBytes
	responseInputToMessages = responsesapi.InputToMessages
	toolOutputText          = responsesapi.ToolOutputText
	contentText             = responsesapi.ContentText
	responseObject          = responsesapi.Object
	responseObjectTurn      = responsesapi.ObjectTurn
	withCachedTokens        = responsesapi.WithCachedTokens
	respStatus              = responsesapi.Status
	sseEvent                = streamframe.SendEvent
)

func (s *server) handleResponses(w http.ResponseWriter, r *http.Request) {
	var req responseReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if s.fastRoute(r, &req.Model) {
		req.ApplyFast()
	}
	req.Tools = dropUnrunnableTools(w, req.Tools)
	s.withModel(w, req.Model, func(lm *loadedModel) { s.serveResponsesWith(w, r, req, lm) })
}

// serveResponsesWith runs a /v1/responses generation. Reached ONLY through withModel (liveness RLock held).
func (s *server) serveResponsesWith(w http.ResponseWriter, r *http.Request, req responseReq, lm *loadedModel) {
	var messages []chatMessage
	if req.PreviousResponseID != "" {
		prior := s.responses.Get(req.PreviousResponseID)
		if prior == nil {
			writeErr(w, http.StatusNotFound, fmt.Sprintf("previous_response_id %q not found", req.PreviousResponseID))
			return
		}
		messages = append(messages, prior.Messages...)
	} else if req.Instructions != "" {
		messages = append(messages, chatMessage{Role: "system", Content: rawStr(req.Instructions)})
	}
	inputMsgs, err := responseInputToMessages(req.Input)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	messages = append(messages, inputMsgs...)

	if err := lm.promptTooLargeForContext(chatInputBytes(messages) + toolSchemaBytes(req.Tools)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	noteOmittedImages(w.Header(), omitChatHistoryImages(messages))
	imgs, ierr := chatImages(messages)
	if ierr != nil {
		writeErr(w, http.StatusBadRequest, ierr.Error())
		return
	}

	sm := req.SamplingSettings()
	id := "resp_" + reqID()
	created := time.Now().Unix()
	store := req.ShouldStore()

	toolsActive := len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none"
	if len(imgs) > 0 {
		if toolsActive {
			writeErr(w, http.StatusBadRequest, "tools are not supported together with image inputs; send images or tools, not both")
			return
		}
		s.serveVisionResponses(w, r, lm, req, messages, imgs, sm, id, created, store)
		return
	}
	if toolsActive {
		if lm.tmpl == nil || !lm.tmpl.SupportsTools() {
			writeErr(w, http.StatusBadRequest, "this model has no tool-calling template")
			return
		}
		s.respondTools(w, r, lm, req, messages, sm, id, created, store)
		return
	}

	ts, terr := s.resolveThink(req.ThinkRequest())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if sm.ConstrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	system, turns, carriers := messagesToTurnsCarried(messages)
	ids, idsOut, err := lm.encodePrompt(tm, system, turns, nil, s.idsCheckFor(lm, r, carriers))
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	s.noteIDs(r, idsOut)
	gr, err := lm.prepareTurn(sm, tm, system, turns, nil, ids)
	if err != nil {
		writeResponsesPrepareErr(w, err, req.Stream, id, lm.name, created)
		return
	}
	gr.id = id
	s.routeThink(lm, &gr, tm, turns, ts, nil)
	mint := s.mintFor(lm, r, gr)
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)

	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		rs := newRespStream(ss, id)
		if mint != nil {
			rs.SetMintBlob(mint.blob)
		}
		rs.Emit("response.created", map[string]any{"response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		if gr.think != nil {
			gr.think.SetOnReasoning(rs.ReasoningDelta)
		}
		var sb strings.Builder
		stopBeat := sseHeartbeat(ss)
		defer stopBeat()
		res, gerr := lm.drive(streamCtx, gr, s.gens, s.jobs, func(t string) {
			sb.WriteString(t)
			rs.Delta(t)
		})
		stopBeat()
		if gerr != nil {
			sseEvent(ss, "error", map[string]any{"type": "error", "message": "generation failed: " + gerr.Error()})
			sseDone(ss)
			return
		}
		out := withCarrier(rs, mint, req.Include, id, withReasoning(rs, gr.think, id, mint, []any{rs.CloseMessage(sb.String())}))
		rs.Emit("response.completed", map[string]any{"response": withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens)})
		if res.cancelReason != "" {
			rs.Emit("response.cancelled", map[string]any{"reason": res.cancelReason})
		}
		sseDone(ss)
		s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
		return
	}

	var sb strings.Builder
	res, gerr := lm.drive(r.Context(), gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+res.cancelReason)
		return
	}
	out := withCarrier(nil, mint, req.Include, id, withReasoning(nil, gr.think, id, mint, []any{outputMessage(id+"-msg", sb.String())}))
	writeJSON(w, http.StatusOK, withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens))
	s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
}

// serveVisionResponses runs a /v1/responses request carrying image or audio inputs.
func (s *server) serveVisionResponses(w http.ResponseWriter, r *http.Request, lm *loadedModel, req responseReq, messages []chatMessage, imgs []imageRef, sm sampling, id string, created int64, store bool) {
	var hasAudio, hasImage bool
	for _, im := range imgs {
		if im.Audio {
			hasAudio = true
		} else {
			hasImage = true
		}
	}
	if hasAudio && !lm.audioCapable() {
		writeErr(w, http.StatusBadRequest, "this model has no audio tower (audio input needs a Gemma 4 checkpoint with an audio_config, e.g. E2B or E4B)")
		return
	}
	if hasImage && !lm.visionCapable() {
		writeErr(w, http.StatusBadRequest, "this model has no vision tower (start with --vision <dir> to enable image input)")
		return
	}
	if len(imgs) > maxImagesPerTurn {
		writeErr(w, http.StatusBadRequest, tooManyImages(len(imgs)))
		return
	}
	ts, terr := s.resolveThink(req.ThinkRequest())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if sm.ConstrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	system, turns := messagesToTurns(messages)
	msgText, ordered := chatMediaText(messages)
	vi, err := lm.visionPromptN(tm, system, turns, imgs, ordered, msgText)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	gr, err := lm.prepare(sm, vi.IDs, false)
	if err != nil {
		writeResponsesPrepareErr(w, err, req.Stream, id, lm.name, created)
		return
	}
	gr.id = id
	s.routeThink(lm, &gr, tm, turns, ts, nil)
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)
	if req.Stream {
		ss, ok := sseStart(w)
		if !ok {
			return
		}
		streamCtx, cancelStream := ss.CancelOnFail(r.Context())
		defer cancelStream()
		rs := newRespStream(ss, id)
		rs.Emit("response.created", map[string]any{"response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		if gr.think != nil {
			gr.think.SetOnReasoning(rs.ReasoningDelta)
		}
		var sb strings.Builder
		stopBeat := sseHeartbeat(ss)
		defer stopBeat()
		res, gerr := lm.driveVL(streamCtx, gr, vi, s.gens, s.jobs, func(t string) {
			sb.WriteString(t)
			rs.Delta(t)
		})
		stopBeat()
		if gerr != nil {
			sseEvent(ss, "error", map[string]any{"type": "error", "message": "generation failed: " + gerr.Error()})
			sseDone(ss)
			return
		}
		out := withReasoning(rs, gr.think, id, nil, []any{rs.CloseMessage(sb.String())})
		rs.Emit("response.completed", map[string]any{"response": withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens)})
		if res.cancelReason != "" {
			rs.Emit("response.cancelled", map[string]any{"reason": res.cancelReason})
		}
		sseDone(ss)
		s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
		return
	}
	var sb strings.Builder
	res, gerr := lm.driveVL(r.Context(), gr, vi, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	if res.cancelReason != "" {
		writeErr(w, statusCancelled, "generation cancelled: "+res.cancelReason)
		return
	}
	out := withReasoning(nil, gr.think, id, nil, []any{outputMessage(id+"-msg", sb.String())})
	writeJSON(w, http.StatusOK, withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(res.finish), out, inTok, res.nComp), emptyAnswerOf(gr, res.finish, res.stopHit, 0, res.nComp)), res.reused.Tokens))
	s.maybeStore(store, id, lm.name, messages, sb.String(), gr.think.ReasoningText(), nil)
}

// respondTools generates a buffered tool-calling response and emits function_call output items.
func (s *server) respondTools(w http.ResponseWriter, r *http.Request, lm *loadedModel, req responseReq, messages []chatMessage, sm sampling, id string, created int64, store bool) {
	if err := checkToolCount(len(req.Tools)); err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	tools := make([]chat.Tool, len(req.Tools))
	for i, t := range req.Tools {
		tools[i] = chat.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters}
	}
	system, turns, carriers := messagesToTurnsCarried(messages)
	ts, terr := s.resolveThink(req.ThinkRequest())
	if terr != nil {
		writeErr(w, http.StatusBadRequest, terr.Error())
		return
	}
	tm := lm.templateFor(ts)
	if toolsConstrainedFromStart(forcedTool(req.ToolChoice, tools, endsWithToolResult(turns)), toolChoiceMode(req.ToolChoice) == "function", openAIUnionMode(req.ToolChoice), tools) {
		tm = lm.constrainedTemplate(ts)
	}
	ids, idsOut, err := lm.encodePrompt(tm, system, turns, tools, s.idsCheckFor(lm, r, carriers))
	if err != nil {
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	s.noteIDs(r, idsOut)
	gr, err := lm.prepareTurn(sm, tm, system, turns, tools, ids)
	if err != nil {
		writeResponsesPrepareErr(w, err, req.Stream, id, lm.name, created)
		return
	}
	gr.id = id
	forced := forcedTool(req.ToolChoice, tools, endsWithToolResult(turns))
	namedForce := toolChoiceMode(req.ToolChoice) == "function"
	if cerr := constrainForcedTool(lm, &gr, forced, namedForce, openAIUnionMode(req.ToolChoice), tools); cerr != nil {
		writeErr(w, http.StatusBadRequest, cerr.Error())
		return
	}
	s.routeThink(lm, &gr, tm, turns, ts, nil)
	mint := s.mintFor(lm, r, gr)
	if !lm.enter(w, r, admissionRecord{promptIDs: gr.promptIDs, clock: gr.clock}, s.haltState) {
		return
	}
	defer lm.exit()
	inTok := len(gr.promptIDs)
	var ss *sseWriter
	if req.Stream {
		var ok bool
		if ss, ok = sseStart(w); !ok {
			return
		}
	}
	var onProse func(string)
	var rs *respStream
	if ss != nil {
		rs = newRespStream(ss, id)
		if mint != nil {
			rs.SetMintBlob(mint.blob)
		}
		rs.Emit("response.created", map[string]any{"response": responseObject(id, lm.name, created, "in_progress", []any{}, inTok, 0)})
		onProse = rs.Delta
		if gr.think != nil {
			gr.think.SetOnReasoning(rs.ReasoningDelta)
		}
	}
	t, gerr := s.runToolTurn(r.Context(), lm, gr, tools, ss, onProse)
	if gerr != nil {
		if ss != nil {
			msg := gerr.Error()
			if !errors.Is(gerr, errProseDiverged) {
				msg = "generation failed: " + msg
			}
			sseErr(ss, msg)
			sseDone(ss)
			return
		}
		writeServerErr(w, "generation failed: "+gerr.Error())
		return
	}
	finish, nComp := t.finish, t.nComp
	if t.cancelReason != "" {
		if ss != nil {
			resp := responseObjectTurn(gr.clock, id, lm.name, created, respStatus(finish), withReasoning(rs, gr.think, id, mint, []any{rs.CloseMessage(t.raw)}), inTok, nComp)
			rs.Emit("response.completed", map[string]any{"response": resp})
			rs.Emit("response.cancelled", map[string]any{"reason": t.cancelReason})
			sseDone(ss)
			return
		}
		writeErr(w, statusCancelled, "generation cancelled: "+t.cancelReason)
		return
	}
	calls, lead := t.calls, ""
	if len(calls) > 0 {
		lead = t.lead
	}
	if rs != nil && wantsCarrier(req.Include) {
		if enc := mint.blob(""); enc != "" {
			rs.CarrierOnly(enc)
		}
	}
	if ss != nil && t.rest != "" {
		onProse(t.rest)
	}

	out, toolCalls := responsesapi.BuildToolOutputs(id, calls, t.raw, lead, rs, reqID)
	out = withReasoning(rs, gr.think, id, mint, out)
	if rs == nil {
		out = withCarrier(nil, mint, req.Include, id, out)
	}
	resp := withCachedTokens(markEmptyResponse(responseObjectTurn(gr.clock, id, lm.name, created, respStatus(finish), out, inTok, nComp), emptyAnswerOf(gr, finish, t.stopHit, len(calls), nComp)), t.reused.Tokens)
	if t.fromReasoning && len(calls) > 0 {
		resp[fromReasoningField] = true
	}
	if req.Stream {
		rs.Emit("response.completed", map[string]any{"response": resp})
		sseDone(ss)
	} else {
		writeJSON(w, http.StatusOK, resp)
	}
	s.maybeStore(store, id, lm.name, messages, lead, gr.think.ReasoningText(), toolCalls)
}

// maybeStore records this turn's assistant output for a later previous_response_id continuation.
func (s *server) maybeStore(store bool, id, model string, messages []chatMessage, assistant, reasoning string, toolCalls []apiToolCall) {
	if !store || s.responses == nil {
		return
	}
	s.responses.Put(id, model, responsesapi.AppendTurn(messages, assistant, reasoning, toolCalls))
}

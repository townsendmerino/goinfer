package serveapp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/anthropicapi"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

const batchLineWorkers = 8

// runBatchPool dispatches indices 0..n-1 across at most batchLineWorkers goroutines so a max-size batch
// does not spawn thousands of concurrent pre-queue goroutines at once.
func runBatchPool(n int, fn func(i int)) {
	if n <= 0 {
		return
	}
	workers := min(n, batchLineWorkers)
	var next atomic.Int64
	for range workers {
		go runBatchWorker(n, &next, fn)
	}
}

func runBatchWorker(n int, next *atomic.Int64, fn func(i int)) {
	for {
		i := int(next.Add(1) - 1)
		if i >= n {
			return
		}
		fn(i)
	}
}

// runBatchChatLine is one line of an OpenAI batch: the same validate -> resolveAndLock -> encode -> prepare -> admission
// -> drive pipeline that handleCreateJob (jobs_http.go) and runJob (jobs_run.go) run, replicated rather than called
// because both are tied to a single HTTP handler's or a fire-and-forget job's lifecycle, neither of which fits "run
// synchronously in this line's own goroutine and hand back a result." Every EXPENSIVE step (admission, lm.drive) is the
// identical shared call, not a second implementation.
//
// A validation failure is THIS LINE'S error, never the whole batch's, as in the real batch API.
func runBatchChatLine(s *server, batch *stores.BatchRecord, i int, req chatReq) {
	customID := batch.CustomIDs[i]
	defer batch.DoneLine()
	defer batch.LineFinished()
	defer recoverBatchLine(batch, i) // after DoneLine: runs first, so the line's result is set before the batch can finalize

	if batch.IsCancelling() {
		batch.SetResult(i, stores.LineError(customID, statusCancelled, "cancelled", "batch cancelled"))
		return
	}
	if len(req.Messages) == 0 {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error",
			"messages is required and must contain at least one message"))
		return
	}
	if wirecore.HasChatMedia(req.Messages) {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support image or audio inputs this pass; use /v1/chat/completions"))
		return
	}
	if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support tools this pass; use /v1/chat/completions"))
		return
	}

	lm, release := s.resolveAndLock(req.Model)
	if lm == nil {
		batch.SetResult(i, stores.LineError(customID, 404, "invalid_request_error",
			fmt.Sprintf("model %q not found", req.Model)))
		return
	}
	defer release() // held for this line's whole life, same reasoning as runJob (jobs_run.go)

	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error", err.Error()))
		return
	}
	ts, terr := s.resolveThink(req.Think())
	if terr != nil {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error", terr.Error()))
		return
	}
	tm := lm.templateFor(ts)
	if req.Sampling.ConstrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	bsys, bturns := messagesToTurns(req.Messages)
	ids, err := lm.promptForT(tm, bsys, bturns)
	if err != nil {
		batch.SetResult(i, stores.LineError(customID, 500, "api_error", "encode: "+err.Error()))
		return
	}
	gr, err := lm.prepare(req.Sampling, ids, lm.residentPath())
	if err != nil {
		batch.SetResult(i, stores.LineError(customID, prepareErrStatus(err), "invalid_request_error", err.Error()))
		return
	}

	jobID := "batch_req_" + reqID()
	gr.id = jobID
	s.jobs.CreatePending(jobID, lm.name, gr.promptIDs)

	// Stored before tryEnter, same ordering as runJob — POST .../cancel must reach a still-queued
	// line, not only a running one.
	bgCtx, bgCancel := context.WithCancel(s.lifeContext())
	defer bgCancel()
	if !batch.RegisterJob(i, jobID, s.jobs, bgCancel) {
		s.jobs.Finish(s.jobs.Get(jobID), jobs.Cancelled, nil, "batch cancelled", nil)
		batch.SetResult(i, stores.LineError(customID, statusCancelled, "cancelled", "batch cancelled"))
		return
	}

	ok, haltReason := lm.tryEnter(bgCtx, admissionRecord{promptIDs: gr.promptIDs, id: jobID, clock: gr.clock}, s.haltState)
	if !ok {
		why := notAdmitted(bgCtx, haltReason, lm) // a full queue is a failure, not a cancellation
		s.jobs.Finish(s.jobs.Get(jobID), why.state, nil, why.reason, nil)
		batch.SetResult(i, stores.LineError(customID, why.status, why.errType, why.reason))
		return
	}
	defer lm.exit()

	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, bturns, ts, func(t string) { rb.WriteString(t) })
	res, gerr := lm.drive(
		bgCtx, gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if r, failed := driveLineError(bgCtx, customID, gerr, res.cancelReason); failed {
		batch.SetResult(i, r)
		return
	}

	bmsg := map[string]any{"role": "assistant", "content": sb.String()}
	if rb.Len() > 0 {
		bmsg["reasoning_content"] = rb.String()
	}
	body := map[string]any{
		"id": jobID, "object": "chat.completion", "created": time.Now().Unix(), "model": lm.name,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       bmsg,
			"finish_reason": res.finish,
		}},
		"usage": newUsage(len(gr.promptIDs), res.nComp, res.reused),
	}
	batch.SetResult(i, stores.BatchLineResult{CustomID: customID, StatusCode: 200, Body: gr.clock.AddTurn(body)})
}

// runBatchMessageLine is runBatchChatLine's Anthropic twin — one line of a Message Batch,
// mirroring serveMessagesWith's (anthropic.go) non-streaming, non-vision, non-tool path for the
// identical reason runBatchChatLine mirrors handleCreateJob rather than calling it: reused wholesale
// where it's expensive (lm.prepare, lm.tryEnter, lm.drive, anthropicStopReason), replicated only in
// the thin validation glue neither existing handler can be called headless for.
func runBatchMessageLine(s *server, batch *stores.BatchRecord, i int, req anthropicReq) {
	customID := batch.CustomIDs[i]
	defer batch.DoneLine()
	defer batch.LineFinished()
	defer recoverBatchLine(batch, i) // after DoneLine: runs first, so the line's result is set before the batch can finalize

	if batch.IsCancelling() {
		batch.SetResult(i, stores.LineError(customID, statusCancelled, "cancelled", "batch cancelled"))
		return
	}
	if req.MaxTokens == nil || *req.MaxTokens <= 0 {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error", "max_tokens is required and must be > 0"))
		return
	}
	if len(req.Messages) == 0 {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error", "messages must not be empty"))
		return
	}
	if anthropicapi.ContainsImages(&req) {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support image inputs this pass; use /v1/messages"))
		return
	}
	if mode, _ := anthropicToolMode(req.ToolChoice); len(req.Tools) > 0 && mode != "none" {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error",
			"batch requests do not support tools this pass; use /v1/messages"))
		return
	}

	lm, release := s.resolveAndLock(req.Model)
	if lm == nil {
		batch.SetResult(i, stores.LineError(customID, 404, "not_found_error",
			fmt.Sprintf("model %q not found", req.Model)))
		return
	}
	defer release()

	system, turns, aerr := anthropicTurns(&req)
	if aerr != nil {
		batch.SetResult(i, stores.LineError(customID, aerr.Code, aerr.Kind, aerr.Msg))
		return
	}
	if err := lm.promptTooLargeForContext(anthropicInputBytes(&req)); err != nil {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error", err.Error()))
		return
	}
	ts, terr := s.resolveThink(req.ThinkRequest())
	if terr != nil {
		batch.SetResult(i, stores.LineError(customID, 400, "invalid_request_error", terr.Error()))
		return
	}
	sm, serr := req.ToSampling()
	if serr != nil {
		batch.SetResult(i, stores.LineError(customID, serr.Code, serr.Kind, serr.Msg))
		return
	}
	tm := lm.templateFor(ts)
	if sm.ConstrainsOutput() {
		tm = lm.constrainedTemplate(ts)
	}
	ids, err := lm.promptForT(tm, system, turns)
	if err != nil {
		batch.SetResult(i, stores.LineError(customID, 500, "api_error", "encode: "+err.Error()))
		return
	}
	gr, err := lm.prepare(sm, ids, lm.residentPath())
	if err != nil {
		batch.SetResult(i, stores.LineError(customID, prepareErrStatus(err), "invalid_request_error", err.Error()))
		return
	}

	jobID := "batch_req_" + reqID()
	gr.id = jobID
	s.jobs.CreatePending(jobID, lm.name, gr.promptIDs)

	bgCtx, bgCancel := context.WithCancel(s.lifeContext())
	defer bgCancel()
	if !batch.RegisterJob(i, jobID, s.jobs, bgCancel) {
		s.jobs.Finish(s.jobs.Get(jobID), jobs.Cancelled, nil, "batch cancelled", nil)
		batch.SetResult(i, stores.LineError(customID, statusCancelled, "cancelled", "batch cancelled"))
		return
	}

	ok, haltReason := lm.tryEnter(bgCtx, admissionRecord{promptIDs: gr.promptIDs, id: jobID, clock: gr.clock}, s.haltState)
	if !ok {
		why := notAdmitted(bgCtx, haltReason, lm) // a full queue is a failure, not a cancellation
		s.jobs.Finish(s.jobs.Get(jobID), why.state, nil, why.reason, nil)
		batch.SetResult(i, stores.LineError(customID, why.status, why.errType, why.reason))
		return
	}
	defer lm.exit()

	var sb, rb strings.Builder
	s.routeThink(lm, &gr, tm, turns, ts, func(t string) { rb.WriteString(t) })
	res, gerr := lm.drive(
		bgCtx, gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if r, failed := driveLineError(bgCtx, customID, gerr, res.cancelReason); failed {
		batch.SetResult(i, r)
		return
	}

	reason, seq := anthropicStopReason(res.finish, res.stopHit)
	content := []map[string]any{textBlock(sb.String())}
	if req.WantsThinking() && rb.Len() > 0 {
		content = append([]map[string]any{thinkingBlock(rb.String(), "")}, content...)
		if sb.Len() == 0 { // cut off while thinking: no answer, so no text block
			content = content[:1]
		}
	}
	body := map[string]any{
		"id": jobID, "type": "message", "role": "assistant", "model": lm.name,
		"content":       content,
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         anthropicUsage(len(gr.promptIDs), res.reused.Tokens, res.nComp),
	}
	batch.SetResult(i, stores.BatchLineResult{CustomID: customID, StatusCode: 200, Body: gr.clock.AddTurn(body)})
}

// driveLineError is a batch line's result when its drive did not complete: shutdown first, since a drive cut short by it
// returns its context error, then a failed drive, then a cancelled one. failed is false when the line has output to record.
func driveLineError(bgCtx context.Context, customID string, gerr error, cancelReason string) (r stores.BatchLineResult, failed bool) {
	switch {
	case errors.Is(context.Cause(bgCtx), errServerShutdown):
		return stores.LineError(customID, http.StatusServiceUnavailable, "server_shutdown", "server shutting down"), true
	case gerr != nil:
		return stores.LineError(customID, 500, "api_error", "generation failed: "+gerr.Error()), true
	case cancelReason != "":
		return stores.LineError(customID, statusCancelled, "cancelled", "generation cancelled: "+cancelReason), true
	}
	return stores.BatchLineResult{}, false
}

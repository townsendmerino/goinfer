package serveapp

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// setResult records line i's outcome under the batch's own lock. Safe to call concurrently from
// every line's goroutine: each writes a disjoint index, so only Status/the file-id fields
// (untouched here) actually need the lock — taken anyway for the same reason job.go's mu is,
// rather than reasoning a slice-index write never needs it.
func setResult(batch *batchRecord, i int, r batchLineResult) {
	batch.mu.Lock()
	batch.Results[i] = r
	batch.mu.Unlock()
}

func lineError(customID string, status int, errType, msg string) batchLineResult {
	return batchLineResult{CustomID: customID, StatusCode: status, ErrType: errType, ErrMsg: msg}
}

// setJobID records line i's job id under the batch's own lock. Found by -race, not by
// inspection, the same way job.go's own mu gained coverage in J3 (job.go:142's doc comment):
// requestCancel (batches.go) reads the whole JobIDs slice from a different goroutine while lines
// are still writing their own slot — disjoint indices don't save a plain slice write from racing a
// concurrent read of the slice's backing array with no synchronization between them.
func setJobID(batch *batchRecord, i int, id string) {
	batch.mu.Lock()
	batch.JobIDs[i] = id
	batch.mu.Unlock()
}

// runBatchChatLine is one line of an OpenAI batch (task-work-queue-2026-09.md J4): the same
// validate → resolveAndLock → encode → prepare → admission → drive pipeline handleCreateJob
// (jobs_http.go) and runJob (jobs_run.go) already run, replicated rather than called directly —
// both are tied to a single HTTP handler's or a fire-and-forget job's own lifecycle, neither of
// which fits "run synchronously in this line's own goroutine and hand back a result." Every
// EXPENSIVE step (admission, lm.drive) is the identical shared call, not a second implementation.
//
// A validation failure here is THIS LINE'S error, never the whole batch's — matches real batch-API
// semantics (task doc: "a line that asks for [scope this pass excludes] gets a per-line error").
func runBatchChatLine(s *server, batch *batchRecord, i int, req chatReq) {
	customID := batch.CustomIDs[i]
	defer batch.wg.Done()

	if len(req.Messages) == 0 {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error",
			"messages is required and must contain at least one message"))
		return
	}
	if imgs, ierr := chatImages(req.Messages); ierr != nil {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error", ierr.Error()))
		return
	} else if len(imgs) > 0 {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error",
			"batch requests do not support image inputs this pass; use /v1/chat/completions"))
		return
	}
	if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error",
			"batch requests do not support tools this pass; use /v1/chat/completions"))
		return
	}

	lm, release := s.resolveAndLock(req.Model)
	if lm == nil {
		setResult(batch, i, lineError(customID, 404, "invalid_request_error",
			fmt.Sprintf("model %q not found", req.Model)))
		return
	}
	defer release() // held for this line's whole life, same reasoning as runJob (jobs_run.go)

	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error", err.Error()))
		return
	}
	ids, err := lm.chatPrompt(req.Messages)
	if err != nil {
		setResult(batch, i, lineError(customID, 500, "api_error", "encode: "+err.Error()))
		return
	}
	gr, err := lm.prepare(req.sampling, ids, lm.adapter == "")
	if err != nil {
		setResult(batch, i, lineError(customID, prepareErrStatus(err), "invalid_request_error", err.Error()))
		return
	}

	jobID := "batch_req_" + reqID()
	gr.id = jobID
	s.jobs.createPending(jobID, lm.name, gr.promptIDs)
	setJobID(batch, i, jobID)

	// Stored before tryEnter, same ordering as runJob — POST .../cancel must reach a still-queued
	// line, not only a running one.
	bgCtx, bgCancel := context.WithCancel(context.Background())
	s.jobs.setCancel(jobID, bgCancel)
	defer bgCancel()

	ok, haltReason := lm.tryEnter(bgCtx, admissionRecord{promptIDs: gr.promptIDs, id: jobID}, s.haltState)
	if !ok {
		why := notAdmitted(bgCtx, haltReason, lm) // a full queue is a failure, not a cancellation
		s.jobs.finish(s.jobs.get(jobID), why.state, nil, why.reason, nil)
		setResult(batch, i, lineError(customID, why.status, why.errType, why.reason))
		return
	}
	defer lm.exit()

	var sb strings.Builder
	finish, nComp, _, _, prefillReused, cancelReason, gerr := lm.drive(
		bgCtx, gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		setResult(batch, i, lineError(customID, 500, "api_error", "generation failed: "+gerr.Error()))
		return
	}
	if cancelReason != "" {
		setResult(batch, i, lineError(customID, statusCancelled, "cancelled", "generation cancelled: "+cancelReason))
		return
	}

	body := map[string]any{
		"id": jobID, "object": "chat.completion", "created": time.Now().Unix(), "model": lm.name,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": sb.String()},
			"finish_reason": finish,
		}},
		"usage": usage{
			PromptTokens: len(gr.promptIDs), CompletionTokens: nComp,
			TotalTokens: len(gr.promptIDs) + nComp, PrefillReusedTokens: prefillReused,
		},
	}
	setResult(batch, i, batchLineResult{CustomID: customID, StatusCode: 200, Body: body})
}

// runBatchMessageLine is runBatchChatLine's Anthropic twin — one line of a Message Batch,
// mirroring serveMessagesWith's (anthropic.go) non-streaming, non-vision, non-tool path for the
// identical reason runBatchChatLine mirrors handleCreateJob rather than calling it: reused wholesale
// where it's expensive (lm.prepare, lm.tryEnter, lm.drive, anthropicStopReason), replicated only in
// the thin validation glue neither existing handler can be called headless for.
func runBatchMessageLine(s *server, batch *batchRecord, i int, req anthropicReq) {
	customID := batch.CustomIDs[i]
	defer batch.wg.Done()

	if req.MaxTokens == nil || *req.MaxTokens <= 0 {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error", "max_tokens is required and must be > 0"))
		return
	}
	if len(req.Messages) == 0 {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error", "messages must not be empty"))
		return
	}
	if imgs, ierr := anthropicImages(&req); ierr != nil {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error", ierr.Error()))
		return
	} else if len(imgs) > 0 {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error",
			"batch requests do not support image inputs this pass; use /v1/messages"))
		return
	}
	if mode, _ := anthropicToolMode(req.ToolChoice); len(req.Tools) > 0 && mode != "none" {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error",
			"batch requests do not support tools this pass; use /v1/messages"))
		return
	}

	lm, release := s.resolveAndLock(req.Model)
	if lm == nil {
		setResult(batch, i, lineError(customID, 404, "not_found_error",
			fmt.Sprintf("model %q not found", req.Model)))
		return
	}
	defer release()

	system, turns, aerr := anthropicTurns(&req)
	if aerr != nil {
		setResult(batch, i, lineError(customID, aerr.code, aerr.kind, aerr.msg))
		return
	}
	if err := lm.promptTooLargeForContext(anthropicInputBytes(&req)); err != nil {
		setResult(batch, i, lineError(customID, 400, "invalid_request_error", err.Error()))
		return
	}
	ids, err := lm.promptFor(system, turns)
	if err != nil {
		setResult(batch, i, lineError(customID, 500, "api_error", "encode: "+err.Error()))
		return
	}
	gr, err := lm.prepare(req.toSampling(), ids, lm.adapter == "")
	if err != nil {
		setResult(batch, i, lineError(customID, prepareErrStatus(err), "invalid_request_error", err.Error()))
		return
	}

	jobID := "batch_req_" + reqID()
	gr.id = jobID
	s.jobs.createPending(jobID, lm.name, gr.promptIDs)
	setJobID(batch, i, jobID)

	bgCtx, bgCancel := context.WithCancel(context.Background())
	s.jobs.setCancel(jobID, bgCancel)
	defer bgCancel()

	ok, haltReason := lm.tryEnter(bgCtx, admissionRecord{promptIDs: gr.promptIDs, id: jobID}, s.haltState)
	if !ok {
		why := notAdmitted(bgCtx, haltReason, lm) // a full queue is a failure, not a cancellation
		s.jobs.finish(s.jobs.get(jobID), why.state, nil, why.reason, nil)
		setResult(batch, i, lineError(customID, why.status, why.errType, why.reason))
		return
	}
	defer lm.exit()

	var sb strings.Builder
	finish, nComp, _, stopHitOut, _, cancelReason, gerr := lm.drive(
		bgCtx, gr, s.gens, s.jobs, func(t string) { sb.WriteString(t) })
	if gerr != nil {
		setResult(batch, i, lineError(customID, 500, "api_error", "generation failed: "+gerr.Error()))
		return
	}
	if cancelReason != "" {
		setResult(batch, i, lineError(customID, statusCancelled, "cancelled", "generation cancelled: "+cancelReason))
		return
	}

	reason, seq := anthropicStopReason(finish, stopHitOut)
	body := map[string]any{
		"id": jobID, "type": "message", "role": "assistant", "model": lm.name,
		"content":       []map[string]any{textBlock(sb.String())},
		"stop_reason":   reason,
		"stop_sequence": seq,
		"usage":         map[string]any{"input_tokens": len(gr.promptIDs), "output_tokens": nComp},
	}
	setResult(batch, i, batchLineResult{CustomID: customID, StatusCode: 200, Body: body})
}

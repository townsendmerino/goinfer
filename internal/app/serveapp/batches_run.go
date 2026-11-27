package serveapp

import (
	"context"
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/batchapi"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
)

const batchLineWorkers = batchapi.LineWorkers

var runBatchPool = batchapi.RunPool

func runBatchChatLine(s *server, batch *stores.BatchRecord, i int, req chatReq) {
	customID := batch.CustomIDs[i]
	defer batch.DoneLine()
	defer batch.LineFinished()
	defer recoverBatchLine(batch, i)

	if errRes, ok := batchapi.ValidateChatLine(customID, req, statusCancelled, batch.IsCancelling()); !ok {
		batch.SetResult(i, errRes)
		return
	}

	lm, release := s.resolveAndLock(req.Model)
	if lm == nil {
		batch.SetResult(i, stores.LineError(customID, 404, "invalid_request_error",
			fmt.Sprintf("model %q not found", req.Model)))
		return
	}
	defer release()

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

	bgCtx, bgCancel := context.WithCancel(s.lifeContext())
	defer bgCancel()
	if !batch.RegisterJob(i, jobID, s.jobs, bgCancel) {
		s.jobs.Finish(s.jobs.Get(jobID), jobs.Cancelled, nil, "batch cancelled", nil)
		batch.SetResult(i, stores.LineError(customID, statusCancelled, "cancelled", "batch cancelled"))
		return
	}

	ok, haltReason := lm.tryEnter(bgCtx, admissionRecord{promptIDs: gr.promptIDs, id: jobID, clock: gr.clock}, s.haltState)
	if !ok {
		why := notAdmitted(bgCtx, haltReason, lm)
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

	body := batchapi.ChatCompletionBody(jobID, lm.name, sb.String(), rb.String(), res.finish, len(gr.promptIDs), res.nComp, res.reused)
	batch.SetResult(i, stores.BatchLineResult{CustomID: customID, StatusCode: 200, Body: gr.clock.AddTurn(body)})
}

func runBatchMessageLine(s *server, batch *stores.BatchRecord, i int, req anthropicReq) {
	customID := batch.CustomIDs[i]
	defer batch.DoneLine()
	defer batch.LineFinished()
	defer recoverBatchLine(batch, i)

	if errRes, ok := batchapi.ValidateMessageLine(customID, &req, statusCancelled, batch.IsCancelling()); !ok {
		batch.SetResult(i, errRes)
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
		why := notAdmitted(bgCtx, haltReason, lm)
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

	body := batchapi.MessageBody(jobID, lm.name, sb.String(), rb.String(), res.finish, res.stopHit, req.WantsThinking(), len(gr.promptIDs), res.reused.Tokens, res.nComp)
	batch.SetResult(i, stores.BatchLineResult{CustomID: customID, StatusCode: 200, Body: gr.clock.AddTurn(body)})
}

func driveLineError(bgCtx context.Context, customID string, gerr error, cancelReason string) (r stores.BatchLineResult, failed bool) {
	return batchapi.ClassifyDriveError(bgCtx, errServerShutdown, statusCancelled, customID, gerr, cancelReason)
}

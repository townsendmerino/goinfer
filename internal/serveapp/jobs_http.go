package serveapp

import (
	"net/http"
	"time"
)

// J3 (task-work-queue-2026-09.md): POST /v1/jobs submits a generation and returns its id
// immediately; GET /v1/jobs/{id} polls state+result; GET /v1/jobs/{id}/events re-attaches to its
// output (replay-then-live); DELETE /v1/jobs/{id} cancels it. Text-only chat body for this pass —
// see jobs_run.go's own doc comment and the task doc's closure note for what's scoped out
// (vision, tools, the K4 user/user_id key that doesn't exist yet).

// handleCreateJob is POST /v1/jobs. Deliberately NOT routed through withModel/handleChat: those
// bake in "hold the model for exactly this synchronous handler's lifetime", which is precisely
// wrong here — resolveAndLock is called directly so the returned release can be handed to the
// background goroutine instead of deferred in this handler.
func (s *server) handleCreateJob(w http.ResponseWriter, r *http.Request) {
	var req chatReq
	if !decodeJSON(w, r, &req) {
		return
	}
	if len(req.Messages) == 0 {
		writeErr(w, http.StatusBadRequest, "messages is required and must contain at least one message")
		return
	}
	if imgs, ierr := chatImages(req.Messages); ierr != nil {
		writeErr(w, http.StatusBadRequest, ierr.Error())
		return
	} else if len(imgs) > 0 {
		writeErr(w, http.StatusBadRequest, "POST /v1/jobs does not support image inputs yet; use /v1/chat/completions")
		return
	}
	if len(req.Tools) > 0 && toolChoiceMode(req.ToolChoice) != "none" {
		writeErr(w, http.StatusBadRequest, "POST /v1/jobs does not support tools yet; use /v1/chat/completions")
		return
	}

	lm, release := s.resolveAndLock(req.Model)
	if lm == nil {
		s.modelNotFound(w, req.Model)
		return
	}
	// From here, release() is EITHER called by this handler (every early-return path below) OR
	// handed to runJob to call once the job is fully finished — never both, never neither.
	if err := lm.promptTooLargeForContext(chatInputBytes(req.Messages)); err != nil {
		release()
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	ids, err := lm.chatPrompt(req.Messages)
	if err != nil {
		release()
		writeServerErr(w, "encode: "+err.Error())
		return
	}
	gr, err := lm.prepare(req.sampling, ids, lm.adapter == "")
	if err != nil {
		release()
		writeErr(w, prepareErrStatus(err), err.Error())
		return
	}
	if hi := s.haltState(); hi != nil {
		release()
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": "halted", "reason": hi.reason})
		return
	}

	id := "job_" + reqID()
	gr.id = id
	s.jobs.createPending(id, lm.name, gr.promptIDs)
	go runJob(s, lm, release, gr, admissionRecord{promptIDs: gr.promptIDs}, time.Now().Unix())

	writeJSON(w, http.StatusAccepted, map[string]any{"id": id, "status": string(jobPending)})
}

// handleGetJob is GET /v1/jobs/{id}: state + result, a plain JSON projection of a job snapshot.
// Uses snapshot(), not get() — this handler's goroutine is a genuine concurrent reader against
// whichever goroutine (runJob, or a synchronous handler) is still mutating this job via
// markRunning/finish; see jobStore.snapshot's own doc comment.
func (s *server) handleGetJob(w http.ResponseWriter, r *http.Request) {
	j, ok := s.jobs.snapshot(r.PathValue("id"))
	if !ok {
		writeErr(w, http.StatusNotFound, "job not found")
		return
	}
	out := map[string]any{
		"id": j.ID, "model": j.Model, "status": string(j.State), "created": j.Created,
	}
	if j.Started != nil {
		out["started"] = *j.Started
	}
	if j.Finished != nil {
		out["finished"] = *j.Finished
	}
	if j.Usage != nil {
		out["usage"] = j.Usage
	}
	if j.Error != "" {
		out["error"] = j.Error
	}
	if j.result != nil {
		out["result"] = j.result
	}
	writeJSON(w, http.StatusOK, out)
}

// handleCancelJob is DELETE /v1/jobs/{id} — K1's cancel, addressed by job id (task doc: "the two
// registries are joined, not parallel"). 200 either way (found or not), mirroring
// handleAdminGenerationCancel's own idempotent shape (admin.go:38) — a caller cancelling an id
// that already finished, or never existed, is not an error: the job is stopped either way, which
// is what was asked for.
func (s *server) handleCancelJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	found := s.jobs.cancel(id)
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "cancelled": found})
}

// handleJobEvents is GET /v1/jobs/{id}/events — replay everything already emitted, then continue
// live until the job reaches a terminal state or the client disconnects. A reconnect is a NEW
// call to this handler (a new snapshot(0)), not a resumed one — see jobeventlog.go's own doc
// comment for why that needs no per-reader bookkeeping.
func (s *server) handleJobEvents(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if s.jobs.get(id) == nil {
		writeErr(w, http.StatusNotFound, "job not found")
		return
	}
	log := s.jobs.eventLog(id)
	if log == nil {
		// A job restored from the journal on restart (or otherwise never attach()'d) has no live
		// output to replay — its own GET /v1/jobs/{id} still answers state+result.
		writeErr(w, http.StatusGone, "no live event stream for this job (server restarted, or it predates this process)")
		return
	}
	ss, ok := sseStart(w)
	if !ok {
		return
	}
	pos := 0
	for {
		events, wait, done := log.snapshot(pos)
		for _, e := range events {
			ss.frame("data: %s\n\n", e)
		}
		pos += len(events)
		if done {
			sseDone(ss)
			return
		}
		select {
		case <-wait:
		case <-r.Context().Done():
			return
		}
	}
}

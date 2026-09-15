package serveapp

import (
	"context"
	"encoding/json"
)

// mustJSON marshals v (always one of this package's own known-safe map[string]any chunk shapes,
// the same values sseSend already trusts json.Marshal not to fail on) into the event log's stored
// payload form. A marshal failure here would mean a chunk-builder started returning something
// unmarshalable, which is a bug to see loudly rather than silently drop a frame for.
func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic("serveapp: job event payload failed to marshal: " + err.Error())
	}
	return b
}

// runJob is J3's asynchronous counterpart to serveChatText (task-work-queue-2026-09.md): the same
// admission + drive() pipeline, but decoupled from any HTTP request's lifetime. Launched with
// `go runJob(...)` from handleCreateJob, BEFORE that handler returns.
//
// release is s.resolveAndLock's own release func (liveness.go:69) — normally deferred by
// withModel for the span of one synchronous handler; here it must be deferred for the span of
// this WHOLE job instead, so /admin unload still can't free the model out from under a
// generation that is still running after its submitting POST request has already returned.
func runJob(s *server, lm *loadedModel, release func(), gr genRequest, rec admissionRecord, created int64) {
	defer release()
	log := newJobEventLog()
	s.jobs.attach(gr.id, log)
	defer log.markDone()

	// bgCtx, not any http.Request's context: this job outlives the POST that created it. Stored
	// before tryEnter is even attempted, so DELETE /v1/jobs/{id} can stop a still-QUEUED job, not
	// just a running one — cancelling it propagates to whatever drive derives from it once
	// admission is granted, the same way any other context cancellation in this codebase does.
	bgCtx, bgCancel := context.WithCancel(context.Background())
	s.jobs.setCancel(gr.id, bgCancel)
	defer bgCancel() // release resources tied to bgCtx once this job is fully done, either way

	ok, haltReason := lm.tryEnter(bgCtx, rec, s.haltState)
	if !ok {
		reason := haltReason
		if reason == "" {
			reason = "cancelled before a turn was granted" // bgCtx ended while still queued (DELETE)
		}
		s.jobs.finish(s.jobs.get(gr.id), jobCancelled, nil, reason, nil)
		return
	}
	defer lm.exit()

	_, _, _, _, _, _, _ = lm.drive(bgCtx, gr, s.gens, s.jobs, func(t string) {
		log.append(mustJSON(chatChunk(gr.id, created, lm.name, delta{Content: t}, nil)))
	})
	// drive's own J2/J3 integration (openai.go's job block) already recorded the terminal
	// transition and result on this job — nothing left to do here but let the defers run.
}

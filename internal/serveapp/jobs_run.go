package serveapp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
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

// refusal says why tryEnter did not admit a background request, which it reports only as ok=false plus a
// halt reason. Three different things hide behind that: a halt, the request's own context ending
// (DELETE /v1/jobs/{id} while it waited), and the model's queue being full. The last one used to be
// recorded as "cancelled before a turn was granted" — telling a user they cancelled something they did not.
type refusal struct {
	state   jobState
	status  int    // for a batch line's per-line error
	errType string // OpenAI error type, for the job's closing error event
	reason  string
}

func notAdmitted(ctx context.Context, haltReason, model string) refusal {
	switch {
	case haltReason != "":
		return refusal{jobCancelled, statusCancelled, "cancelled", haltReason}
	case ctx.Err() != nil:
		return refusal{jobCancelled, statusCancelled, "cancelled", "cancelled before a turn was granted"}
	default:
		return refusal{jobFailed, http.StatusTooManyRequests, "rate_limit_error", fmt.Sprintf("model %q queue full; retry", model)}
	}
}

// jobTerminalEvents are the closing events of a job's stream, built from its final record (W27): the same
// shapes /v1/chat/completions ends a stream with — a chunk carrying finish_reason, and a usage chunk — or,
// for a job that failed, an OpenAI-style error event. Without them a client re-attaching through
// GET /v1/jobs/{id}/events got the text and then [DONE], and had to poll to learn whether the reply was
// complete, how many tokens it used, or that it had failed at all.
func jobTerminalEvents(j job, created int64, errType string) [][]byte {
	var out [][]byte
	switch {
	case j.State == jobFailed:
		if errType == "" {
			errType = "api_error"
		}
		out = append(out, mustJSON(map[string]any{"error": map[string]any{"message": j.Error, "type": errType}}))
	case j.result != nil && j.result.FinishReason != "":
		f := j.result.FinishReason
		out = append(out, mustJSON(chatChunk(j.ID, created, j.Model, delta{}, &f)))
	case j.State == jobCancelled:
		f := "cancelled"
		out = append(out, mustJSON(chatChunk(j.ID, created, j.Model, delta{}, &f)))
	}
	if j.Usage != nil {
		out = append(out, mustJSON(usageChunk(j.ID, created, j.Model, *j.Usage)))
	}
	return out
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

	rec.id = gr.id // W28: so GET /v1/jobs/{id} can report this job's place in line while it waits
	ok, haltReason := lm.tryEnter(bgCtx, rec, s.haltState)
	if !ok {
		why := notAdmitted(bgCtx, haltReason, lm.name)
		s.jobs.finish(s.jobs.get(gr.id), why.state, nil, why.reason, nil)
		if j, found := s.jobs.snapshot(gr.id); found {
			for _, e := range jobTerminalEvents(j, created, why.errType) {
				log.append(e)
			}
		}
		return
	}
	defer lm.exit()

	_, _, _, _, _, _, _ = lm.drive(bgCtx, gr, s.gens, s.jobs, func(t string) {
		log.append(mustJSON(chatChunk(gr.id, created, lm.name, delta{Content: t}, nil)))
	})
	// drive's own J2/J3 integration (openai.go's job block) already recorded the terminal transition and
	// result on this job; the stream closes with the same finish/usage/error events the chat route ends on.
	if j, found := s.jobs.snapshot(gr.id); found {
		for _, e := range jobTerminalEvents(j, created, "") {
			log.append(e)
		}
	}
}

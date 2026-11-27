package serveapp

import (
	"fmt"
	"net/http"
	"os"
	"runtime/debug"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
)

// A panic on a goroutine a handler spawned is not recovered by net/http (which recovers only its own handler goroutine), so it
// ends the process. The detached job and batch goroutines run the whole request pipeline, so each takes one of these as its first
// deferred call: the panic fails that one job or line, with a generic message to the client and the stack on stderr, and the server
// keeps serving. Each must be the deferred function itself (recover only works there), registered AFTER any wg.Done so it runs first.

func batchTooLarge(w http.ResponseWriter, n int) bool {
	if n <= stores.MaxBatchLines {
		return false
	}
	writeErr(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("a batch holds at most %d requests, this one has %d: split it", stores.MaxBatchLines, n))
	return true
}

// recoverBatchLine fails line i of a batch when its goroutine panics.
func recoverBatchLine(batch *stores.BatchRecord, i int) {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: batch %s line %d panicked: %v\n%s", batch.ID, i, r, debug.Stack())
		batch.SetResult(i, stores.LineError(batch.CustomIDs[i], http.StatusInternalServerError, "api_error", "internal error"))
	}
}

// recoverJob fails a job when its goroutine panics, and closes the event stream the way a failed job does.
func (s *server) recoverJob(id string, log *jobs.EventLog, created int64) {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: job %s panicked: %v\n%s", id, r, debug.Stack())
		if j := s.jobs.Get(id); j != nil {
			s.jobs.Finish(j, jobs.Failed, nil, "internal error", nil)
		}
		if j, ok := s.jobs.Snapshot(id); ok && log != nil {
			for _, e := range jobTerminalEvents(j, created, "api_error") {
				log.Append(e)
			}
		}
	}
}

// recoverHeartbeat ends an SSE heartbeat whose write panicked; the response it pinged carries on without keep-alives.
func recoverHeartbeat() {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: SSE heartbeat panicked: %v\n%s", r, debug.Stack())
	}
}

// recoverDrain keeps a panic while an unloaded model is drained and freed from taking the process down. The model stays marked
// draining, so /health shows a free that never finished.
func recoverDrain(name string) {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: unloading model %q panicked: %v\n%s", name, r, debug.Stack())
	}
}

// Request amplifiers: fields a compatible client sends a handful of, whose cost grows with their count at every generated token (stop
// strings are matched against the tail each step; a tool-name union is a branch per tool at every step of the name). Each cap is far
// above what a real request carries and rejects with a 400 before any work.
const (
	maxStopSequences    = 16 // OpenAI documents 4
	maxToolsPerRequest  = 128
	maxLogitBiasEntries = 1024 // each entry is applied to every step's logits
)

func checkToolCount(n int) error {
	if n > maxToolsPerRequest {
		return fmt.Errorf("a request may declare at most %d tools (got %d)", maxToolsPerRequest, n)
	}
	return nil
}

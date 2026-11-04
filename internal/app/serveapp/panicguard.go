package serveapp

import (
	"fmt"
	"net/http"
	"os"
	"runtime/debug"
	"time"
)

// A panic on a goroutine a handler spawned is not recovered by net/http (which recovers only its own handler goroutine), so it
// ends the process. The detached job and batch goroutines run the whole request pipeline, so each takes one of these as its first
// deferred call: the panic fails that one job or line, with a generic message to the client and the stack on stderr, and the server
// keeps serving. Each must be the deferred function itself (recover only works there), registered AFTER any wg.Done so it runs first.

// maxBatchLines bounds the requests in one batch. Each line becomes a goroutine and a parsed request before any admission queue sees
// it, and the body cap alone (64 MiB) admits millions of minimal lines.
const maxBatchLines = 10000

func batchTooLarge(w http.ResponseWriter, n int) bool {
	if n <= maxBatchLines {
		return false
	}
	writeErr(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("a batch holds at most %d requests, this one has %d: split it", maxBatchLines, n))
	return true
}

// recoverBatchLine fails line i of a batch when its goroutine panics.
func recoverBatchLine(batch *batchRecord, i int) {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: batch %s line %d panicked: %v\n%s", batch.ID, i, r, debug.Stack())
		setResult(batch, i, lineError(batch.CustomIDs[i], http.StatusInternalServerError, "api_error", "internal error"))
	}
}

// recoverJob fails a job when its goroutine panics, and closes the event stream the way a failed job does.
func (s *server) recoverJob(id string, log *jobEventLog, created int64) {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: job %s panicked: %v\n%s", id, r, debug.Stack())
		if j := s.jobs.get(id); j != nil {
			s.jobs.finish(j, jobFailed, nil, "internal error", nil)
		}
		if j, ok := s.jobs.snapshot(id); ok && log != nil {
			for _, e := range jobTerminalEvents(j, created, "api_error") {
				log.append(e)
			}
		}
	}
}

// recoverFinalize keeps a panic while a batch's output is assembled from taking the process down: the batch is marked failed.
func (s *server) recoverFinalize(b *batchRecord) {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: finalizing batch %s panicked: %v\n%s", b.ID, r, debug.Stack())
		b.Status = "failed"
		now := time.Now()
		b.CompletedAt = &now
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

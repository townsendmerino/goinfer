package serveapp

import "sync"

// jobEventLog is J3's replay-then-live buffer (task-work-queue-2026-09.md): every event a job
// emits is appended here, in order, and kept for the job's lifetime — a client that reconnects to
// GET /v1/jobs/{id}/events is a NEW reader starting its own snapshot(0), not a resumed one, so it
// gets the full backlog before going live. No per-reader registration or cleanup is needed: a
// reader just holds a position (an int) and re-snapshots.
//
// The channel-swap-on-write pattern (append/markDone close and replace ch under the lock) is what
// lets a reader `select` on "more data or done" alongside `r.Context().Done()` in the same
// statement — a sync.Cond can't do that, since Wait() isn't select-compatible.
type jobEventLog struct {
	mu     sync.Mutex
	events [][]byte
	done   bool
	ch     chan struct{}
}

func newJobEventLog() *jobEventLog {
	return &jobEventLog{ch: make(chan struct{})}
}

// append adds one pre-marshaled event (a chatChunk-shaped JSON payload, same as the streaming
// SSE handlers already send) and wakes every waiter.
func (l *jobEventLog) append(payload []byte) {
	l.mu.Lock()
	l.events = append(l.events, payload)
	ch := l.ch
	l.ch = make(chan struct{})
	l.mu.Unlock()
	close(ch)
}

// markDone marks the log closed — no further events will ever be appended — and wakes every
// waiter so a blocked reader notices the job ended even if it produced no final event of its own.
// Idempotent-safe to call once, by runJob's own defer chain; calling it twice would double-close
// an already-replaced channel and panic, so it must not be called more than once per job.
func (l *jobEventLog) markDone() {
	l.mu.Lock()
	l.done = true
	ch := l.ch
	l.ch = make(chan struct{})
	l.mu.Unlock()
	close(ch)
}

// snapshot returns every event from index `from` onward, a channel that closes the next time
// append or markDone fires (so a caller can select on it), and whether the log is done. Reading
// events and computing the wait channel under the SAME lock acquisition is what makes this race-
// free: a snapshot() that returns 0 new events and a not-yet-closed wait channel is a guarantee
// that nothing will be missed by waiting on that exact channel next — either it is closed by an
// append that already has new data for the next snapshot, or by markDone.
func (l *jobEventLog) snapshot(from int) (events [][]byte, wait <-chan struct{}, done bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if from < 0 || from > len(l.events) {
		from = len(l.events)
	}
	return l.events[from:], l.ch, l.done
}

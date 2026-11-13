package jobs

import "sync"

// EventLog is a job's replay-then-live buffer (docs/tasks/task-work-queue-2026-09.md): every event a job emits is
// appended here in order and kept for the job's lifetime. A client reconnecting to GET /v1/jobs/{id}/events is a new
// reader starting its own snapshot(0), not a resumed one, so it gets the full backlog before going live. No
// per-reader registration or cleanup is needed: a reader holds a position (an int) and re-snapshots.
//
// Append and markDone close and replace ch under the lock, which lets a reader select on "more data or done"
// alongside r.Context().Done() in one statement; a sync.Cond cannot, since Wait is not select-compatible.
type EventLog struct {
	mu     sync.Mutex
	events [][]byte
	done   bool
	ch     chan struct{}
}

func NewEventLog() *EventLog {
	return &EventLog{ch: make(chan struct{})}
}

// Append adds one pre-marshaled event (a chatChunk-shaped JSON payload, same as the streaming
// SSE handlers already send) and wakes every waiter.
func (l *EventLog) Append(payload []byte) {
	l.mu.Lock()
	l.events = append(l.events, payload)
	ch := l.ch
	l.ch = make(chan struct{})
	l.mu.Unlock()
	close(ch)
}

// MarkDone marks the log closed (no further events will be appended) and wakes every waiter, so a blocked reader
// notices the job ended even if it produced no final event of its own. It is called once per job, by runJob's defer
// chain.
func (l *EventLog) MarkDone() {
	l.mu.Lock()
	l.done = true
	ch := l.ch
	l.ch = make(chan struct{})
	l.mu.Unlock()
	close(ch)
}

// Snapshot returns every event from index `from` onward, a channel that closes the next time append or markDone fires
// (so a caller can select on it), and whether the log is done. Reading the events and taking the wait channel under
// one lock acquisition is what makes this race-free: a snapshot that returns no new events and a not-yet-closed
// channel guarantees nothing is missed by waiting on that exact channel next.
func (l *EventLog) Snapshot(from int) (events [][]byte, wait <-chan struct{}, done bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if from < 0 || from > len(l.events) {
		from = len(l.events)
	}
	return l.events[from:], l.ch, l.done
}

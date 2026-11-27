package streamframe

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"runtime/debug"
	"sync"
	"time"
)

// WriteTimeout bounds a single SSE frame's write and flush so a stalled reader frees the handler instead of pinning it.
var WriteTimeout = 30 * time.Second

// HeartbeatInterval controls how often Heartbeat writes a keep-alive SSE comment frame while a handler is silent.
var HeartbeatInterval = 10 * time.Second

// SSEWriter serializes writes and flushes onto an SSE ResponseWriter under a per-frame write deadline.
// Both the OpenAI and Anthropic stream surfaces write through it so a background heartbeat goroutine and
// incremental handler deltas never race on net/http's underlying response writer. The first write error is
// sticky and cancels any context derived via CancelOnFail.
type SSEWriter struct {
	mu    sync.Mutex
	w     http.ResponseWriter
	f     http.Flusher
	rc    *http.ResponseController
	onErr func()
	err   error
}

// NewSSEWriter wraps w and f with an http.ResponseController for per-frame write deadlines.
func NewSSEWriter(w http.ResponseWriter, f http.Flusher) *SSEWriter {
	return &SSEWriter{w: w, f: f, rc: http.NewResponseController(w)}
}

// NewSSEWriterRaw constructs an SSEWriter without an http.ResponseController, for unit tests driving custom writers.
func NewSSEWriterRaw(w http.ResponseWriter, f http.Flusher) *SSEWriter {
	return &SSEWriter{w: w, f: f}
}

// Lock acquires the writer's serialization mutex. Used by race tests that read an underlying recorder concurrently.
func (s *SSEWriter) Lock() { s.mu.Lock() }

// Unlock releases the writer's serialization mutex.
func (s *SSEWriter) Unlock() { s.mu.Unlock() }

// Frame formats and flushes one SSE frame under the writer's lock and write deadline.
func (s *SSEWriter) Frame(format string, args ...any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return s.err
	}
	if s.rc != nil {
		_ = s.rc.SetWriteDeadline(time.Now().Add(WriteTimeout))
	}
	if _, err := fmt.Fprintf(s.w, format, args...); err != nil {
		s.err = err
		if s.onErr != nil {
			s.onErr()
		}
		return err
	}
	if s.rc != nil {
		if err := s.rc.Flush(); err != nil && !errors.Is(err, http.ErrNotSupported) {
			s.err = err
			if s.onErr != nil {
				s.onErr()
			}
			return err
		}
	} else if s.f != nil {
		s.f.Flush()
	}
	return nil
}

// Err returns the first sticky write or flush error observed on s, or nil if the stream is healthy.
func (s *SSEWriter) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// CancelOnFail derives a child context from parent that cancels as soon as s encounters a write or flush failure.
// If s has already failed, the returned context is cancelled immediately.
func (s *SSEWriter) CancelOnFail(parent context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		cancel()
		return ctx, cancel
	}
	if prev := s.onErr; prev != nil {
		s.onErr = func() { prev(); cancel() }
	} else {
		s.onErr = cancel
	}
	return ctx, cancel
}

// JSONSender returns a closure that marshals payload as a named SSE event and invokes cancel on the first write error.
func JSONSender(w http.ResponseWriter, f http.Flusher, cancel context.CancelFunc) func(event string, payload any) {
	sw := NewSSEWriter(w, f)
	sw.onErr = cancel
	return func(event string, payload any) {
		b, err := json.Marshal(payload)
		if err != nil {
			return
		}
		_ = sw.Frame("event: %s\ndata: %s\n\n", event, b)
	}
}

// SendData marshals v as JSON and writes an unnamed `data: <json>` SSE frame.
func SendData(ss *SSEWriter, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	_ = ss.Frame("data: %s\n\n", b)
}

// SendEvent marshals v as JSON and writes a named `event: <event>\ndata: <json>` SSE frame.
func SendEvent(ss *SSEWriter, event string, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	_ = ss.Frame("event: %s\ndata: %s\n\n", event, b)
}

// SendDone writes the OpenAI `data: [DONE]` stream terminator.
func SendDone(ss *SSEWriter) {
	_ = ss.Frame("data: [DONE]\n\n")
}

// SendOpenAIError writes an OpenAI-shaped mid-stream error frame.
func SendOpenAIError(ss *SSEWriter, msg string) {
	SendData(ss, map[string]any{"error": map[string]any{"message": msg, "type": "api_error"}})
}

// SendAnthropicError writes an Anthropic-shaped mid-stream `error` event.
func SendAnthropicError(ss *SSEWriter, msg string) {
	SendEvent(ss, "error", map[string]any{
		"type": "error", "error": map[string]any{"type": "api_error", "message": msg},
	})
}

func recoverHeartbeat() {
	if r := recover(); r != nil {
		fmt.Fprintf(os.Stderr, "serve: SSE heartbeat panicked: %v\n%s", r, debug.Stack())
	}
}

// Heartbeat starts a background goroutine that writes `: ping` SSE comment frames at HeartbeatInterval until the
// returned stop function is called. Calling stop joins the goroutine and is idempotent.
func Heartbeat(ss *SSEWriter) (stop func()) {
	done, finished := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(finished)
		defer recoverHeartbeat()
		t := time.NewTicker(HeartbeatInterval)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				_ = ss.Frame(": ping\n\n")
			}
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() {
			close(done)
			<-finished
		})
	}
}

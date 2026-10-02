package serveapp

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// -log-requests (R25, docs/tasks/task-first-hour.md): one line per generation request on stderr — route, model, status, prompt and completion tokens,
// time to first token, total time. A cold-user run on goinfer-serve learned its prompt sizes only from error bodies: nothing said what a client had sent.
// Off by default, so a quiet server stays quiet.
//
// The middleware owns the status and the clock; the token counts and the first-token time can only be known where tokens are read, which is
// streamTokens — the shared tail of every generation (drive and driveVL), so a request that makes several generations (a tool loop) is counted once
// per request: prompt tokens of its first generation, completion tokens summed, first token of the first.

type reqTrace struct {
	start time.Time
	mu    sync.Mutex
	model string
	// seen is whether any generation reported in; prompt is the first one's prompt, completion the sum of all of them.
	seen               bool
	prompt, completion int
	first              time.Time
}

type reqTraceKey struct{}

func traceFrom(ctx context.Context) *reqTrace {
	t, _ := ctx.Value(reqTraceKey{}).(*reqTrace)
	return t
}

// firstToken stamps the first token the request's first generation produced.
func (t *reqTrace) firstToken() {
	t.mu.Lock()
	if t.first.IsZero() {
		t.first = time.Now()
	}
	t.mu.Unlock()
}

// generated adds one finished generation.
func (t *reqTrace) generated(model string, prompt, completion int) {
	t.mu.Lock()
	if !t.seen {
		t.seen, t.model, t.prompt = true, model, prompt
	}
	t.completion += completion
	t.mu.Unlock()
}

// statusRecorder remembers the status a handler wrote and still lets the handler flush (SSE) or hijack.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (w *statusRecorder) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusRecorder) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

func (w *statusRecorder) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *statusRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if h, ok := w.ResponseWriter.(http.Hijacker); ok {
		return h.Hijack()
	}
	return nil, nil, http.ErrNotSupported
}

func (w *statusRecorder) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// logRequests wraps h so each request writes one line to out when it finishes. out == nil returns h untouched: the flag off costs nothing.
func logRequests(out io.Writer, h http.HandlerFunc) http.HandlerFunc {
	if out == nil {
		return h
	}
	return func(w http.ResponseWriter, r *http.Request) {
		tr := &reqTrace{start: time.Now()}
		rec := &statusRecorder{ResponseWriter: w}
		h(rec, r.WithContext(context.WithValue(r.Context(), reqTraceKey{}, tr)))
		fmt.Fprintln(out, tr.line(r.Method+" "+r.URL.Path, rec.status, time.Now()))
	}
}

// line formats one request: "-" where a value does not exist (no generation ran, so no model or tokens; no token came, so no first-token time).
func (t *reqTrace) line(route string, status int, end time.Time) string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if status == 0 {
		status = http.StatusOK // a handler that wrote nothing answers 200
	}
	model, prompt, completion, ttft := "-", "-", "-", "-"
	if t.seen {
		model, prompt, completion = t.model, strconv.Itoa(t.prompt), strconv.Itoa(t.completion)
	}
	if !t.first.IsZero() {
		ttft = fmt.Sprintf("%dms", t.first.Sub(t.start).Milliseconds())
	}
	return fmt.Sprintf("request: %s model=%s status=%d prompt_tokens=%s completion_tokens=%s ttft=%s total=%dms",
		route, model, status, prompt, completion, ttft, end.Sub(t.start).Milliseconds())
}

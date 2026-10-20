package serveapp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
)

// -log-requests: one line per generation request on stderr: route, model, status, prompt and completion tokens, time to first
// token, total time. Off by default, so a quiet server stays quiet (docs/tasks/task-first-hour.md).
//
// The middleware owns the status and the clock; the token counts and the first-token time can only be known where tokens are
// read, which is streamTokens, the shared tail of every generation (drive and driveVL). A request that makes several
// generations (a tool loop) is counted once per request: prompt tokens of its first generation, completion tokens summed,
// first token of the first.
//
// After total= the line carries the first generation's reuse and the request's carried ids (docs/measurements/
// integration-harness-2026-10/id-roundtrip-step0-2026-10-15.md, §(e)): reused (Generation.PrefillReused), slot (the bound
// slot's record length before the request, PrefillSlot), div (the first position the prompt left that record, when it left
// it before its end, else -), canon (1 when every generation's ids are, run by run between added tokens, the tokenizer's
// own encoding of their text), ids=<used>/<carried> and ids_fallback (carried_ids.go). prefill_fallback=<path> follows when
// the batched prefill declined and another path ran the prompt (decoder.PrefillReport); the reason is on stderr. No content
// is written.
//
// --log-token-ids writes one JSON line per generation to its file, for measurement runs: time, route, response id, prompt
// length and SHA-256 (idsSHA256), reused, slot, the generated ids and canon.

type reqTrace struct {
	start time.Time
	route string
	out   io.Writer   // -log-requests, nil when off
	tl    *tokenIDLog // --log-token-ids, nil when off
	mu    sync.Mutex
	model string
	// seen is whether any generation reported in; prompt is the first one's prompt, completion the sum of all of them.
	seen               bool
	prompt, completion int
	first              time.Time
	// the first generation's reuse (reuseKnown false when its path reports none), and canon over all of them
	reuseKnown          bool
	reused, slot, match int
	canon               bool
	ids                 idsOutcome
	idsSeen             bool
	fast                bool   // the request named the fast alias (fast_alias.go)
	snapRows            int    // rows a host snapshot restored for the first generation (decoder.Generation.KVSnapRestored)
	kv                  string // the KV precision the model's cache holds (decoder.Model.KVDelivered), "" when unknown
	// prefill records a declined batched prefill and the path that ran the prompt instead (decoder.PrefillReport)
	prefill *decoder.PrefillReport
	// fromModel names the first generation's sampling fields taken from the model (genRequest.fromModel)
	fromModel []string
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

// generated adds one finished generation: gr's prompt, the ids it generated, and gen's reuse (gen is nil on the scripted path).
func (t *reqTrace) generated(lm *loadedModel, gr genRequest, gen *decoder.Generation, ids []int) {
	canon := canonicalIDs(lm, ids)
	t.mu.Lock()
	if !t.seen {
		t.seen, t.model, t.prompt, t.canon = true, lm.name, len(gr.promptIDs), true
		t.fromModel = gr.fromModel
		if gen != nil {
			t.reuseKnown, t.reused, t.slot, t.match = true, gen.PrefillReused, gen.PrefillSlot, gen.PrefillMatch
			t.snapRows = gen.KVSnapRestored
			if lm.model != nil {
				t.kv = lm.model.KVDelivered()
			}
		}
	}
	t.completion += len(ids)
	t.canon = t.canon && canon
	t.mu.Unlock()
	if t.tl != nil {
		line := map[string]any{"t": time.Now().UTC().Format(time.RFC3339Nano), "route": t.route, "id": gr.id,
			"prompt_len": len(gr.promptIDs), "prompt_sha256": idsSHA256(gr.promptIDs), "gen": ids, "canon": b2i(canon)}
		if gen != nil {
			line["reused"], line["slot"] = gen.PrefillReused, gen.PrefillSlot
		}
		if path, reason := t.prefill.Fallback(); path != "" {
			line["prefill_fallback"], line["prefill_fallback_reason"] = path, reason
		}
		t.tl.write(line)
	}
}

// carried records the request's carried-id outcome (one per request: a route renders its prompt once).
func (t *reqTrace) carried(o idsOutcome) {
	t.mu.Lock()
	t.ids, t.idsSeen = o, true
	t.mu.Unlock()
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// canonicalIDs reports whether ids are, run by run between added tokens, the tokenizer's own encoding of their text.
func canonicalIDs(lm *loadedModel, ids []int) bool {
	if lm.tk == nil {
		return true
	}
	for i := 0; i < len(ids); {
		if lm.tk.IsAdded(ids[i]) {
			i++
			continue
		}
		j := i
		for j < len(ids) && !lm.tk.IsAdded(ids[j]) {
			j++
		}
		text, err := lm.tk.DecodeContinuation(ids[i:j])
		if err != nil {
			return false
		}
		enc, err := lm.tk.EncodeSegments([]chat.Segment{{Text: text}}, false)
		if err != nil || !slices.Equal(enc, ids[i:j]) {
			return false
		}
		i = j
	}
	return true
}

// tokenIDLog is --log-token-ids' file: one JSON line per generation, appended under a lock.
type tokenIDLog struct {
	mu sync.Mutex
	f  *os.File
}

func openTokenIDLog(path string) (*tokenIDLog, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, fmt.Errorf("--log-token-ids: %w", err)
	}
	return &tokenIDLog{f: f}, nil
}

func (l *tokenIDLog) write(v map[string]any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	_, _ = l.f.Write(append(b, '\n'))
}

// closeTokenLog closes --log-token-ids' file, if open.
func (s *server) closeTokenLog() {
	if s.tokenLog != nil {
		s.tokenLog.mu.Lock()
		_ = s.tokenLog.f.Close()
		s.tokenLog.mu.Unlock()
	}
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

// logRequests wraps h so each request writes one line to out when it finishes. With out == nil it writes nothing and only
// attaches the request's prefill report (traceRequests).
func logRequests(out io.Writer, h http.HandlerFunc) http.HandlerFunc {
	return (&server{}).traceRequests(out, h)
}

// traceRequests is logRequests with the server's --log-token-ids file. With both off it only attaches a prefill report.
func (s *server) traceRequests(out io.Writer, h http.HandlerFunc) http.HandlerFunc {
	if out == nil && s.tokenLog == nil {
		// No log, but every generation still carries a prefill report: a backend announces its prefill fallback once per
		// request off it (decoder.NotePrefillFallback).
		return func(w http.ResponseWriter, r *http.Request) {
			ctx, _ := decoder.WithPrefillReport(r.Context())
			h(w, r.WithContext(ctx))
		}
	}
	return func(w http.ResponseWriter, r *http.Request) {
		tr := &reqTrace{start: time.Now(), route: r.URL.Path, out: out, tl: s.tokenLog}
		rec := &statusRecorder{ResponseWriter: w}
		ctx, rep := decoder.WithPrefillReport(context.WithValue(r.Context(), reqTraceKey{}, tr))
		tr.prefill = rep
		h(rec, r.WithContext(ctx))
		if out != nil {
			fmt.Fprintln(out, tr.line(r.Method+" "+r.URL.Path, rec.status, time.Now()))
		}
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
	reused, slot, div, canon := "-", "-", "-", "-"
	if t.reuseKnown {
		reused, slot = strconv.Itoa(t.reused), strconv.Itoa(t.slot)
		if t.match < t.slot {
			div = strconv.Itoa(t.match)
		}
	}
	if t.seen {
		canon = strconv.Itoa(b2i(t.canon))
	}
	ids := "ids=- ids_fallback=-"
	if t.idsSeen {
		ids = t.ids.fields()
	}
	line := fmt.Sprintf("request: %s model=%s status=%d prompt_tokens=%s completion_tokens=%s ttft=%s total=%dms reused=%s slot=%s div=%s canon=%s %s",
		route, model, status, prompt, completion, ttft, end.Sub(t.start).Milliseconds(), reused, slot, div, canon, ids)
	if t.kv != "" {
		line += " kv=" + t.kv // the precision the request's KV cache holds; absent when the route reports no model
	}
	if t.fast {
		line += " fast=1" // only a request that named the alias carries the field, so every other line keeps its shape
	}
	if t.snapRows > 0 {
		line += fmt.Sprintf(" snap=%d", t.snapRows) // rows a host snapshot restored; absent when none did
	}
	if len(t.fromModel) > 0 {
		line += " sampling_from_model=" + strings.Join(t.fromModel, ",") // absent when the request set every field it uses
	}
	if path, reason := t.prefill.Fallback(); path != "" {
		line += fmt.Sprintf(" prefill_fallback=%s prefill_fallback_reason=%q", path, reason) // the batched prefill declined, and why
	}
	return line
}
